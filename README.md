# Astra-to-Nero MuJoCo Control Stack

This repository is a starting point for controlling an AgileX NERO arm from a vision-capable model such as Astra.

The intended architecture is:

```text
Astra / vision
        ↓
desired end-effector pose
        ↓
Python target planner
        ↓
MuJoCo
  - inverse kinematics
  - trajectory generation
  - collision checking
  - joint-limit checking
        ↓
validated joint trajectory
        ↓
pyAgxArm / NERO
        ↓
real robot
```

MuJoCo is used as the local robot model, geometry engine, trajectory validator, and eventually the simulation environment. `pyAgxArm` remains the hardware backend for the real NERO.

## Design goals

- Keep the high-level control loop in Python.
- Treat the vision model as a target selector, not as a motor controller.
- Validate proposed motion before sending it to the real arm.
- Start with discrete Cartesian target updates and visual re-checks.
- Keep the real-hardware safety checks independent of the simulator.
- Make it possible to test planning and collision behavior without powering the arm.

## Components

### 1. Astra / vision

Astra receives camera images and task context and proposes a desired end-effector pose.

The output should be a structured target, not free-form text:

```json
{
  "frame": "nero_base",
  "position_m": [0.35, -0.15, 0.20],
  "orientation_xyzw": [0.0, 1.0, 0.0, 0.0],
  "gripper": 1.0,
  "reason": "Move above the red block before lowering"
}
```

The target is expressed in a known robot coordinate frame. If Astra initially returns image pixels, a separate perception/calibration stage must convert pixels into robot-frame coordinates before this interface.

Astra should not send joint angles or CAN commands. It should make relatively infrequent, deliberate decisions such as:

```text
observe → choose target pose → execute a short move → observe again
```

### 2. Python target planner

The target planner is the application-level layer between Astra and MuJoCo. It is responsible for:

- validating the target schema;
- converting poses between coordinate frames;
- limiting target-step size;
- choosing approach, grasp, lift, and retreat waypoints;
- selecting an IK seed from the measured robot state;
- requesting one or more candidate trajectories;
- rejecting targets that are clearly outside the workspace;
- sending only validated trajectories to the hardware adapter.

The planner should maintain two kinds of state:

```text
real_state:
  measured NERO joint positions, status, gripper state, errors

simulation_state:
  MuJoCo state synchronized from real_state and used for planning
```

The simulation state is a planning model. It is not assumed to be a perfect copy of the physical arm.

### 3. MuJoCo model

The MuJoCo model should contain:

- the seven NERO joints;
- joint limits and nominal velocity limits;
- link frames and inertial parameters;
- a named end-effector site;
- simplified collision geometries for the links and gripper;
- the table/platform;
- known fixed obstacles;
- optional cameras and lighting for synthetic observations.

Use a separate collision model rather than relying on detailed visual meshes. Boxes, capsules, cylinders, and convex meshes are easier to debug and usually better for fast collision checks.

MuJoCo's native model format is MJCF. Its Python bindings expose forward kinematics, contacts, joint state, actuator state, and simulation stepping. Collision detection produces contacts in `mjData.contact`; the number of active contacts is available through `data.ncon`.

MuJoCo is a physics and contact engine, not a complete motion planner. The project must still choose or implement the IK and path-generation method.

### 4. Inverse kinematics

Given a desired end-effector pose, IK finds a joint configuration that reaches it:

```text
desired pose → q_target[0:7]
```

The first implementation should use a differential or numerical IK solver with:

- the current measured joint configuration as the seed;
- joint-position limits;
- optional joint-velocity limits;
- position and orientation tolerances;
- a penalty for large joint motion;
- collision avoidance where available.

Potential Python options include:

- the NERO SDK's Cartesian/IK functions for a minimal hardware-oriented prototype;
- [`mink`](https://github.com/kevinzakka/mink), a MuJoCo-based differential IK library;
- [`cuRobo`](https://curobo.org/) later, if GPU-accelerated IK and trajectory optimization become useful.

IK success is not the same as motion-planning success. A valid final pose may still require moving through a collision.

### 5. Trajectory generation

The planner must produce a time-ordered sequence of joint configurations:

```text
q_start → q_1 → q_2 → ... → q_goal
```

The simplest first version is a joint-space interpolation. Every interpolated point must be checked, not only the start and goal:

```python
for alpha in np.linspace(0.0, 1.0, num_steps):
    q = (1.0 - alpha) * q_start + alpha * q_goal
    check_joint_limits(q)
    check_collisions(q)
```

For cluttered scenes, replace straight interpolation with a real planner such as RRT-Connect through [OMPL](https://ompl.kavrakilab.org/), or use trajectory optimization. OMPL supplies planning algorithms but intentionally leaves collision checking and robot modeling to the application; MuJoCo can provide those checks.

The real arm should receive modest, bounded moves. Long open-loop trajectories are inappropriate while the vision model is still learning the camera-to-robot relationship.

### 6. Collision checking

For each candidate configuration:

1. Copy the candidate joint positions into MuJoCo's `qpos`.
2. Set velocities to zero for a static configuration check.
3. Run forward kinematics and collision detection.
4. Inspect active contacts and minimum clearance.
5. Reject self-collisions, table collisions, forbidden obstacle contacts, and configurations too close to hazards.

Conceptual implementation:

```python
def is_safe_configuration(model, data, q, minimum_clearance_m=0.03):
    data.qpos[:7] = q
    data.qvel[:] = 0.0

    mujoco.mj_forward(model, data)

    for joint_id in range(model.njnt):
        address = model.jnt_qposadr[joint_id]
        lower, upper = model.jnt_range[joint_id]
        if not lower <= data.qpos[address] <= upper:
            return False

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if contact.dist < minimum_clearance_m:
            return False

    return True
```

The production implementation should classify contact pairs by geom name or ID. Not every contact is necessarily forbidden: for example, gripper fingers may intentionally contact a grasped object. The allowed-contact policy must be explicit.

MuJoCo's collision result is only as accurate as the model and collision geometry. Add conservative margins around the table, obstacles, and robot links. Standard MuJoCo collision detection primarily operates on convex geometries, so complex meshes may need decomposition or simplified proxies.

### 7. Validated joint trajectory

The output of the MuJoCo stage should be a typed, validated object rather than a bare list:

```text
ValidatedTrajectory
  joint_names
  positions[N, 7]
  timestamps[N]
  source_target_pose
  minimum_clearance
  ik_error
  validation_status
```

Validation should include:

- joint-position limits;
- joint-velocity and acceleration limits;
- collision and clearance checks;
- end-effector final-pose error;
- maximum joint displacement;
- expected execution duration;
- a valid starting state matching the measured arm state.

The trajectory should be rejected if the real arm has moved materially since the planning state was captured. Re-synchronize and plan again instead.

### 8. pyAgxArm / NERO adapter

The adapter is the only component that should issue real NERO motion commands.

It is responsible for:

- CAN connection and configuration;
- reading measured joint positions;
- checking arm status and error codes;
- checking that all joints are enabled;
- sending joint or Cartesian commands;
- limiting speed;
- monitoring execution;
- stopping or disabling the arm on failure.

The current repository already contains direct `pyAgxArm` connection and safety helpers in [`nero_safety_common.py`](nero_safety_common.py) and [`control_nero.py`](control_nero.py). These should remain below the planner. The planner must not bypass the adapter to access CAN directly.

The simulator may approve a motion, but the adapter must perform a final real-hardware validation immediately before execution.

### 9. Real robot feedback

After execution, read the achieved joint state and end-effector pose. Do not assume the arm reached the commanded target.

The next vision observation should be based on:

```text
actual measured state + new camera images
```

This is important when the arm lags, encounters contact, reaches a limit, or is stopped by a safety condition.

## Suggested execution loop

```python
while task_is_active:
    observation = cameras.capture()
    robot_state = nero.read_state()

    target = astra.choose_target(observation, robot_state)
    target = planner.normalize_target(target)

    planner.sync_mujoco(robot_state)
    candidates = planner.solve_ik_and_generate_candidates(target)

    trajectory = planner.select_safe_trajectory(candidates)
    if trajectory is None:
        report_failure("No safe trajectory found")
        break

    nero.execute_if_still_safe(trajectory)
    execution_result = nero.wait_for_completion()

    if not execution_result.success:
        report_failure(execution_result.reason)
        break
```

The first version should execute one short target motion per vision observation. Continuous visual servoing can be added later as a separate controller.

## Piper and NERO MuJoCo model availability

### AgileX Piper

AgileX Piper already has several MuJoCo models available:

- [Google DeepMind MuJoCo Menagerie: `agilex_piper`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper)
- [Community `Piper_mujoco` model](https://github.com/soulde/Piper_mujoco)
- [AgileX Piper MuJoCo + `ros2_control` package](https://github.com/renesas-rdk/agilex_piper_mujoco)

The Menagerie model is the best starting point for a maintained reference model. The community packages are useful examples of controllers and integration, but their geometry, joint conventions, actuator parameters, and licenses should be checked before using them as a hardware-accurate source.

### AgileX NERO

NERO is a different robot: it has seven degrees of freedom, while Piper has six. I did not find an official NERO MuJoCo model in the public MuJoCo Menagerie or an obvious official AgileX NERO MJCF package.

AgileX does provide NERO support through its Python SDK and ROS drivers. The likely NERO simulation path is therefore:

1. Obtain the official NERO URDF and meshes from the AgileX driver/package.
2. Verify joint names, axes, limits, base frame, flange frame, and gripper frame against the physical robot.
3. Convert or recreate the model in MJCF.
4. Start with simplified collision geometries.
5. Calibrate the model against measured NERO forward-kinematics poses.
6. Validate every simulated trajectory with conservative real-world margins.

Piper is useful for learning the MuJoCo tooling, but its model cannot be used directly for NERO planning because the kinematic chain and joint count differ.

## Recommended implementation phases

### Phase 1: offline model validation

- Add the NERO MJCF model.
- Load it in the MuJoCo viewer.
- Verify the joint order and zero pose.
- Verify forward-kinematics end-effector positions against the SDK.
- Add the table and basic collision geometry.

### Phase 2: manual target planning

- Use a hard-coded end-effector target.
- Solve IK from a measured starting configuration.
- Generate a short joint-space trajectory.
- Validate joint limits and collision clearance.
- Visualize the trajectory without hardware.

### Phase 3: hardware dry run

- Read the real NERO state.
- Synchronize MuJoCo to that state.
- Plan a small motion.
- Require an explicit human confirmation.
- Execute at low speed.
- Compare measured and simulated motion.

### Phase 4: closed-loop vision

- Add camera calibration and frame transforms.
- Have Astra return structured target poses.
- Re-plan after every short movement.
- Add grasp/lift/place waypoints.
- Add object and obstacle updates.

### Phase 5: advanced planning

- Add OMPL, trajectory optimization, or cuRobo if straight-line interpolation is insufficient.
- Add force/contact checks.
- Add synthetic camera testing.
- Add randomized simulation tests for model and perception error.

## Safety boundary

MuJoCo is a planning and validation tool, not a substitute for physical safety systems.

The real system must retain:

- physical emergency stop;
- conservative speed limits;
- workspace and table-clearance checks;
- arm-status and error monitoring;
- joint-enable checks;
- execution timeout;
- immediate stop/disable behavior;
- human supervision during initial tests.

The simulator should make dangerous motions less likely. It cannot guarantee that the physical arm will never collide, especially when the model, calibration, payload, tool, or environment is inaccurate.

## Current repository status

The repository contains direct Python/CAN control and safety scripts, an official-URDF-derived NERO MuJoCo scene, and a simulation-only pose executor in `nero_planner`. The executor provides numerical IK, timed joint interpolation, conservative full-path collision/clearance validation, and kinematic playback. Camera/Astra integration and a hardware trajectory adapter are not implemented.

The implemented milestone is a simulation-only target executor:

```text
hard-coded Pose
→ MuJoCo IK
→ trajectory generation
→ collision/joint-limit validation
→ visualization
```

The simulation executor does not import `pyAgxArm` or connect to CAN. Its validation is relative to the proxy model and documented contact exclusions; it is not hardware clearance certification.

## Preparing the MuJoCo environment

The official NERO source is [AgileX's `agx_arm_urdf` repository](https://github.com/agilexrobotics/agx_arm_urdf). It provides the NERO URDF/Xacro files and meshes. The repository is fetched into an ignored directory; no vendor copy is committed here.

Create an isolated environment and install the simulation dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mujoco.txt
```

Fetch the official NERO URDF plus its Piper-style gripper and rewrite mesh paths for standalone MuJoCo loading:

```bash
python scripts/prepare_nero_mujoco.py
python scripts/check_mujoco_model.py
```

The checker must report a loaded model and the NERO plus gripper joint list. The default preparation merges the official gripper files directly. If you prefer ROS Xacro expansion, install the ROS `xacro` command and run:

```bash
python scripts/prepare_nero_mujoco.py --xacro
python scripts/check_mujoco_model.py
```

Build and view the first interactive scene with a table and one position actuator per NERO joint:

```bash
python scripts/build_nero_scene.py
python -m mujoco.viewer --mjcf=models/nero/nero_scene.xml
```

The actuators are conservative position actuators for simulation inspection only; they are not connected to `pyAgxArm` and cannot move the physical arm. The scene includes the official Piper-style gripper and a fixed red apple on the table. Gravity is disabled and detailed meshes are visual-only. Padded boxes provide planning collision geometry; the vendor dynamics are not calibrated. The raw viewer's control sliders are for inspection, not validated execution.

## Running the simulation-only executor

After preparing the URDF, rebuild the scene to add the planning geometry and tool site:

```bash
source .venv/bin/activate
python scripts/build_nero_scene.py
python -m nero_planner --demo
python -m nero_planner --target examples/reach.json --output /tmp/nero-trajectory.json
python -m nero_planner --demo --viewer
```

On macOS, launch viewer playback with MuJoCo's `mjpython` launcher instead of `python`:

```bash
mjpython -m nero_planner --demo --viewer
# Or explicitly use the virtual environment's launcher:
.venv/bin/mjpython -m nero_planner --demo --viewer
```

MuJoCo installs `mjpython` with its macOS package. It satisfies the main-thread rendering requirement of `launch_passive` ([MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)). Model preparation and headless execution still use ordinary `python`; no scene rebuild is needed for this launcher error.

If the executor reports `Required model element is missing: gripper`, your generated scene is outdated or was prepared without the gripper. Generated models are Git-ignored, so pulling new code does not update them. Run `python scripts/prepare_nero_mujoco.py` without `--no-gripper`, then `python scripts/build_nero_scene.py`, and retry the executor. The error includes the exact scene path being loaded; a custom `--scene` must also be regenerated.

Without `--viewer`, execution is headless kinematic playback. With it, the viewer replays the same quintic joint path in real time and closes at completion. Neither mode steps actuator dynamics. Successful commands print a JSON summary with the achieved pose and validation metrics. Rejected requests print a JSON reason to stderr and exit with status 2. `--output` writes the complete trajectory and achieved pose after successful execution; this report is not an executable hardware command or an importable validation token.

`--start path.json` supplies a JSON array of seven joint angles in radians, ordered `joint1` through `joint7`. The default is the all-zero configuration, with all three gripper joints explicitly synchronized to fully open. The demo moves the grasp center 2 cm in base +X and 2 mm in base -Z, preserving orientation. `examples/reach.json` is an equivalent fixed target for the default starting pose.

The target interface accepts `frame`, `position_m`, `orientation_xyzw`, optional `gripper`, and optional text `reason`. Only `frame="nero_base"` and `gripper=1.0` are supported in this milestone. A quaternion must be finite and unit length within 0.001; small roundoff is normalized, and opposite quaternion signs are equivalent. Unknown fields, unsupported frames, nonfinite numbers, and gripper movement requests are rejected.

The controlled `grasp_center` site is the midpoint of the two finger joint origins at zero opening, using the gripper-base orientation (+Z approach direction). It lies 0.138 m along the vendor gripper-base +Z axis. The fixed transform is derived from the compiled URDF, including the flange attachment, and the site is attached to `link7`. In the zero arm pose it is approximately `[0, 0, 0.89301]` metres in `nero_base`. This convention is a model tool frame, not a measured physical TCP calibration.

### Planning and validation defaults

`--limits path.json` accepts a JSON object overriding fields of `nero_planner.Limits`. Defaults are:

| Setting | Default |
| --- | --- |
| Forbidden-pair clearance | 0.03 m |
| Maximum joint velocity | 0.2 rad/s |
| Maximum joint acceleration | 0.5 rad/s² |
| Maximum per-joint displacement per request | 0.25 rad |
| Maximum target translation per request | 0.05 m |
| Final position/orientation tolerances | 0.002 m / 2° |
| Maximum execution duration | 15 s |
| Playback sample period | 0.02 s |
| Maximum clearance subdivision depth | 16 |
| Maximum validation samples / trajectory samples | 20,000 each |
| Maximum IK iterations | 400 |

These are simulation defaults, not calibrated NERO hardware limits. For example, `{"max_velocity_rad_s": 0.1}` halves the allowed velocity. Joint-position bounds come from the imported model. The planner uses the current simulated joint configuration as the numerical IK seed and applies damping, joint bounds, and a small posture preference. Failure to converge within bounded joint motion is rejected; it does not search alternative routes around obstacles.

The joint path uses `s(u) = 10u³ - 15u⁴ + 6u⁵`, with duration chosen from the exact peak velocity and acceleration of that polynomial. Every configuration stays between its endpoints in joint space. A `ValidatedTrajectory` carries immutable tuples of joint names, positions, timestamps, target pose, captured starting state, validation metrics, and limits. Planning restores the original simulator state. Playback accepts only an unmodified trajectory issued by that planner and rechecks the complete starting joint state, zero velocity, scene fingerprint, and limits. Load a new planner after editing a scene; re-plan after the arm state changes.

### Collision geometry and explicit exclusions

Every imported robot mesh is enclosed by an oriented box padded by 3 mm on each side. A box also encloses the fixed apple and stem with 3 mm padding. The table top remains at `z=0`. The viewer executor hides proxy group 3 by default; enable that group in the viewer to inspect the boxes. Tests verify that the boxes contain every mesh vertex at multiple arm poses.

Validation queries MuJoCo forward kinematics and evaluates all forbidden box pairs, including separated pairs without contacts. It uses the 15 separating axes for oriented boxes to obtain a conservative distance lower bound. This avoids the documented positive-distance limitations of some colliders in older MuJoCo releases ([MuJoCo 3.2 API reference](https://mujoco.readthedocs.io/en/3.2.5/APIreference/APIfunctions.html#mj-geomdistance)). The reported clearance is a certified lower bound, not an exact closest-point distance.

At each interval midpoint, validation subtracts a conservative bound on both geometries' possible motion over the interval. If the remaining distance exceeds the configured clearance, the entire interval is certified. Otherwise it bisects the joint interval, rejecting on a clearance violation or an exhausted depth/sample budget. Checking only endpoints or `data.ncon` is insufficient.

The explicit structural exclusions in `nero_planner/model.py` are:

- Geometries rigidly attached to the same robot body.
- Adjacent arm-body pairs (`world/link1`, `link1/link2`, through `link6/link7`) and gripper housing/finger pairs (`link7/gripper_link1`, `link7/gripper_link2`).
- The compact shoulder assembly `world/link2` and wrist assembly `link5/link7`, whose enclosing boxes overlap across an intermediate joint body.
- The fixed base proxy against the table and tabletop; environment/environment contacts are also outside robot-motion validation.

All other robot/table, robot/apple, and robot/robot pairs must maintain clearance, including finger/finger pairs. Additional fixed box obstacles with collision enabled are checked too. Unsupported non-box collision shapes or movable obstacles are rejected at load time. **Excluded assembly pairs are not collision-checked**, even if a configuration could produce a real collision within that assembly. These explicit model limitations must be reviewed, and proxy geometry refined as necessary, before any hardware use. No grasp contacts are permitted.

### Offline acceptance tests

```bash
python -m unittest discover -s tests -v
```

This builds an isolated scene from the prepared URDF and tests successful reaching, mesh enclosure, tool/gripper conventions, table/apple/self-collision rejection, near misses without forbidden contacts, unsafe path interiors with safe endpoints, validation-budget exhaustion, pose/step/timing limits, malformed inputs, stale states/scenes, and modified trajectory rejection. It requires neither CAN nor a display. The root-level `test_nero.py`, `test_backend.py`, and related scripts are existing hardware utilities and are not part of this offline suite.

Remaining milestones are hardware FK/TCP calibration, refined assembly collision models, dynamic tracking validation, supervised hardware integration, camera calibration, and Astra target selection.
