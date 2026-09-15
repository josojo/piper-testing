"""Offline experiment tests: real MuJoCo, fake OpenRouter and fake CAN only."""

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nero_planner import Planner, PlanningError
from nero_experiment.__main__ import main
from nero_experiment.hardware import (
    ArmState, HardwareConfig, NeroHardware, check_calibration, certify_hardware_box,
    execute, hardware_schedule,
)
from nero_experiment.openrouter import DEFAULT_MODEL, OfflineChooser, OpenRouter, strict_json, validate_selection
from nero_experiment.workflow import DEMO_START, EXPERIMENT_LIMITS, plan_experiment, reachable_candidates
from scripts.build_nero_scene import build, DEFAULT_MODEL as URDF


class Clock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeRobot:
    """A delayed move with idle reports before it starts, and renewable packets."""

    def __init__(self, clock, q):
        self.clock, self.q = clock, list(q)
        self._parser = NS()
        self.commands, self.stop_count = [], 0
        self.target, self.command_time = None, None
        self.last_update = clock.time()
        self.fault = None
        self.missing_packet, self.stale_packet = None, None
        self.enabled, self.never_reach = True, False
        self.update()

    def packet(self, msg):
        return NS(msg=msg, timestamp=self.clock.time())

    def update(self):
        if self.target is not None and self.clock.time() - self.command_time >= 0.1 and not self.never_reach:
            dt = self.clock.time() - self.last_update
            self.q = list(np.asarray(self.q) + np.clip(np.asarray(self.target) - self.q, -0.03 * dt, 0.03 * dt))
        self.last_update = self.clock.time()
        for name, indices in (("joint_12", (1, 2)), ("joint_34", (3, 4)), ("joint_56", (5, 6)), ("joint_7", (7,))):
            packet = self.packet(NS(**{f"joint_{i}": self.q[i - 1] for i in indices}))
            if self.missing_packet == name:
                packet = None
            elif self.stale_packet == name:
                packet.timestamp -= 1
            setattr(self._parser, name, packet)
        for name in ("end_pose_xy", "end_pose_zrx", "end_pose_ryrz"):
            setattr(self._parser, name, self.packet(NS()))

    def get_arm_status(self):
        return self.packet(NS(arm_status=7 if self.fault and self.commands else 0,
                              err_code=0, ctrl_mode=1, motion_status=0))

    def get_driver_states(self, index):
        flags = dict.fromkeys(("voltage_too_low", "motor_overheating", "driver_overcurrent", "driver_overheating",
                               "collision_status", "driver_error_status", "stall_status"), False)
        return self.packet(NS(foc_status=NS(**flags, driver_enable_status=self.enabled)))

    def get_motor_states(self, index):
        return self.packet(NS(velocity=0.0))

    def get_gripper_status(self):
        flags = dict.fromkeys(("voltage_too_low", "motor_overheating", "driver_overcurrent", "driver_overheating",
                               "sensor_status", "driver_error_status", "homing_status"), False)
        return self.packet(NS(value=0.1, mode="width", foc_status=NS(**flags)))

    def get_flange_pose(self):
        return self.packet([0.0] * 6)

    def move_j(self, goal):
        self.commands.append(goal)
        self.target, self.command_time = goal, self.clock.time()

    def set_speed_percent(self, speed):
        self.speed = speed

    def electronic_emergency_stop(self):
        self.stop_count += 1


def fake_hardware(q):
    clock = Clock()
    robot = FakeRobot(clock, q)

    def sleep(seconds):
        clock.sleep(seconds)
        robot.update()

    hardware = NeroHardware(robot, robot, clock.time, clock.time, sleep)
    return hardware


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.scene = Path(cls.temp.name) / "scene.xml"
        with redirect_stdout(io.StringIO()):
            build(URDF, cls.scene)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.planner = Planner(self.scene, EXPERIMENT_LIMITS)

    def config(self):
        return HardwareConfig(self.planner.scene.fingerprint(), True, True, True, True, True, True,
                              (0.0,) * 7, (0.0,) * 3, (0.0, 0.0, 0.0, 1.0))

    def small_plan(self):
        return replace(plan_experiment(self.planner, [0, -0.01, 0, 0.01, 0, 0, 0], OfflineChooser()), model=DEFAULT_MODEL)

    def test_complete_upright_return_and_exact_original_joints(self):
        plan = plan_experiment(self.planner, DEMO_START, OfflineChooser())
        self.assertEqual(plan.outward_count, 4)
        self.assertEqual(len(plan.trajectories), 8)
        np.testing.assert_allclose(plan.trajectories[3].positions[-1], [0] * 7, atol=1e-12)
        np.testing.assert_allclose(plan.trajectories[-1].positions[-1], DEMO_START, atol=1e-12)
        for index, segment in enumerate(plan.trajectories):
            self.planner.playback(segment)
            self.assertGreaterEqual(segment.validation.minimum_clearance_m, 0.03)
        np.testing.assert_allclose(self.planner.scene.data.qpos[self.planner.scene.qadr], DEMO_START)

    def test_reachable_only_selection_and_bounded_retry(self):
        self.planner.set_start(DEMO_START)
        candidates = reachable_candidates(self.planner, [0] * 7)
        answer = OfflineChooser().choose({}, candidates)
        answer["target"]["position_m"] = [100, 0, 0]
        with self.assertRaisesRegex(PlanningError, "coordinates"):
            validate_selection(answer, candidates)
        chooser = NS(model="fake", choose=lambda *args: answer)
        with patch.object(chooser, "choose", wraps=chooser.choose) as called:
            with self.assertRaisesRegex(PlanningError, "repeatedly"):
                plan_experiment(self.planner, DEMO_START, chooser)
            self.assertEqual(called.call_count, 3)
        np.testing.assert_allclose(self.planner.scene.data.qpos[self.planner.scene.qadr], DEMO_START)

    def test_budget_and_already_upright(self):
        with self.assertRaisesRegex(PlanningError, "budget"):
            plan_experiment(self.planner, DEMO_START, OfflineChooser(), max_steps=1)
        chooser = NS(model="unused", choose=lambda *args: self.fail("No LLM needed when already upright"))
        plan = plan_experiment(self.planner, [0] * 7, chooser)
        self.assertFalse(plan.trajectories)

    def test_bad_llm_json_and_explicit_stop(self):
        for text in ('{"a": NaN}', '{"a":1,"a":2}', '{"a":Infinity}'):
            with self.assertRaises(PlanningError):
                strict_json(text)
        with self.assertRaisesRegex(RuntimeError, "declined"):
            validate_selection({"candidate_id": "stop", "target": None, "reason": "No suitable route"}, [])

    def test_openrouter_payload_and_response_guards(self):
        self.planner.set_start(DEMO_START)
        candidates = reachable_candidates(self.planner, [0] * 7)
        expected = OfflineChooser().choose({}, candidates)
        calls = []

        def transport(request, timeout):
            calls.append((request, timeout))
            return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(expected)}}]}).encode())

        client = OpenRouter(api_key="test-secret", transport=transport)
        actual = client.choose({"current_joints_rad": list(DEMO_START)}, candidates)
        self.assertEqual(actual["candidate_id"], expected["candidate_id"])
        body = json.loads(calls[0][0].data)
        self.assertEqual(body["model"], DEFAULT_MODEL)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertTrue(body["provider"]["require_parameters"])
        self.assertNotIn("test-secret", calls[0][0].data.decode())
        for finish in ("length", "content_filter"):
            client.transport = lambda *args, **kwargs: io.BytesIO(json.dumps({"choices": [{
                "finish_reason": finish, "message": {"content": "{}"}}]}).encode())
            with self.assertRaises(PlanningError):
                client.choose({}, candidates)

    def test_joint_goal_preserves_velocity_preconditions(self):
        self.planner.scene.data.qvel[0] = 0.1
        with self.assertRaisesRegex(PlanningError, "stationary"):
            self.planner.plan_joint_goal([0] * 7)
        self.assertEqual(self.planner.scene.data.qvel[0], 0.1)

    def test_box_certificate_bounds_independent_joint_motion(self):
        rng = np.random.default_rng(12)
        for center in (np.zeros(7), np.asarray(DEMO_START), np.array([0.2, -0.3, 0.1, 0.3, 0.2, 0.1, -0.2])):
            lower, upper = center - 0.006, center + 0.008
            bound = self.planner.scene.joint_box_clearance(lower, upper)
            for _ in range(100):
                q = rng.uniform(lower, upper)
                self.assertGreaterEqual(self.planner.scene.distances(q).min(), bound - 1e-9)
        with self.assertRaises(PlanningError):
            certify_hardware_box(self.planner, np.zeros(7), np.ones(7))

    def test_calibration_mismatch_and_config_validation(self):
        hardware = fake_hardware([0] * 7)
        state = hardware.read()
        with self.assertRaisesRegex(PlanningError, "disagreement"):
            check_calibration(self.planner, state, self.config())
        with self.assertRaisesRegex(PlanningError, "differs"):
            check_calibration(self.planner, state, replace(self.config(), scene_fingerprint="0" * 64))
        invalid = asdict(self.config())
        invalid["physical_estop_tested"] = "true"
        with self.assertRaisesRegex(PlanningError, "physical_estop_tested"):
            HardwareConfig.from_dict(invalid)
        HardwareConfig.from_dict(asdict(self.config()))

    def test_hardware_requires_confirmation_and_rechecks_start(self):
        plan = self.small_plan()
        hardware = fake_hardware(plan.original_joints)
        with patch("nero_experiment.hardware.check_calibration"):
            with self.assertRaisesRegex(PlanningError, "cancelled"):
                execute(self.planner, plan, hardware, self.config(), lambda prompt: False)
            self.assertFalse(hardware.robot.commands)

            def moved_during_confirmation(prompt):
                hardware.robot.q[0] += 0.01
                hardware.robot.update()
                return True

            with self.assertRaisesRegex(PlanningError, "changed"):
                execute(self.planner, plan, hardware, self.config(), moved_during_confirmation)
            self.assertFalse(hardware.robot.commands)

    def test_hardware_needs_measured_arrival_not_idle_flag(self):
        plan = self.small_plan()
        hardware = fake_hardware(plan.original_joints)
        # Idle flag is always zero, but positions change only after 0.1 s.
        # Use a tiny motion so this test's instantaneous simulated jump remains
        # below the independent estimated speed limit.
        goal = list(plan.original_joints)
        goal[1] += 0.001
        started = hardware.clock()
        state = hardware.move_and_settle(self.planner, goal, plan.original_joints)
        self.assertGreater(hardware.clock() - started, 0.7)
        np.testing.assert_allclose(state.joints_rad, goal)
        hardware.robot.never_reach = True
        goal[1] += 0.001
        with self.assertRaisesRegex(PlanningError, "timeout"):
            hardware.move_and_settle(self.planner, goal, state.joints_rad)

    def test_fault_or_interrupt_sends_stop_and_no_return(self):
        for error in (PlanningError("injected tracking failure"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                plan = self.small_plan()
                hardware = fake_hardware(plan.original_joints)
                with patch("nero_experiment.hardware.check_calibration"), patch.object(hardware, "move_and_settle", side_effect=error):
                    with self.assertRaises(type(error)):
                        execute(self.planner, plan, hardware, self.config(), lambda prompt: True)
                self.assertEqual(hardware.robot.stop_count, 1)
        hardware = fake_hardware(plan.original_joints)
        hardware.robot.fault = True
        with patch("nero_experiment.hardware.check_calibration"):
            with self.assertRaisesRegex(PlanningError, "unsafe"):
                execute(self.planner, plan, hardware, self.config(), lambda prompt: True)
        self.assertEqual(len(hardware.robot.commands), 1)
        self.assertEqual(hardware.robot.stop_count, 1)

    def test_complete_monitored_hardware_workflow_with_fake_robot(self):
        plan = self.small_plan()
        hardware = fake_hardware(plan.original_joints)
        events = []
        with patch("nero_experiment.hardware.check_calibration"):
            result = execute(self.planner, plan, hardware, self.config(), lambda prompt: True, record=events.append)
        np.testing.assert_allclose(result.joints_rad, plan.original_joints, atol=0.0005)
        self.assertGreater(len(hardware.robot.commands), 1)
        self.assertEqual(hardware.robot.speed, 3)
        self.assertEqual(hardware.robot.stop_count, 0)
        self.assertEqual(len(events), 2 * len(hardware.robot.commands))
        self.assertEqual(events[-1]["event"], "settled")

    def test_stop_delivery_failure_is_explicit(self):
        plan = self.small_plan()
        hardware = fake_hardware(plan.original_joints)
        hardware.robot.fault = True
        with patch("nero_experiment.hardware.check_calibration"), patch.object(hardware, "stop", side_effect=OSError()):
            with self.assertRaisesRegex(RuntimeError, "physical emergency stop"):
                execute(self.planner, plan, hardware, self.config(), lambda prompt: True)

    def test_state_is_refreshed_after_report_io(self):
        plan = self.small_plan()
        hardware = fake_hardware(plan.original_joints)

        def record(event):
            hardware.robot.q[0] += 0.01
            hardware.robot.update()

        with patch("nero_experiment.hardware.check_calibration"):
            with self.assertRaisesRegex(PlanningError, "before the motion command"):
                execute(self.planner, plan, hardware, self.config(), lambda prompt: True, record=record)
        self.assertFalse(hardware.robot.commands)
        self.assertEqual(hardware.robot.stop_count, 1)

    def test_runtime_tracking_and_stale_feedback_stop(self):
        for fault in ("jump", "stale"):
            plan = self.small_plan()
            hardware = fake_hardware(plan.original_joints)
            original_move = hardware.robot.move_j

            def move(goal):
                original_move(goal)
                if fault == "jump":
                    hardware.robot.q[0] += 0.05
                else:
                    hardware.robot.stale_packet = "joint_12"
                hardware.robot.update()

            hardware.robot.move_j = move
            with patch("nero_experiment.hardware.check_calibration"):
                with self.assertRaises(PlanningError):
                    execute(self.planner, plan, hardware, self.config(), lambda prompt: True)
            self.assertEqual(hardware.robot.stop_count, 1)
            self.assertEqual(len(hardware.robot.commands), 1)

    def test_tampered_plan_rejected_before_motion(self):
        plan = self.small_plan()
        bad = replace(plan.trajectories[0], positions=((0.0,) * 7,) * 2)
        with self.assertRaisesRegex(PlanningError, "modified"):
            hardware_schedule(self.planner, replace(plan, trajectories=(bad,)))

    def test_cli_offline_and_execute_flag_guards(self):
        output = Path(self.temp.name) / "report.json"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), patch.dict("sys.modules", {"pyAgxArm": None}):
            self.assertEqual(main(["run", "--offline-demo", "--scene", str(self.scene), "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["status"], "simulation_passed")
            self.assertEqual(main(["run", "--offline-demo", "--execute", "--output", str(output)]), 2)
        original = output.read_text()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "--start", str(output), "--output", str(output)]), 2)
        self.assertEqual(output.read_text(), original)


class FeedbackTests(unittest.TestCase):
    def test_fresh_complete_feedback_and_no_motion_during_capture(self):
        hardware = fake_hardware(DEMO_START)
        state = hardware.initial_state()
        self.assertEqual(state.joints_rad, DEMO_START)
        self.assertFalse(hardware.robot.commands)
        self.assertEqual(hardware.robot.stop_count, 0)

    def test_partial_and_stale_joint_groups_rejected(self):
        for name in ("joint_12", "joint_34", "joint_56", "joint_7"):
            for kind in ("missing_packet", "stale_packet"):
                with self.subTest(packet=name, kind=kind):
                    hardware = fake_hardware(DEMO_START)
                    setattr(hardware.robot, kind, name)
                    hardware.robot.update()
                    with self.assertRaises(PlanningError):
                        hardware.read()

    def test_disabled_motors_are_readable_but_cannot_move(self):
        hardware = fake_hardware(DEMO_START)
        hardware.robot.enabled = False
        hardware.read()
        with self.assertRaisesRegex(PlanningError, "enabled"):
            hardware.read(require_enabled=True)


if __name__ == "__main__":
    unittest.main()
