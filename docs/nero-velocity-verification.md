# NERO velocity feedback verification

Verified 2026-09-16 without connecting to hardware or changing motion behavior.

## Revision and decoding

Installed `pyAgxArm` distribution metadata identifies commit
`e7aef17d54cac80cbaeb1b4110ab3d8f1337a95b`, matching requirements-hardware.txt.
This verifies the local installation, not the independently built Docker image.
Our connection factory selects `NeroFW.V121`.

The vendor's NERO high-speed feedback message definition specifies CAN IDs
0x251–0x257, velocity in bytes 0–1 as signed int16, with units of 0.001 rad/s.
The inherited decoder reads big-endian bytes, converts two's-complement sign,
and multiplies by 0.001. There is no degrees conversion or velocity gear-ratio
factor in that path. Our reader uses `motor.velocity` without additional scaling.
For example, FF BE decodes to -66 * 0.001 = -0.066 rad/s.

An offline check exercised all 65,536 possible velocity encodings against
Python's signed big-endian conversion: all passed. A second check injected
synthetic python-can messages into the NERO parser for all seven CAN IDs:
joint mapping, -0.066 rad/s decoding, and timestamp propagation all passed.
No CAN bus was opened.

V121 inherits the V120 motor getter, returning decoded velocity without
zeroing or changing its sign. Older default/V111/V112 getters zero velocity;
switching to those would hide feedback, not repair it. The vendor's current
[firmware reference](https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/nero/firmware_reference.md)
confirms these version differences.

## Timing and interpretation limits

The parser updates a mutable cached message, then assigns the python-can frame
timestamp. The local SocketCAN implementation obtains a receive timestamp from
the socket ancillary data. It is not an embedded motor acquisition timestamp.
See the vendor's [parser implementation](https://github.com/agilexrobotics/pyAgxArm/blob/master/pyAgxArm/protocols/can_protocol/drivers/core/table_driven.py).

Position and motor velocity arrive in separate frames. Close receive timestamps
do not prove simultaneous internal measurement. Polling at 100 Hz can miss
intermediate frames. Our deepcopy isolates subsequent cache mutation but does
not provide synchronization with the SDK receiver while copying; coherent
value/timestamp snapshots are not guaranteed by this mechanism.

Vendor descriptions call the field motor speed. The reviewed material does not
establish its estimator bandwidth, filtering, standstill accuracy, or explicitly
resolve motor-shaft versus joint-output semantics for every NERO joint. The
absence of a decoder gear-ratio factor is not proof of either interpretation.
Do not invent a conversion or treat sampled position slopes as exact velocity.

## Implication for the recorded failures

No velocity signedness, byte order, or numerical scaling bug was found in the
reviewed software path. This does not prove the physical accuracy of the field.
The passive diagnostic records a -0.066 rad/s joint-4 sample with nearby position
changes, despite sending no commands. The moving-abort trace separately records
larger joint-2/4 excursions after the hold command. Neither is dismissed by this
audit.

The next distinguishing evidence is passive raw CAN capture alongside the
diagnostic, to compare every received velocity payload against SDK samples and
eliminate cache/sampling ambiguity. Vendor clarification is still needed on
velocity reference, estimator behavior near standstill, and supported controlled
stop behavior. No thresholds, stop logic, or execution gates were changed.
