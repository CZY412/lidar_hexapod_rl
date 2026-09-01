# Archived EA2 validation scripts

`validate_ea2_path_noise.py` (moved here 2026-08): one-off decision-support
audit for the A* path-noise mechanism. Its conclusion (keep noise OFF in
stage 1; noise mechanism well-behaved) is recorded; stage-1 config keeps
`path.noise_amp_range = [0.0, 0.0]`.

The script replicates env kinematics/sampling/collision logic in numpy, so it
silently drifts from the env. Do NOT run it before re-syncing; re-derive the
telemetry when stage 2 actually enables path noise. Planner mechanics stay
covered by `test_path_planner.py` / `test_path_batch.py`.
