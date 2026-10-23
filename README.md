# SO-ARM101 block pick and place

Minimal MuJoCo imitation learning, with [pick-101](https://github.com/ggand0/pick-101) as the reference for damped IK, locked-wrist grasping, and the approach → descend → close → lift sequence. This implementation extends that sequence through transfer, lowering, release, retreat, and settling. It reuses the existing MuJoCo Menagerie SO-ARM101 model and its finger collision geometry; no additional robot assets or RL framework.

## Run

```bash
cd so101-sim
uv sync --locked

# Load the supplied trained sensor policy and watch live physics in mjviser.
uv run python pick.py view --mode sensors

# RGB-only policy (Linux headless rendering).
MUJOCO_GL=egl uv run python pick.py view --mode camera
```

Open **http://127.0.0.1:8080**. mjviser provides orbit/zoom, pause, speed, and reset controls. Episodes repeat with randomized source and destination positions. Both camera modes also display the initial RGB policy input. Use `--port 8081` to run a second viewer, or `--host 0.0.0.0` for container port forwarding (only on a trusted network).

For headless RGB rendering, Linux needs system graphics libraries, not a GPU:

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1-mesa-dri
```

Alternatively install `libosmesa6` and use `MUJOCO_GL=osmesa` for software rendering. On a desktop with working OpenGL (including macOS), omit `MUJOCO_GL=egl`. Sensor evaluation and its mjviser viewer do not need an offscreen OpenGL context.

## Train and evaluate

```bash
# Collect 192 physics-validated demonstrations and fit BOTH checkpoints on CPU.
MUJOCO_GL=egl uv run python pick.py train --episodes 192 --seed 0

# Train the open-loop wrist policy with physical initial-view search.
MUJOCO_GL=egl uv run python pick.py train --mode wrist --episodes 192 --seed 0 --weights checkpoints/wrist_search

# Held-out randomized episodes, identical object positions for all modes.
uv run python pick.py eval --mode sensors --episodes 20 --seed 10000
MUJOCO_GL=egl uv run python pick.py eval --mode camera --episodes 20 --seed 10000
MUJOCO_GL=egl uv run python pick.py eval --mode wrist --weights checkpoints/wrist_search --episodes 20 --seed 10000
MUJOCO_GL=egl uv run python pick.py view --mode wrist --weights checkpoints/wrist_search
```

Without `--mode`, `pick.py train` writes `checkpoints/sensors.npz` and `checkpoints/camera.npz`. With `--mode sensors`, `--mode camera`, or `--mode wrist`, it trains only that policy. `--weights DIR` selects its checkpoint directory. A trained search-capable wrist policy is in `checkpoints/wrist_search/wrist.npz`; use that directory for wrist evaluation/viewing. Its calibrated-search format rejects old fixed-home wrist checkpoints with a retraining message; the supplied sensor and overhead-camera checkpoints remain compatible. Evaluation prints per-episode measurements and an aggregate JSON summary, exits nonzero if any episode fails, and accepts `--report PATH` to save full results. Training uses ground-truth poses and an IK teacher; inference uses saved fitted weights, not runtime IK.

### Separate closed-loop variant

`pick_closed_loop.py` is the feedback-control version; it does not change
`pick.py` into a closed-loop controller.

```bash
# Fit a separate Cartesian-to-joint controller for each observation mode.
uv run python pick_closed_loop.py train --episodes 192 --seed 0

# Sensors remain sensor-only: no RGB rendering is used by this policy.
uv run python pick_closed_loop.py eval --mode sensors --episodes 20 --seed 10000
MUJOCO_GL=egl uv run python pick_closed_loop.py eval --mode camera --episodes 20 --seed 10000
MUJOCO_GL=egl uv run python pick_closed_loop.py eval --mode wrist --episodes 20 --seed 10000

MUJOCO_GL=egl uv run python pick_closed_loop.py view --mode wrist
```

Checkpoints live in `checkpoints/closed_loop/`, separately from the open-loop
weights. `--weights DIR` overrides that directory; `train --mode MODE` fits only
one mode. Checkpoint contracts are versioned and reject open-loop weights.
IK runs during training only: a supervised polynomial maps reachable Cartesian
targets to actuator positions during execution.

The controller consumes **fresh observations at 20 Hz**, updating the commanded
joint targets while MuJoCo advances at 200 Hz. Sensor mode uses current block and
destination position sensors, never images. Overhead and wrist modes use current
256×256 RGB, joint encoders, robot kinematics, and known camera/object calibration;
they do not read simulated object poses, depth, segmentation, or scoring contacts.
Known table and object geometry resolve image scale and moving-camera perspective.
The live camera viewer displays the latest consumed RGB image.

Feedback tracks the source during approach/descent and the destination before
placement/release. After closing begins, the source is deliberately held in
controller memory: projecting a carried block back onto the table would steer
the arm incorrectly. Near final approach/descent and placement/release, at most
2.2 seconds of location memory is allowed only when current RGB identifies
robot/held-block self-occlusion and joint feedback places the tool near that
location. Other required-view loss pauses phase progress; recovery raises and,
when necessary, sidesteps the tool to reacquire a view. The 35-second episode
timeout reports failure rather than blindly completing. This remains a staged,
bounded pick-and-place controller, not a general visual manipulation policy or
a claim of recovery after a dropped block.

Current closed-loop smoke checks passed two randomized nominal episodes per mode
(seed 0). A separate physical rollout moved the block +18 mm in X after motion
began and moved the destination +15 mm in both X and Y during lift:

| Closed-loop mode | Final placement error | Fresh observations consumed |
|---|---:|---:|
| Sensors | 4.34 mm | 250 |
| Overhead camera | 11.99 mm | 291 |
| Wrist camera | 3.79 mm | 250 |

All three met the unchanged grip/lift/release/settle success criterion. The
overhead result is close to the 12 mm placement tolerance; these limited smoke
checks are not a robustness guarantee. Fully hidden target motion during a
guarded self-occlusion interval cannot be measured until it becomes visible.

### Save evaluations and replay them

```bash
# Saving is optional; each command creates a randomly named run subdirectory.
uv run python pick.py eval --mode sensors --episodes 20 --save-dir runs/eval
MUJOCO_GL=egl uv run python pick.py eval --mode camera --episodes 20 --save-dir runs/eval
MUJOCO_GL=egl uv run python pick.py eval --mode wrist --weights checkpoints/wrist_search --episodes 20 --save-dir runs/eval

# Use the run directory printed by evaluation (brave_otter is an example).
uv run python pick.py replay --recording runs/eval/brave_otter/camera-10000-0000.npz

# Closed-loop recording: use one episode initially because every frame is saved.
MUJOCO_GL=egl uv run python pick_closed_loop.py eval --mode wrist --episodes 1 --save-dir runs/closed_loop

# Substitute the run directory printed by the closed-loop evaluation.
uv run python pick_closed_loop.py replay --recording runs/closed_loop/brave_otter/wrist-10000-0000.npz
```

Open **http://127.0.0.1:8080**. Pause, single-step, speed controls, and Reset work
in mjviser and keep the recorded RGB image synchronized with the 3D state.
Playback holds both the final pose and image; Reset returns both to the initial
frame. Use `--port` / `--host` as with live viewing.

Both scripts use the same replay format and viewer. Closed-loop `eval --save-dir`
works with `sensors`, `camera`, and `wrist`; failed evaluations are recorded too,
even when evaluation exits nonzero. `replay --recording` needs neither `--mode`
nor trained weights, does not rerun the controller, and needs no offscreen
OpenGL context, including when playing stored RGB. `--save-dir` is evaluation-only.
Closed-loop recording metadata identifies `controller: "closed_loop"` and includes
the result, final phase, observation/control counts, and actual episode duration.

Recordings contain the initial state and every physics step, including joint and
block poses, velocities, actuator controls, simulation time, and the randomized
destination pose, plus the timestep and evaluation result. Both overhead- and
wrist-camera policy recordings include a 256×256 uint8 RGB image from the policy's
camera for **every saved state**, stored as `rgb_frames`
with shape `(number_of_states, 256, 256, 3)`. This includes the initial and final
frames. Open-loop execution normally saves 2,501 images for a 12.5-second episode
at a 5 ms timestep. Closed-loop episodes save however many states actually ran,
including recovery pauses or timeout; they are not truncated to the nominal horizon.

mjviser displays the stream under **Recorded camera RGB**, selecting the image
at the same recorded index as the 3D scene. The browser refresh rate may skip
intermediate images at normal speed; slow playback or single-step to inspect
individual saved frames. `pick.py` still executes one frozen plan after initial
observation/search. Sensor recordings do not capture RGB.

Capturing every frame substantially increases recording time and storage.
One default open-loop episode contains about 469 MiB of uncompressed image data;
NPZ compresses it on disk. Closed-loop recording uses a temporary disk-backed RGB
buffer in the run directory and removes it after saving; sufficient temporary disk
space is required for the episode. Replay loads one episode's frames into memory,
so longer closed-loop recovery episodes require correspondingly more RAM.

The recording format is now version 2. Re-run evaluation to replace older
state-only/initial-image version-1 recordings; those cannot provide a complete
RGB stream and are rejected by the new loader. Recordings contain neither
checkpoints nor robot meshes. Replay uses the repository's scene and assets with
the same MuJoCo version and needs neither trained weights nor offscreen OpenGL.

Each saved evaluation run gets a random Docker-style adjective–noun directory,
such as `runs/eval/brave_otter/`. All episodes from that invocation go inside it,
named `<mode>-<seed>-<episode:04d>.npz`. The run name is printed at startup, stored
as `run_name` in each recording's metadata, and its path appears as `run_dir` in
the evaluation summary and JSON report.

Names are independent of the evaluation seed: repeating a seeded evaluation
preserves its simulated trajectories but creates a new directory. Existing names
are skipped atomically, so previous runs are not overwritten. There are 1,024
adjective–noun combinations per save directory; if all are occupied, choose a
different `--save-dir`.

Without `--save-dir`, no run directory, trajectory buffers, or recording files
are created. `--report` remains an independent JSON metrics export at the path
you specify.

### Measured checkpoint results

The supplied open-loop policies were fitted on 192 successful demonstrations
(seed 0). On 20 held-out episodes per policy
(seed 10000, MuJoCo 3.16.0):

| Policy | Successful pick-and-place episodes | Mean placement error |
|---|---:|---:|
| Sensors | 20 / 20 | 3.62 mm |
| Camera-only | 20 / 20 | 3.86 mm |
| Wrist camera with initial-view search | 20 / 20 | 3.67 mm |

RGB evaluation used CPU OSMesa rendering. The supplied sensor and overhead policies'
live mjviser surfaces and pause/reset controls were exercised in a browser. An open-gripper negative
control correctly failed the lift-and-place criterion.

### Model and observation contract

- **Sensors:** initial block and target XY coordinates from MuJoCo `framepos` sensors. These are ideal simulated position sensors, not a claim that the physical arm measures object poses.
- **Overhead camera (`--mode camera`):** a single initial 256×256 RGB image from a fixed overhead camera. Red/blue color centroids feed a learned mapping directly to joint trajectories. No joint encoders, object coordinates, depth, segmentation IDs, camera calibration, or contact signals enter this policy.
- **Wrist camera (`pick.py --mode wrist`):** physically sweeps bounded elevated pan/roll poses until both colored objects have a qualified view. Requires at least 80 pixels per object, an 8-pixel image margin, at least 45% bounding-box occupancy, and a silhouette fit within 3 pixels. Known camera intrinsics, encoder-derived extrinsics, and block/pad dimensions compensate for the selected camera pose; a learned polynomial maps the RGB-derived locations to joint waypoints. The scene's true object positions are not inputs to this policy. Search failure or invalid predicted joint limits holds instead of executing a blind plan.
- All three `pick.py` policies are **supervised polynomial movement-primitive policies**, not neural networks or SAC/DrQ-v2 agents. Each fits 19 cubic features to nine six-joint waypoints from successful demonstrations. A fixed smooth interpolator executes these learned targets through the robot's position actuators.
- `pick.py` policies are **open-loop after their initial observation/search**, apart from MuJoCo's actuator servo dynamics. Wrist search moves the camera using physics before committing; task timing/scoring starts at the selected pose without teleporting the arm. Recording starts at that execution pose, not during search. After commitment, policies do not recover from disturbances, occlusion, or moved targets. Contact and object-state measurements during rollout are for scoring only.
- The block is moved by frictional two-finger contact and gravity. No weld, attachment, body teleportation, or pose correction occurs during rollout. Scene reset alone initializes object and robot positions.

### Task bounds and success

One axis-aligned, red **30 mm / 25 g block per episode**, placed on a blue destination region. New block positions are sampled on every reset; this is repeated block transfer, not simultaneous multi-object sorting or stacking.

| Position | X range | Y range |
|---|---|---|
| Block | 0.21–0.25 m | −0.075 to −0.035 m |
| Destination | 0.21–0.25 m | 0.045–0.085 m |

A successful episode must hold the block above 65 mm with contact on **both fingers for at least 0.5 seconds**, then leave it within 12 mm of the destination, at table height, released and stationary for at least 0.5 seconds after retreat. The open-loop execution runs 12.5 simulated seconds, excluding wrist initial-view search. The closed-loop variant pauses for required perception and has a bounded recovery timeout. This excludes merely pushing the block into the target or dropping it nearby.

The camera mounting and optics, colors, lighting, cube geometry, home pose, and bounded workspace are part of the training distribution. Retrain after changing them. No real-arm deployment or sim-to-real reliability is claimed; hardware requires calibration, perception, collision limits, and independent safety validation.

## Files

- `so101-sim/pick.py`: environment, teacher, training, learned policies, evaluation, mjviser integration.
- `so101-sim/pick_closed_loop.py`: separate trained feedback controller, live RGB/sensor observation, training, evaluation, recording, replay, and live viewing.
- `so101-sim/pick.xml`: block, target, camera, and position sensors; includes the existing robot model.
- `so101-sim/checkpoints/`: separate open-loop and closed-loop trained parameters, loaded without pickle.

Only NumPy, MuJoCo, mjviser, and viser are used by this implementation; all were already present directly or transitively. The existing MJX/JAX/Warp examples remain available:

```bash
uv run python sim_native.py
uv run python sim_mjx.py
uv run python sim_warp.py
# Linux CUDA setup:
uv run --extra cuda python sim_mjx.py
# Inspect the original robot without a policy:
uv run mjviser mujoco_menagerie/robotstudio_so101/scene.xml
```
