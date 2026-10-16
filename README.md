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

Open **http://127.0.0.1:8080**. mjviser provides orbit/zoom, pause, speed, and reset controls. Episodes repeat with randomized source and destination positions. The camera version also displays the initial RGB policy input. Use `--port 8081` to run a second viewer, or `--host 0.0.0.0` for container port forwarding (only on a trusted network).

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

# Held-out randomized episodes, identical scenes for both modes.
uv run python pick.py eval --mode sensors --episodes 20 --seed 10000
MUJOCO_GL=egl uv run python pick.py eval --mode camera --episodes 20 --seed 10000
```

`train` writes `checkpoints/sensors.npz` and `checkpoints/camera.npz`; `--weights DIR` selects another directory. Evaluation prints per-episode measurements and an aggregate JSON summary, exits nonzero if any episode fails, and accepts `--report PATH` to save full results. Training uses ground-truth poses and an IK teacher; inference uses the saved fitted weights, not IK or a scripted pose lookup.

### Save evaluations and replay them

```bash
# Saving is optional; one compressed NPZ per episode, including failures.
uv run python pick.py eval --mode sensors --episodes 20 --save-dir runs/eval
MUJOCO_GL=egl uv run python pick.py eval --mode camera --episodes 20 --save-dir runs/eval

# Replay the actual recorded motion, not a fresh policy rollout.
uv run python pick.py replay --recording runs/eval/camera-10000-0000.npz
```

Open **http://127.0.0.1:8080**. Pause, single-step, speed controls, and Reset work
in mjviser. Playback holds the final frame; Reset returns to the initial frame.
Use `--port` / `--host` as with live viewing.

Recordings contain the initial state and every physics step, including joint and
block poses, velocities, actuator controls, simulation time, and the randomized
destination pose, plus the timestep and evaluation result. They do not contain
camera video, checkpoints, or robot meshes. Replay uses the repository's scene
and assets with the same MuJoCo version; it requires neither trained weights nor
offscreen OpenGL, even for camera-policy recordings.

Files are named `<mode>-<seed>-<episode:04d>.npz`. Repeating the same evaluation
in the same directory overwrites matching filenames. Without `--save-dir`, no
trajectory buffers or recording files are created. `--report` remains an
independent JSON metrics export.

### Measured checkpoint results

Both supplied policies were fitted on 192 successful demonstrations (seed 0).
On 20 held-out episodes per policy (seed 10000, MuJoCo 3.16.0):

| Policy | Successful pick-and-place episodes | Mean placement error |
|---|---:|---:|
| Sensors | 20 / 20 | 3.62 mm |
| Camera-only | 20 / 20 | 3.86 mm |

RGB evaluation used CPU OSMesa rendering. Both live mjviser surfaces and their
pause/reset controls were exercised in a browser. An open-gripper negative
control correctly failed the lift-and-place criterion.

### Model and observation contract

- **Sensors:** initial block and target XY coordinates from MuJoCo `framepos` sensors. These are ideal simulated position sensors, not a claim that the physical arm measures object poses.
- **Camera-only:** a single initial 256×256 RGB image from a fixed overhead camera. Red/blue color centroids feed a learned mapping directly to joint trajectories. No joint encoders, object coordinates, depth, segmentation IDs, camera calibration, or contact signals enter this policy.
- Both are **supervised polynomial movement-primitive policies**, not neural networks or SAC/DrQ-v2 agents. Each fits 19 cubic features to nine six-joint waypoints from successful demonstrations. A fixed smooth interpolator executes these learned targets through the robot's position actuators.
- Policies are **open-loop after their initial observation**, apart from MuJoCo's actuator servo dynamics. They do not recover from disturbances, occlusion, or moved targets. Contact and object-state measurements during rollout are for scoring only.
- The block is moved by frictional two-finger contact and gravity. No weld, attachment, body teleportation, or pose correction occurs during rollout. Scene reset alone initializes object and robot positions.

### Task bounds and success

One axis-aligned, red **30 mm / 25 g block per episode**, placed on a blue destination region. New block positions are sampled on every reset; this is repeated block transfer, not simultaneous multi-object sorting or stacking.

| Position | X range | Y range |
|---|---|---|
| Block | 0.21–0.25 m | −0.075 to −0.035 m |
| Destination | 0.21–0.25 m | 0.045–0.085 m |

A successful episode must hold the block above 65 mm with contact on **both fingers for at least 0.5 seconds**, then leave it within 12 mm of the destination, at table height, released and stationary for at least 0.5 seconds after retreat. Each episode runs 12.5 simulated seconds. This excludes merely pushing the block into the target or dropping it nearby.

The fixed camera, colors, lighting, cube geometry, home pose, and bounded workspace are part of the training distribution. Retrain after changing them. No real-arm deployment or sim-to-real reliability is claimed; hardware requires calibration, perception, collision limits, and independent safety validation.

## Files

- `so101-sim/pick.py`: environment, teacher, training, learned policies, evaluation, mjviser integration.
- `so101-sim/pick.xml`: block, target, camera, and position sensors; includes the existing robot model.
- `so101-sim/checkpoints/*.npz`: trained parameters and training seed/demo count, loaded without pickle.

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
