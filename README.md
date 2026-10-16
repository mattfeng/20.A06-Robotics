# robotics


## so101-sim

```bash
uv sync --locked
```


```bash
# run the simulation
uv run python -m mujoco.viewer \
  --mjcf mujoco_menagerie/robotstudio_so101/scene.xml
```


```bash

# linux with CUDA GPU
uv run --extra cuda python sim_mjx.py
```
