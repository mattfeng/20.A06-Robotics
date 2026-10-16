from pathlib import Path

import jax
import mujoco
from mujoco import mjx

scene = (
    Path(__file__).resolve().parent
    / "mujoco_menagerie/robotstudio_so101/scene.xml"
)

mj_model = mujoco.MjModel.from_xml_path(str(scene))

mx = mjx.put_model(mj_model, impl="warp")
data = mjx.make_data(
    mj_model,
    impl="warp",
    naconmax=128,
    njmax=64,
)

step = jax.jit(mjx.step)
data = step(mx, data)

print("Time:", data.time)
print("Joint positions:", data.qpos)
