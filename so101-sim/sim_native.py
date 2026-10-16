from pathlib import Path

import mujoco

scene = (
    Path(__file__).resolve().parent
    / "mujoco_menagerie/robotstudio_so101/scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)

print("Joint positions:", model.nq)
print("Actuators:", model.nu)
print("Timestep:", model.opt.timestep)

# Command a small shoulder-pan movement.
data.ctrl[:] = 0.0
data.ctrl[0] = 0.3

# Simulate one second.
for _ in range(round(1.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)

print("Final joint positions:", data.qpos)
