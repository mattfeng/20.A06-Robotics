from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

scene = (
    Path(__file__).resolve().parent
    / "mujoco_menagerie/robotstudio_so101/scene.xml"
)

# Load and compile the native MuJoCo model.
mj_model = mujoco.MjModel.from_xml_path(str(scene))

# Transfer it to the MJX-JAX backend.
mx = mjx.put_model(mj_model, impl="jax")


def rollout(target):
    data = mjx.make_data(mx)

    def step(data, _):
        data = data.replace(ctrl=target)
        data = mjx.step(mx, data)
        return data, None

    data, _ = jax.lax.scan(
        step,
        data,
        xs=None,
        length=20,
    )

    return data.qpos


# Four independent SO-101 simulations.
targets = jnp.zeros((4, mj_model.nu))

# Different shoulder-pan targets in each simulation.
targets = targets.at[:, 0].set(
    jnp.linspace(-0.3, 0.3, 4)
)

simulate = jax.jit(jax.vmap(rollout))
final_positions = simulate(targets)

print("Devices:", jax.devices())
print("Final positions shape:", final_positions.shape)
print(final_positions)
