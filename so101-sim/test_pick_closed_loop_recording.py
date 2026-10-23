import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

import pick_closed_loop as closed_loop
from pick import Replay


class RecordingTests(unittest.TestCase):
    def test_timeout_keeps_initial_and_final_states_and_replays_to_end(self):
        sim = closed_loop.Simulation()
        try:
            policy = closed_loop.Policy(closed_loop.ROOT / "checkpoints/closed_loop/sensors.npz", "sensors")
            sim.reset(np.array([.225, -.06, .225, .06]), policy.home)
            width = mujoco.mj_stateSize(sim.model, mujoco.mjtState.mjSTATE_INTEGRATION)
            initial = np.empty(width)
            final = np.empty(width)
            mujoco.mj_getState(sim.model, sim.data, initial, mujoco.mjtState.mjSTATE_INTEGRATION)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "timeout.npz"
                # An off-grid deadline exercises the final partial control period.
                with patch.object(closed_loop, "MAX_SECONDS", .101):
                    result = closed_loop.record_episode(
                        sim, closed_loop.Controller(sim, policy), path,
                        {"mode": "sensors", "controller": "closed_loop", "seed": 0, "episode": 0},
                    )
                mujoco.mj_getState(sim.model, sim.data, final, mujoco.mjtState.mjSTATE_INTEGRATION)
                final_qpos, final_qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
                final_ctrl = sim.data.ctrl.copy()
                self.assertFalse(result["success"])
                self.assertAlmostEqual(result["duration_s"], .105)
                with np.load(path, allow_pickle=False) as recording:
                    self.assertEqual(len(recording["states"]), 22)
                    np.testing.assert_array_equal(recording["states"][0], initial)
                    np.testing.assert_array_equal(recording["states"][-1], final)
                    self.assertFalse(json.loads(str(recording["metadata"]))["success"])
                    self.assertNotIn("rgb_frames", recording.files)
                playback = Replay(sim.model, path)
                playback.reset(sim.model, sim.data)
                self.assertEqual(sim.data.time, 0.)
                for index in range(1, len(playback.states)):
                    playback.step(sim.model, sim.data)
                    self.assertAlmostEqual(sim.data.time, index * sim.model.opt.timestep)
                np.testing.assert_array_equal(sim.data.qpos, final_qpos)
                np.testing.assert_array_equal(sim.data.qvel, final_qvel)
                np.testing.assert_array_equal(sim.data.ctrl, final_ctrl)
                playback.step(sim.model, sim.data)
                self.assertEqual(playback.index, len(playback.states) - 1)
                self.assertAlmostEqual(sim.data.time, .105)
                playback.reset(sim.model, sim.data)
                self.assertEqual(playback.index, 0)
                self.assertEqual(sim.data.time, 0.)
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
