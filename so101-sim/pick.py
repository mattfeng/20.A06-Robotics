"""CPU imitation learning for SO-ARM101. See ../README.md for scope and commands."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
DURATIONS = np.array([1.5, 1.5, 1., 1.5, 2., 1.5, 1., 1.5, 1.])
EDGES = np.cumsum(DURATIONS)
LOW = np.array([.21, -.075, .21, .045])
HIGH = np.array([.25, -.035, .25, .085])


def image_features(rgb):
    """RGB-only color moments; no depth, segmentation IDs, or camera calibration."""
    rgb = np.asarray(rgb, dtype=np.float32)
    features = []
    for channel in (0, 2):
        others = np.max(rgb[..., [i for i in range(3) if i != channel]], axis=-1)
        mask = (rgb[..., channel] > 60) & (rgb[..., channel] > others * 1.8)
        y, x = np.nonzero(mask)
        if len(x) < 12:
            raise ValueError("Camera must see both the red block and blue destination")
        features.extend((x.mean() / rgb.shape[1], y.mean() / rgb.shape[0]))
    return np.array(features)


def basis(x):
    """Separate cubic features for source and destination (19 coefficients)."""
    x = np.asarray(x)
    terms = [np.ones(x.shape[:-1])]
    for a, b in ((x[..., 0], x[..., 1]), (x[..., 2], x[..., 3])):
        terms.extend((a, b, a*a, a*b, b*b, a**3, a*a*b, a*b*b, b**3))
    return np.stack(terms, axis=-1)


class Policy:
    """Learned observation -> joint waypoints. Does not receive a simulator."""
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as weights:
            self.mode = str(weights["mode"])
            self.mean = weights["mean"]
            self.scale = weights["scale"]
            self.weights = weights["weights"]
            self.home = weights["home"]

    def plan(self, observation):
        x = image_features(observation) if self.mode == "camera" else observation
        return (basis((x - self.mean) / self.scale) @ self.weights).reshape(9, 6)


class Simulation:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "pick.xml"))
        self.data = mujoco.MjData(self.model)
        self.block_adr = self.model.jnt_qposadr[self.model.joint("block_joint").id]
        self.target_id = self.model.body("target").id
        self.block_geom = self.model.geom("block_geom").id
        self.finger_bodies = {self.model.body(n).id for n in ("gripper", "moving_jaw_so101_v1")}
        self.renderer = None

    def reset(self, xy, home):
        self.model.body_pos[self.target_id, :2] = xy[2:]
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.block_adr:self.block_adr+3] = [*xy[:2], .015]
        self.data.qpos[:6] = home
        self.data.ctrl[:] = home
        mujoco.mj_forward(self.model, self.data)
        self.held_steps = self.max_held_steps = self.settled_steps = 0
        self.peak_height = .015
        self.step_count = 0

    def observe(self, mode):
        if mode == "sensors":
            return np.r_[self.data.sensor("block_position").data[:2],
                         self.data.sensor("target_position").data[:2]]
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=256, width=256)
        self.renderer.update_scene(self.data, camera="overhead")
        return self.renderer.render()

    def finger_contacts(self):
        touching = set()
        for c in self.data.contact:
            if c.geom1 == self.block_geom:
                touching.add(self.model.geom_bodyid[c.geom2])
            elif c.geom2 == self.block_geom:
                touching.add(self.model.geom_bodyid[c.geom1])
        return touching & self.finger_bodies

    def step(self, plan, home):
        # The learned controller sees only its initial observation and a clock.
        t = self.step_count * self.model.opt.timestep
        edges = EDGES
        phase = min(int(np.searchsorted(edges, t, side="right")), len(plan)-1)
        start = home if phase == 0 else plan[phase-1]
        elapsed = t - (0 if phase == 0 else edges[phase-1])
        u = np.clip(elapsed / DURATIONS[phase] * 1.3, 0, 1)
        u = u*u*(3-2*u)
        self.data.ctrl[:] = np.clip(start + (plan[phase]-start)*u,
                                   self.model.actuator_ctrlrange[:, 0],
                                   self.model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        # Privileged state below is evaluation-only, never policy input.
        pos = self.data.qpos[self.block_adr:self.block_adr+3]
        self.peak_height = max(self.peak_height, float(pos[2]))
        contacts = self.finger_contacts()
        self.held_steps = self.held_steps + 1 if pos[2] > .065 and len(contacts) == 2 else 0
        self.max_held_steps = max(self.max_held_steps, self.held_steps)
        placed = (np.linalg.norm(pos[:2] - self.model.body_pos[self.target_id, :2]) < .012
                  and abs(pos[2] - .015) < .003 and not contacts
                  and np.linalg.norm(self.data.qvel[6:9]) < .02)
        self.settled_steps = self.settled_steps + 1 if placed and phase == 8 else 0
        return self.step_count >= round(DURATIONS.sum() / self.model.opt.timestep)

    def result(self):
        pos = self.data.qpos[self.block_adr:self.block_adr+3]
        return {"success": bool(self.max_held_steps >= 100 and self.settled_steps >= 100),
                "peak_height_m": self.peak_height,
                "placement_error_m": float(np.linalg.norm(pos[:2] - self.model.body_pos[self.target_id, :2])),
                "held_seconds": self.max_held_steps * self.model.opt.timestep,
                "settled_seconds": self.settled_steps * self.model.opt.timestep}

    def rollout(self, plan, home, states=None):
        while True:
            if states is not None:
                mujoco.mj_getState(self.model, self.data, states[self.step_count],
                                   mujoco.mjtState.mjSTATE_INTEGRATION)
            if self.step(plan, home):
                break
        if states is not None:
            mujoco.mj_getState(self.model, self.data, states[self.step_count],
                               mujoco.mjtState.mjSTATE_INTEGRATION)
        return self.result()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


class Teacher:
    """Damped IK used only to collect demonstrations, never during learned rollouts."""
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site("gripperframe").id
        self.jac = np.zeros((3, model.nv))

    def solve(self, xyz, grip):
        self.data.qpos[:6] = [0, 0, 0, np.pi/2, np.pi/2, grip]
        for _ in range(150):
            mujoco.mj_forward(self.model, self.data)
            error = np.asarray(xyz) - self.data.site_xpos[self.site]
            if np.linalg.norm(error) < 1e-5:
                return self.data.qpos[:6].copy()
            mujoco.mj_jacSite(self.model, self.data, self.jac, None, self.site)
            j = self.jac[:, :3]
            delta = j.T @ np.linalg.solve(j @ j.T + np.eye(3)*1e-5, error)
            self.data.qpos[:3] = np.clip(self.data.qpos[:3] + np.clip(delta, -.15, .15),
                                       self.model.jnt_range[:3, 0], self.model.jnt_range[:3, 1])
        raise ValueError(f"Teacher cannot reach {xyz}")

    def plan(self, xy):
        source, goal = xy[:2], xy[2:]
        stages = [(source, .09, .7), (source, .019, .7), (source, .019, -.1),
                  (source, .12, -.1), (goal, .12, -.1), (goal, .025, -.1),
                  (goal, .025, .7), (goal, .14, .7), (goal, .14, .7)]
        return np.array([self.solve([*p, z], g) for p, z, g in stages])


def train(sim, args):
    rng = np.random.default_rng(args.seed)
    teacher = Teacher(sim.model)
    home = teacher.solve([.20, 0, .15], .7)
    observations = {"sensors": [], "camera": []}
    actions = []
    for i in range(args.episodes):
        xy = rng.uniform(LOW, HIGH)
        sim.reset(xy, home)
        sensors = sim.observe("sensors")
        camera = image_features(sim.observe("camera"))
        plan = teacher.plan(xy)
        result = sim.rollout(plan, home)
        if result["success"]:
            observations["sensors"].append(sensors)
            observations["camera"].append(camera)
            actions.append(plan.ravel())
        if (i+1) % 16 == 0:
            print(f"Demonstrations: {i+1}/{args.episodes}, successful: {len(actions)}", flush=True)
    if len(actions) < 40:
        raise RuntimeError("Need at least 40 successful demonstrations; increase --episodes")
    args.weights.mkdir(parents=True, exist_ok=True)
    for mode, values in observations.items():
        x = np.array(values)
        mean, scale = x.mean(axis=0), x.std(axis=0)
        design = basis((x - mean) / scale)
        weights = np.linalg.lstsq(design, np.array(actions), rcond=1e-8)[0]
        path = args.weights / f"{mode}.npz"
        np.savez_compressed(path, mode=mode, weights=weights, mean=mean, scale=scale,
                            home=home, seed=args.seed, demonstrations=len(actions))
        print(json.dumps({"checkpoint": str(path), "demonstrations": len(actions),
                          "joint_rmse_rad": float(np.sqrt(np.mean((design @ weights - actions)**2)))}))


def evaluate(sim, policy, args):
    rng = np.random.default_rng(args.seed)
    results = []
    states = None
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        frames = round(DURATIONS.sum() / sim.model.opt.timestep) + 1
        width = mujoco.mj_stateSize(sim.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        states = np.empty((frames, width))
    for i in range(args.episodes):
        sim.reset(rng.uniform(LOW, HIGH), policy.home)
        plan = policy.plan(sim.observe(policy.mode))
        result = sim.rollout(plan, policy.home, states)
        if states is not None:
            path = args.save_dir / f"{policy.mode}-{args.seed}-{i:04d}.npz"
            metadata = {"mode": policy.mode, "seed": args.seed, "episode": i, **result}
            np.savez_compressed(path, version=1, states=states,
                                timestep=sim.model.opt.timestep,
                                metadata=json.dumps(metadata))
            print(f"Recording: {path}", flush=True)
        results.append(result)
        print(json.dumps({"episode": i, **result}), flush=True)
    summary = {"mode": policy.mode, "seed": args.seed, "episodes": len(results),
               "successes": sum(r["success"] for r in results),
               "mean_placement_error_m": float(np.mean([r["placement_error_m"] for r in results]))}
    print(json.dumps(summary), flush=True)
    if args.report:
        args.report.write_text(json.dumps({"summary": summary, "episodes": results}, indent=2) + "\n")
    return summary["successes"] == len(results)


def view(sim, policy, args):
    from mjviser import Viewer
    import viser

    rng = np.random.default_rng(args.seed)
    plan = None
    server = viser.ViserServer(host=args.host, port=args.port)
    status = server.gui.add_markdown(f"### SO-ARM101 / {policy.mode}")
    camera_display = None

    def reset(model, data):
        nonlocal plan, camera_display
        sim.reset(rng.uniform(LOW, HIGH), policy.home)
        obs = sim.observe(policy.mode)
        plan = policy.plan(obs)
        if policy.mode == "camera":
            if camera_display is None:
                camera_display = server.gui.add_image(obs, label="Policy input (initial RGB)")
            else:
                camera_display.image = obs

    def step(model, data):
        if sim.step(plan, policy.home):
            result = sim.result()
            print(json.dumps(result), flush=True)
            status.content = f"### SO-ARM101 / {policy.mode}\nLast episode: {'success' if result['success'] else 'failed'}; placement error {result['placement_error_m']*1000:.1f} mm"
            reset(model, data)

    reset(sim.model, sim.data)
    viewer = Viewer(sim.model, sim.data, step_fn=step, reset_fn=reset, server=server)
    viewer.scene.camera_tracking_enabled = False
    viewer.run()


class Replay:
    """Restore recorded integration states; never advance physics or a policy."""
    def __init__(self, model, path):
        with np.load(path, allow_pickle=False) as recording:
            if int(recording["version"]) != 1:
                raise ValueError("Unsupported recording version")
            self.states = recording["states"]
            timestep = float(recording["timestep"])
            self.metadata = json.loads(str(recording["metadata"]))
        width = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
        if (self.states.ndim != 2 or self.states.shape[0] < 2
                or self.states.shape[1] != width or not np.isfinite(self.states).all()
                or not np.isfinite(timestep) or timestep <= 0):
            raise ValueError("Invalid recording or incompatible MuJoCo model")
        model.opt.timestep = timestep
        self.index = 0

    def apply(self, model, data):
        mujoco.mj_setState(model, data, self.states[self.index],
                           mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)

    def reset(self, model, data):
        self.index = 0
        self.apply(model, data)

    def step(self, model, data):
        self.index = min(self.index + 1, len(self.states) - 1)
        self.apply(model, data)


def replay(sim, args):
    from mjviser import Viewer
    import viser

    recording = Replay(sim.model, args.recording)
    recording.reset(sim.model, sim.data)
    server = viser.ViserServer(host=args.host, port=args.port)
    meta = recording.metadata
    server.gui.add_markdown(
        f"### Recorded SO-ARM101 / {meta['mode']}\n"
        f"Seed {meta['seed']}, episode {meta['episode']}: "
        f"{'success' if meta['success'] else 'failed'}.\n\n"
        "Recorded states, not re-simulated. Holds the final frame; Reset to replay."
    )
    viewer = Viewer(sim.model, sim.data, step_fn=recording.step,
                    reset_fn=recording.reset, server=server)
    viewer.scene.camera_tracking_enabled = False
    viewer.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "eval", "view", "replay"))
    parser.add_argument("--mode", choices=("sensors", "camera"), default="sensors")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--save-dir", type=Path, help="Save each eval episode as a replayable NPZ")
    parser.add_argument("--recording", type=Path, help="NPZ episode to replay in mjviser")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    args.episodes = args.episodes if args.episodes is not None else (192 if args.command == "train" else 20)
    args.seed = args.seed if args.seed is not None else (0 if args.command == "train" else 10000)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.save_dir is not None and args.command != "eval":
        parser.error("--save-dir is only supported by eval")
    if (args.recording is not None) != (args.command == "replay"):
        parser.error("replay requires --recording; other commands do not accept it")
    sim = Simulation()
    try:
        if args.command == "train":
            train(sim, args)
        elif args.command == "replay":
            replay(sim, args)
        else:
            policy = Policy(args.weights / f"{args.mode}.npz")
            if args.command == "view":
                view(sim, policy, args)
            elif not evaluate(sim, policy, args):
                raise SystemExit(1)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
