"""Trained Cartesian servo with 20 Hz sensor or calibrated, current-RGB feedback.

The checkpoint is not compatible with pick.py. IK is used only by ``train``.
Camera control receives RGB, joint encoders and static calibration, never object
state. Source tracking ends when closing starts; destination tracking starts at
lift. Red pixels are never incorrectly projected onto the table during carry.
"""
import argparse
import itertools
import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from pick import CAMERAS, DURATIONS, HIGH, LOW, ROOT, Simulation, Teacher, create_run_dir, replay

VERSION = 1
HZ = 20
MAX_SECONDS = 35.
DEGREE = 6
DOMAIN_LOW = np.array([.20, -.115, .018])
DOMAIN_HIGH = np.array([.26, .095, .15])
STAGES = ("approach", "descend", "close", "lift", "transfer", "lower", "release", "retreat", "settle")
HEIGHTS = (.09, .019, .019, .12, .12, .025, .025, .14, .14)
GRIPS = (.7, .7, -.1, -.1, -.1, -.1, .7, .7, .7)


def powers(degree=DEGREE):
    return np.array([p for p in itertools.product(range(degree + 1), repeat=3) if sum(p) <= degree])


def features(x, exponents):
    normalized = (np.asarray(x) - (DOMAIN_LOW + DOMAIN_HIGH) / 2) / ((DOMAIN_HIGH - DOMAIN_LOW) / 2)
    return np.prod(normalized[..., None, :] ** exponents, axis=-1)


class Policy:
    """A fitted reachable Cartesian target -> actuator mapping, with no simulator."""
    def __init__(self, path, mode=None):
        try:
            with np.load(path, allow_pickle=False) as f:
                if int(f["version"]) != VERSION or str(f["contract"]) != "cartesian-servo-rgb-v1":
                    raise ValueError("incompatible contract")
                self.mode = str(f["mode"])
                self.weights = f["weights"]
                self.exponents = f["exponents"]
                self.home = f["home"]
                self.roll = float(f["roll"])
                if (not np.array_equal(f["domain_low"], DOMAIN_LOW)
                        or not np.array_equal(f["domain_high"], DOMAIN_HIGH)
                        or int(f["control_hz"]) != HZ
                        or self.roll != (-1.5 if self.mode == "wrist" else np.pi / 2)):
                    raise ValueError("incompatible workspace, orientation or control frequency")
                if mode is not None and mode != self.mode:
                    raise ValueError("checkpoint mode does not match --mode")
                if self.weights.shape != (len(powers()), 3) or not np.array_equal(self.exponents, powers()):
                    raise ValueError("incompatible fitted model")
        except (KeyError, ValueError, OSError) as exc:
            raise ValueError(f"Cannot load closed-loop checkpoint {path}: {exc}. Retrain with pick_closed_loop.py train.") from exc

    def command(self, xyz, grip):
        xyz = np.clip(xyz, DOMAIN_LOW, DOMAIN_HIGH)
        return np.r_[features(xyz, self.exponents) @ self.weights, np.pi / 2, self.roll, grip]


def training_solve(teacher, xyz, grip, roll):
    """Teacher-only refinement for the alternate physical wrist orientation."""
    q = teacher.solve(xyz, grip)
    if roll == np.pi / 2:
        return q
    teacher.data.qpos[:6] = q
    teacher.data.qpos[4] = roll
    for _ in range(180):
        mujoco.mj_forward(teacher.model, teacher.data)
        error = np.asarray(xyz) - teacher.data.site_xpos[teacher.site]
        if np.linalg.norm(error) < 1e-5:
            return teacher.data.qpos[:6].copy()
        mujoco.mj_jacSite(teacher.model, teacher.data, teacher.jac, None, teacher.site)
        jac = teacher.jac[:, :3]
        delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(3) * 1e-5, error)
        teacher.data.qpos[:3] = np.clip(teacher.data.qpos[:3] + np.clip(delta, -.15, .15),
                                      teacher.model.jnt_range[:3, 0], teacher.model.jnt_range[:3, 1])
    raise ValueError(f"Training target unreachable: {xyz}, roll={roll}")


def train(sim, args):
    teacher = Teacher(sim.model)
    rng = np.random.default_rng(args.seed)
    modes = (args.mode,) if args.mode else ("sensors", "camera", "wrist")
    args.weights.mkdir(parents=True, exist_ok=True)
    # Cover the whole servo volume, not only the nine nominal stage heights.
    axes = [np.linspace(a, b, 10) for a, b in zip(DOMAIN_LOW, DOMAIN_HIGH)]
    xyz = np.array(list(itertools.product(*axes)))
    xyz = np.vstack((xyz, rng.uniform(DOMAIN_LOW, DOMAIN_HIGH, size=(max(192, args.episodes), 3))))
    design = features(xyz, powers())
    fits = {}
    for mode in modes:
        roll = -1.5 if mode == "wrist" else np.pi / 2
        if roll not in fits:
            joints = np.array([training_solve(teacher, p, .7, roll)[:3] for p in xyz])
            weights = np.linalg.lstsq(design, joints, rcond=1e-10)[0]
            check_xyz = rng.uniform(DOMAIN_LOW, DOMAIN_HIGH, size=(192, 3))
            check_joints = np.array([training_solve(teacher, p, .7, roll)[:3] for p in check_xyz])
            rmse = float(np.sqrt(np.mean((features(check_xyz, powers()) @ weights - check_joints) ** 2)))
            if rmse > .003:
                raise RuntimeError(f"Fit inaccurate ({rmse} rad); no checkpoint written for {mode}")
            home = training_solve(teacher, [.22, 0, .15], .7, roll)
            fits[roll] = weights, home, rmse
        weights, home, rmse = fits[roll]
        path = args.weights / f"{mode}.npz"
        np.savez_compressed(path, version=VERSION, contract="cartesian-servo-rgb-v1", mode=mode,
                            weights=weights, exponents=powers(), home=home, roll=roll,
                            seed=args.seed, samples=len(xyz), control_hz=HZ,
                            domain_low=DOMAIN_LOW, domain_high=DOMAIN_HIGH)
        print(json.dumps({"checkpoint": str(path), "samples": len(xyz), "heldout_joint_rmse_rad": rmse}), flush=True)


class RGBObserver:
    """Ray/plane calibration from robot FK and RGB; no depth or object poses.

    Blue lies at z=.001. Red is a 30 mm cube standing on the table, so its
    silhouette is fitted to the projected known cube rather than projecting a
    moving camera's image centroid onto an incorrect table plane.
    """
    def __init__(self, model, mode):
        self.model = model
        self.mode = mode
        self.data = mujoco.MjData(model)
        self.camera = model.camera(CAMERAS[mode]).id
        self.site = model.site("gripperframe").id
        self.focal = 128 / np.tan(np.deg2rad(model.cam_fovy[self.camera]) / 2)
        self.corners = np.array(list(itertools.product((-.015, .015), repeat=3)))
        self.pad_corners = np.array([[x, y, 0.] for x, y in itertools.product((-.028, .028), repeat=2)])
        self.previous = {}

    def pose(self, joints):
        self.data.qpos[:6] = joints
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self.origin = self.data.cam_xpos[self.camera].copy()
        self.rotation = self.data.cam_xmat[self.camera].reshape(3, 3).copy()

    def plane(self, uv, z):
        ray = self.rotation @ np.array([(uv[0] - 127.5) / self.focal,
                                        -(uv[1] - 127.5) / self.focal, -1.])
        if abs(ray[2]) < 1e-6:
            return None
        distance = (z - self.origin[2]) / ray[2]
        if distance <= 0:
            return None
        return (self.origin + distance * ray)[:2]

    def polygon(self, xy, channel=0):
        points = (self.corners + np.r_[xy, .015] if channel == 0
                  else self.pad_corners + np.r_[xy, .001])
        local = (points - self.origin) @ self.rotation
        if np.any(local[:, 2] >= -.002):
            return None
        uv = np.column_stack((127.5 + self.focal * local[:, 0] / -local[:, 2],
                              127.5 - self.focal * local[:, 1] / -local[:, 2]))
        # Monotone-chain convex hull of the known projected box corners.
        points = sorted(set(map(tuple, uv)))
        def cross(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        lower, upper = [], []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return np.array(lower[:-1] + upper[:-1])

    def bbox(self, xy, channel=0):
        polygon = self.polygon(xy, channel)
        return None if polygon is None else np.r_[polygon.min(axis=0), polygon.max(axis=0)]

    def locate(self, rgb, channel):
        image = rgb.astype(np.float32)
        others = np.max(image[..., [c for c in range(3) if c != channel]], axis=-1)
        mask = (image[..., channel] > 65) & (image[..., channel] > 1.8 * others)
        y, x = np.nonzero(mask)
        if len(x) < (12 if channel == 2 else (60 if self.mode == "camera" else 20)):
            return None
        xy = self.plane([(x.min() + x.max()) / 2, (y.min() + y.max()) / 2],
                        .001 if channel == 2 else .02)
        if xy is None:
            return None
        # Fit the amodal silhouette. Robot-colored pixels (and carried red over
        # blue) are occluders, not evidence that the object shrank or moved.
        # Only actual colored object pixels and definite table pixels constrain
        # the fit; no simulator segmentation/depth/object state is consulted.
        background = ((image.max(axis=-1) - image.min(axis=-1) < 35)
                      & (image.max(axis=-1) > 45) & (image.max(axis=-1) < 130))
        x0, x1 = max(0, x.min() - 35), min(256, x.max() + 36)
        y0, y1 = max(0, y.min() - 35), min(256, y.max() + 36)
        rows, cols = np.mgrid[y0:y1, x0:x1]
        positive = mask[y0:y1, x0:x1]
        negative = background[y0:y1, x0:x1] & ~positive
        previous = self.previous.get(channel)
        def score(candidate):
            polygon = self.polygon(candidate, channel)
            if polygon is None:
                return np.inf
            inside = np.ones(rows.shape, dtype=bool)
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                inside &= ((b[0] - a[0]) * (rows - a[1])
                           - (b[1] - a[1]) * (cols - a[0])) >= 0
            regularizer = 0. if previous is None else .05 * np.sum(((candidate - previous) / .001) ** 2)
            return float(2 * np.count_nonzero(positive & ~inside) + np.count_nonzero(negative & inside) + regularizer)
        if previous is not None and score(previous) < score(xy):
            xy = previous.copy()
        best = score(xy)
        for step in (.008, .004, .002, .001, .0005, .00025):
            for _ in range(2):
                candidates = [xy + [dx * step, dy * step] for dx, dy in itertools.product((-1, 0, 1), repeat=2)]
                costs = [score(candidate) for candidate in candidates]
                index = int(np.argmin(costs))
                if costs[index] >= best:
                    break
                xy, best = candidates[index], costs[index]
        if best > max(20, len(x) * .5):
            return None
        self.previous[channel] = xy.copy()
        return xy

    def gripper_occludes(self, rgb, xy, channel=0):
        """Recognize expected final-contact occlusion, not arbitrary missing RGB."""
        if np.std(rgb) < 15:
            return False  # A covered lens is not an identified gripper occlusion.
        bounds = self.bbox(xy, channel)
        if bounds is None:
            return False
        x0, y0, x1, y1 = np.clip(bounds, 0, 255).astype(int)
        patch = rgb[y0:y1 + 1, x0:x1 + 1].astype(float)
        robot = ((patch[..., 0] > 80) & (patch[..., 1] > 50)
                 & (patch[..., 0] > patch[..., 2] * 1.3)) | (patch.max(axis=-1) < 45)
        if channel == 2:
            robot |= (patch[..., 0] > 65) & (patch[..., 0] > 1.8 * patch[..., 1:].max(axis=-1))
        return bool(robot.size and robot.mean() > .4)

    def observe(self, rgb, joints, need_source):
        self.pose(joints)
        source = self.locate(rgb, 0) if need_source else None
        destination = self.locate(rgb, 2)
        return source, destination


class Controller:
    """20 Hz observation/control updates; ``step`` advances one physics tick.

    Only update() reads policy inputs. Evaluation-only contact/success accounting
    is isolated in metrics(). A lost required view freezes stage time and motion;
    wrist recovery lifts to the safe viewing height before trying again.
    """
    def __init__(self, sim, policy):
        self.sim, self.policy = sim, policy
        self.observer = RGBObserver(sim.model, policy.mode) if policy.mode in CAMERAS else None
        self.interval = max(1, round(1 / (HZ * sim.model.opt.timestep)))
        self.phase_index = 0
        self.elapsed = 0.
        self.source = self.destination = None
        self.phase_start = policy.home.copy()
        self.command = policy.home.copy()
        self.latest_rgb = None
        self.observation_count = self.control_count = 0
        self.missing_seconds = 0.
        self.total_seconds = 0.
        self.recovering = False
        self.finished = False
        self.source_seen_at = -np.inf
        self.destination_seen_at = -np.inf

    @property
    def phase(self):
        return STAGES[self.phase_index]

    def update(self):
        sim = self.sim
        dt = self.interval * sim.model.opt.timestep
        need_source = self.phase_index < 2
        if self.policy.mode == "sensors":
            current = sim.observe("sensors")
            source = current[:2].copy() if need_source else None
            destination = current[2:].copy()
        else:
            self.latest_rgb = sim.observe(self.policy.mode)
            source, destination = self.observer.observe(self.latest_rgb, sim.data.qpos[:6].copy(), need_source)
        self.observation_count += 1
        for name, value in (("source", source), ("destination", destination)):
            if value is not None and np.all(np.isfinite(value)):
                setattr(self, name, value)
        if source is not None:
            self.source_seen_at = self.total_seconds
        if destination is not None:
            self.destination_seen_at = self.total_seconds
        if (source is None and self.observer is not None
                and (self.phase_index == 1 or (self.phase_index == 0 and self.elapsed >= .85))
                and self.source is not None and self.total_seconds - self.source_seen_at <= 2.2
                and self.observer.gripper_occludes(self.latest_rgb, self.source)
                and np.linalg.norm(self.observer.data.site_xpos[self.observer.site, :2] - self.source) < .025):
            # Only the final guarded approach may use bounded memory when the
            # current image and FK positively identify the gripper occluder.
            source = self.source
        if (destination is None and self.observer is not None and self.phase_index in (5, 6)
                and self.destination is not None and self.total_seconds - self.destination_seen_at <= 2.2
                and self.observer.gripper_occludes(self.latest_rgb, self.destination, 2)
                and np.linalg.norm(self.observer.data.site_xpos[self.observer.site, :2] - self.destination) < .025):
            destination = self.destination
        # Close is contact-occluded by design: hold the just-observed source.
        # Lift can proceed without blue; transfer/lower/release cannot.
        required = source if need_source else (destination if 4 <= self.phase_index <= 6 else self.source)
        if required is None:
            self.missing_seconds += dt
            if self.observer is not None and self.missing_seconds > .25:
                self.recovering = True
                # Pure joint-feedback FK supplies the current tool position.
                self.observer.pose(sim.data.qpos[:6])
                xyz = self.observer.data.site_xpos[self.observer.site].copy()
                xyz[2] = .15
                if need_source and self.source is not None:
                    xyz[:2] = self.source + [0., -.035]
                elif self.destination is not None:
                    xyz[:2] = self.destination
                recovery = self.policy.command(xyz, GRIPS[self.phase_index])
                if need_source and self.source is None:
                    recovery = self.policy.home.copy()
                    recovery[0] = .3  # Safe calibrated overview sweep, not a task guess.
                self.command += np.clip(recovery - self.command, -.06, .06)
                sim.data.ctrl[:] = self.command
                self.control_count += 1
            return
        self.missing_seconds = 0.
        if self.recovering:
            self.phase_start = self.command.copy()
            self.elapsed = 0.
            self.recovering = False
        xy = (self.source if self.phase_index < 4 else self.destination).copy()
        if self.phase_index == 0:
            # Finish alignment at safe height, then descend vertically. A
            # diagonal low approach would push the block with the fixed jaw.
            xy[1] -= .035 * np.clip((1.35 - self.elapsed) / .5, 0, 1)
        elif self.phase_index == 4:
            xy[1] -= .04  # Keep the pad visible beside the carried block.
        goal = self.policy.command([*xy, HEIGHTS[self.phase_index]], GRIPS[self.phase_index])
        u = np.clip(self.elapsed / DURATIONS[self.phase_index] * 1.3, 0, 1)
        u = u * u * (3 - 2 * u)
        self.command = self.phase_start + (goal - self.phase_start) * u
        sim.data.ctrl[:] = np.clip(self.command, sim.model.actuator_ctrlrange[:, 0], sim.model.actuator_ctrlrange[:, 1])
        self.control_count += 1
        self.elapsed += dt
        if self.elapsed >= DURATIONS[self.phase_index]:
            self.phase_start = self.command.copy()
            self.elapsed = 0.
            if self.phase_index == 8:
                self.finished = True
            else:
                self.phase_index += 1

    def metrics(self):
        sim = self.sim
        pos = sim.data.qpos[sim.block_adr:sim.block_adr + 3]
        sim.peak_height = max(sim.peak_height, float(pos[2]))
        contacts = sim.finger_contacts()
        sim.held_steps = sim.held_steps + 1 if pos[2] > .065 and len(contacts) == 2 else 0
        sim.max_held_steps = max(sim.max_held_steps, sim.held_steps)
        placed = (np.linalg.norm(pos[:2] - sim.model.body_pos[sim.target_id, :2]) < .012
                  and abs(pos[2] - .015) < .003 and not contacts
                  and np.linalg.norm(sim.data.qvel[6:9]) < .02)
        sim.settled_steps = sim.settled_steps + 1 if placed and self.phase_index == 8 else 0

    def step(self):
        if self.finished:
            return True
        if self.sim.step_count % self.interval == 0:
            self.update()
        mujoco.mj_step(self.sim.model, self.sim.data)
        self.sim.step_count += 1
        self.total_seconds += self.sim.model.opt.timestep
        self.metrics()
        # Missing views have a bounded recovery period; failure is reported honestly.
        self.finished = self.finished or self.total_seconds >= MAX_SECONDS
        return self.finished

    def result(self):
        return {**self.sim.result(), "phase": self.phase, "observations": self.observation_count,
                "control_updates": self.control_count, "duration_s": self.total_seconds}


def record_episode(sim, controller, path, metadata):
    """Save every integration state and matching RGB, including both endpoints."""
    capacity = int(np.ceil(MAX_SECONDS / sim.model.opt.timestep)) + 2
    width = mujoco.mj_stateSize(sim.model, mujoco.mjtState.mjSTATE_INTEGRATION)
    states = np.empty((capacity, width))
    # Recovery can extend camera episodes to 35s. Disk-backed RGB avoids holding
    # the maximum-size image stream in RAM, or copying it when stacking frames.
    with tempfile.TemporaryDirectory(prefix=".recording-", dir=path.parent) as scratch:
        rgb = (np.memmap(Path(scratch) / "rgb", mode="w+", dtype=np.uint8,
                         shape=(capacity, 256, 256, 3))
               if controller.policy.mode in CAMERAS else None)
        try:
            frames = 0
            while True:
                mujoco.mj_getState(sim.model, sim.data, states[frames],
                                   mujoco.mjtState.mjSTATE_INTEGRATION)
                if rgb is not None:
                    rgb[frames] = sim.observe(controller.policy.mode)
                frames += 1
                if controller.finished:
                    break
                controller.step()
            result = controller.result()
            np.savez_compressed(path, version=2, states=states[:frames],
                                timestep=sim.model.opt.timestep,
                                metadata=json.dumps({**metadata, **result}),
                                **({"rgb_frames": rgb[:frames]} if rgb is not None else {}))
        finally:
            del rgb
    return result


def evaluate(sim, policy, args):
    rng = np.random.default_rng(args.seed)
    results = []
    run_dir = create_run_dir(args.save_dir) if args.save_dir is not None else None
    if run_dir is not None:
        print(f"Run: {run_dir.name} ({run_dir})", flush=True)
    for episode in range(args.episodes):
        sim.reset(rng.uniform(LOW, HIGH), policy.home)
        controller = Controller(sim, policy)
        if run_dir is None:
            while not controller.step():
                pass
            result = controller.result()
        else:
            path = run_dir / f"{policy.mode}-{args.seed}-{episode:04d}.npz"
            metadata = {"run_name": run_dir.name, "controller": "closed_loop",
                        "mode": policy.mode, "seed": args.seed, "episode": episode}
            result = record_episode(sim, controller, path, metadata)
            print(f"Recording: {path}", flush=True)
        results.append(result)
        print(json.dumps({"episode": episode, **result}), flush=True)
    summary = {"mode": policy.mode, "episodes": len(results), "seed": args.seed,
               "successes": sum(r["success"] for r in results),
               "mean_placement_error_m": float(np.mean([r["placement_error_m"] for r in results]))}
    if run_dir is not None:
        summary["run_dir"] = str(run_dir)
    print(json.dumps(summary), flush=True)
    if args.report:
        args.report.write_text(json.dumps({"summary": summary, "episodes": results}, indent=2) + "\n")
    return summary["successes"] == len(results)


def view(sim, policy, args):
    from mjviser import Viewer
    import viser
    rng = np.random.default_rng(args.seed)
    server = viser.ViserServer(host=args.host, port=args.port)
    status = server.gui.add_markdown(f"### Closed-loop / {policy.mode}")
    display = None
    controller = None

    def reset(model, data):
        nonlocal controller
        sim.reset(rng.uniform(LOW, HIGH), policy.home)
        controller = Controller(sim, policy)

    def step(model, data):
        nonlocal display
        done = controller.step()
        if controller.latest_rgb is not None and sim.step_count % controller.interval == 1:
            if display is None:
                display = server.gui.add_image(controller.latest_rgb, label="Latest consumed RGB (20 Hz)")
            else:
                display.image = controller.latest_rgb
        status.content = f"### Closed-loop / {policy.mode}\nPhase: {controller.phase}; observations: {controller.observation_count}"
        if done:
            print(json.dumps(controller.result()), flush=True)
            reset(model, data)

    reset(sim.model, sim.data)
    viewer = Viewer(sim.model, sim.data, step_fn=step, reset_fn=reset, server=server)
    viewer.scene.camera_tracking_enabled = False
    viewer.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "eval", "view", "replay"))
    parser.add_argument("--mode", choices=("sensors", "camera", "wrist"))
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints" / "closed_loop")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--save-dir", type=Path, help="Save each eval episode as a replayable NPZ")
    parser.add_argument("--recording", type=Path, help="NPZ episode to replay in mjviser")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.episodes is None:
        args.episodes = 192 if args.command == "train" else 20
    if args.seed is None:
        args.seed = 0 if args.command == "train" else 10000
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.save_dir is not None and args.command != "eval":
        parser.error("--save-dir is only supported by eval")
    if (args.recording is not None) != (args.command == "replay"):
        parser.error("replay requires --recording; other commands do not accept it")
    if args.command in ("eval", "view") and args.mode is None:
        parser.error("eval/view requires --mode")
    sim = Simulation()
    try:
        if args.command == "train":
            train(sim, args)
        elif args.command == "replay":
            replay(sim, args)
        else:
            policy = Policy(args.weights / f"{args.mode}.npz", args.mode)
            if args.command == "eval":
                if not evaluate(sim, policy, args):
                    raise SystemExit(1)
            else:
                view(sim, policy, args)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
