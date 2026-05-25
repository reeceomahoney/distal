"""Replay a dataset's action trajectory on the Piper arms."""

import argparse
import time

import numpy as np
from lerobot.datasets import LeRobotDataset
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep

RAMP_SECONDS = 2.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="LeRobot dataset repo id")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--root", default=None)
    parser.add_argument("--fps", type=int, default=None, help="Override dataset fps")
    parser.add_argument(
        "--bias-from",
        default=None,
        help="Path to joint_bias.npz; subtract mean(live-rec) per joint from each action. "
        "Grippers are excluded.",
    )
    args = parser.parse_args()

    register_third_party_plugins()

    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])
    fps = args.fps or dataset.fps
    action_names = dataset.features[ACTION]["names"]
    actions = dataset.select_columns(ACTION)

    bias = load_bias(args.bias_from, action_names) if args.bias_from else None

    robot = make_robot_from_config(
        RobotConfig.get_choice_class("piper")(teleop_mode=False)
    )
    robot.connect()
    try:
        first = build_action(actions[0][ACTION], action_names, bias)
        ramp_to(robot, first, RAMP_SECONDS, fps)

        for idx in range(dataset.num_frames):
            t0 = time.perf_counter()
            robot.send_action(build_action(actions[idx][ACTION], action_names, bias))
            precise_sleep(max(1 / fps - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()


def build_action(row, names, bias):
    if bias is None:
        return {n: float(row[i]) for i, n in enumerate(names)}
    return {n: float(row[i]) - bias[i] for i, n in enumerate(names)}


def load_bias(path: str, action_names: list[str]) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    diag_names = [str(n) for n in data["names"]]
    n_done = int(data["n_done"])
    rec = data["recorded_state"][:n_done]
    live = data["live_state"][:n_done]
    diag_bias = (live - rec).mean(axis=0)

    idx = {n: i for i, n in enumerate(diag_names)}
    bias = np.zeros(len(action_names), dtype=np.float64)
    for i, n in enumerate(action_names):
        if "gripper" in n:
            continue
        if n not in idx:
            raise KeyError(f"diag file missing joint {n}")
        bias[i] = diag_bias[idx[n]]

    print(f"Loaded bias from {path}:")
    for i, n in enumerate(action_names):
        print(f"  {n:24s} bias={bias[i]:+.3f}{'  (skipped)' if 'gripper' in n else ''}")
    return bias


def ramp_to(robot, target: dict, duration_s: float, fps: int) -> None:
    obs = robot.get_observation()
    start = {k: float(obs[k]) for k in target}
    steps = max(int(duration_s * fps), 1)
    for step in range(1, steps + 1):
        t = step / steps
        robot.send_action({k: start[k] * (1 - t) + target[k] * t for k in target})
        precise_sleep(1 / fps)


if __name__ == "__main__":
    main()
