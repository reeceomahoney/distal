"""Replay a dataset episode while logging live state, then emit a per-joint bias.

Produces joint_bias.npz: today's mean (live - recorded) state per joint when
the recorded actions are re-sent. Consumed by replay.py and the Piper plugin
(action_bias_path) to compensate for calibration drift between record and now.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from lerobot.datasets import LeRobotDataset
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep

RAMP_SECONDS = 2.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo_id")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--root", default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--out", default="joint_bias.npz")
    args = p.parse_args()

    register_third_party_plugins()

    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])
    fps = args.fps or dataset.fps
    action_names = dataset.features[ACTION]["names"]
    state_names = dataset.features[OBS_STATE]["names"]
    assert state_names == action_names, (
        f"name mismatch:\n  act={action_names}\n  st={state_names}"
    )

    actions = np.stack(
        [np.asarray(dataset[i][ACTION]) for i in range(dataset.num_frames)]
    )
    recorded_state = np.stack(
        [np.asarray(dataset[i][OBS_STATE]) for i in range(dataset.num_frames)]
    )

    robot = make_robot_from_config(
        RobotConfig.get_choice_class("piper")(teleop_mode=False)
    )
    robot.connect()

    live_state = np.full_like(recorded_state, np.nan)
    n_done = 0
    try:
        ramp_to(
            robot,
            {n: float(actions[0, i]) for i, n in enumerate(action_names)},
            RAMP_SECONDS,
            fps,
        )

        for idx in range(dataset.num_frames):
            t0 = time.perf_counter()
            robot.send_action(
                {n: float(actions[idx, i]) for i, n in enumerate(action_names)}
            )
            obs = robot.get_observation()
            live_state[idx] = [float(obs[n]) for n in state_names]
            n_done = idx + 1
            precise_sleep(max(1 / fps - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()
        save_and_report(
            args.out, action_names, actions, recorded_state, live_state, n_done
        )


def save_and_report(out_path, names, actions, recorded_state, live_state, n_done):
    out = Path(out_path).resolve()
    np.savez(
        out,
        names=np.array(names),
        action=actions,
        recorded_state=recorded_state,
        live_state=live_state,
        n_done=n_done,
    )
    print(f"\nSaved {out}  ({n_done}/{len(actions)} frames replayed)")

    if n_done == 0:
        return

    a = actions[:n_done]
    r = recorded_state[:n_done]
    lv = live_state[:n_done]

    rec_err = np.abs(a - r).mean(axis=0)
    live_err = np.abs(a - lv).mean(axis=0)
    drift = np.abs(r - lv).mean(axis=0)
    bias = (lv - r).mean(axis=0)

    reported = [(i, n) for i, n in enumerate(names) if "gripper" not in n]
    w = max(len(n) for _, n in reported)
    print(
        f"\n{'joint':{w}}  {'|act-state| rec':>15}  "
        f"{'|act-state| live':>17}  {'|rec-live|':>11}  {'live-rec bias':>14}"
    )
    for i, n in reported:
        print(
            f"{n:{w}}  {rec_err[i]:15.3f}  {live_err[i]:17.3f}  "
            f"{drift[i]:11.3f}  {bias[i]:+14.3f}"
        )
    print("(grippers excluded from report)")


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
