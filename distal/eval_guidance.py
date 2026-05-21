"""Evaluate an advantage-conditioned policy across guidance scales.

Runs lerobot-eval (or distal.eval_libero_plus when ``libero_plus=True``) for
each guidance scale and prints a summary table.
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import draccus


@dataclass
class EvalGuidanceConfig:
    policy_path: str = "reece-omahoney/pistar-knn-libero"
    guidance_scales: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5])
    libero_plus: bool = False


@draccus.wrap()
def main(cfg: EvalGuidanceConfig):
    results = {}

    for beta in cfg.guidance_scales:
        print(f"\n{'=' * 60}")
        print(f"  cfg_beta = {beta}")
        print(f"{'=' * 60}\n", flush=True)

        if cfg.libero_plus:
            cmd = [
                sys.executable,
                "-m",
                "distal.eval_libero_plus",
                f"--policy_path={cfg.policy_path}",
                f"--cfg_beta={beta}",
            ]
            eval_root = Path("outputs/eval_libero_plus")
            summary_glob = "summary.json"
        else:
            cmd = [
                sys.executable,
                "-m",
                "lerobot.scripts.lerobot_eval",
                "--config_path=configs/eval.yaml",
                f"--policy.path={cfg.policy_path}",
                f"--policy.cfg_beta={beta}",
                "--env.task_ids=[8]",
                "--eval.n_episodes=25",
            ]
            eval_root = Path("outputs/eval")
            summary_glob = "eval_info.json"

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"WARNING: eval failed for beta={beta}")
            continue

        if not eval_root.exists():
            continue

        latest = max(eval_root.rglob(summary_glob), key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            info = json.load(f)

        if cfg.libero_plus:
            pc_success = info.get("pc_success", 0.0)
        else:
            pc_success = info.get("overall", {}).get("pc_success", 0.0)
        results[beta] = pc_success

        print(f"\nguidance_scale={beta}: success={pc_success:.1f}%")

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Beta':>8} | {'Success Rate':>14}")
    print(f"{'-' * 8}-+-{'-' * 14}")
    for beta, pc in results.items():
        print(f"{beta:>8.1f} | {pc:>13.1f}%")


if __name__ == "__main__":
    main()
