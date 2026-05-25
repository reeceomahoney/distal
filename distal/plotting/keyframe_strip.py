"""Save a horizontal strip of evenly-spaced keyframes from a single episode of
a LeRobot dataset.

By default, samples N frames from episode 0 of `reece-omahoney/remove-pen-lid-2`
and arranges them left-to-right. If multiple camera keys are given, each camera
becomes its own row.
"""

from dataclasses import dataclass, field
from pathlib import Path

import draccus
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Config:
    dataset_repo_ids: list[str] = field(
        default_factory=lambda: [
            "reece-omahoney/remove-pen-lid-2",
            "reece-omahoney/remove-ethernet-2",
            "reece-omahoney/insert-ethernet-2",
        ]
    )
    episode: int = 31
    num_frames: int = 4
    skip_first: int = 10
    skip_last: int = 40
    camera_key: str = "observation.images.top"
    output_path: Path = Path("outputs/keyframe_strip.pdf")
    gap_px: int = 4
    row_gap_px: int = 32
    crop_v_px: int = 80
    gap_color: tuple[int, int, int, int] = (0, 0, 0, 0)
    titles: list[str] = field(
        default_factory=lambda: [
            "Remove pen lid",
            "Remove ethernet",
            "Insert ethernet",
        ]
    )
    title_font_size: int = 48
    title_padding_px: int = 16


def to_pil(image: torch.Tensor) -> Image.Image:
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr)


@draccus.wrap()
def main(cfg: Config) -> None:
    rows: list[list[Image.Image]] = []
    for repo_id in cfg.dataset_repo_ids:
        dataset = LeRobotDataset(repo_id=repo_id, vcodec="auto")
        episode_index = np.array(dataset.hf_dataset["episode_index"])
        frame_indices = np.where(episode_index == cfg.episode)[0]
        if frame_indices.size == 0:
            raise ValueError(f"episode {cfg.episode} not found in {repo_id}")

        start = min(cfg.skip_first, max(frame_indices.size - 1, 0))
        end = max(frame_indices.size - 1 - cfg.skip_last, start)
        n = min(cfg.num_frames, end - start + 1)
        picks = np.linspace(start, end, n).round().astype(int)
        sampled = frame_indices[picks]
        print(
            f"{repo_id} episode {cfg.episode}: {frame_indices.size} frames, "
            f"sampling {n} → {sampled.tolist()}"
        )
        row = [to_pil(dataset[int(i)][cfg.camera_key]) for i in sampled]
        if cfg.crop_v_px > 0:
            row = [
                im.crop((0, cfg.crop_v_px, im.width, im.height - cfg.crop_v_px))
                for im in row
            ]
        rows.append(row)

    n_cols = max(len(r) for r in rows)
    w, h = rows[0][0].size
    gap = cfg.gap_px
    row_gap = cfg.row_gap_px
    strip_w = n_cols * w + (n_cols - 1) * gap
    strip_h = len(rows) * h + (len(rows) - 1) * row_gap
    strip = Image.new("RGBA", (strip_w, strip_h), cfg.gap_color)
    for r, row in enumerate(rows):
        y = r * (h + row_gap)
        for c, im in enumerate(row):
            strip.paste(im, (c * (w + gap), y))

    titles = cfg.titles
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
            cfg.title_font_size,
        )
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(strip)
    pad = cfg.title_padding_px
    for r, title in enumerate(titles):
        y = r * (h + row_gap)
        bbox = draw.textbbox((0, 0), title, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([0, y, tw + 2 * pad, y + th + 2 * pad], fill=(0, 0, 0))
        draw.text(
            (pad - bbox[0], y + pad - bbox[1]),
            title,
            fill=(255, 255, 255),
            font=font,
        )

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.init()
    strip.save(cfg.output_path)
    print(f"saved {strip_w}x{strip_h} strip -> {cfg.output_path}")


if __name__ == "__main__":
    main()
