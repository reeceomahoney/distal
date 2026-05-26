from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    can_interface_left: str = "can_arm_left"
    can_interface_right: str = "can_arm_right"
    joint_names: list[str] = field(
        default_factory=lambda: [
            f"{side}_joint_{i + 1}" for side in ("left", "right") for i in range(6)
        ]
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "left_wrist": RealSenseCameraConfig(
                serial_number_or_name="335122272969",
                fps=30,
                width=640,
                height=480,
            ),
            "right_wrist": RealSenseCameraConfig(
                serial_number_or_name="123622270993",
                fps=30,
                width=640,
                height=480,
            ),
            "top": RealSenseCameraConfig(
                serial_number_or_name="323622271046",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
    teleop_mode: bool = True
    action_bias_path: str | None = None
    apply_bias_to_obs: bool = False
    # EMA smoothing on outgoing actions: smoothed = alpha * new + (1 - alpha) * prev.
    # None disables smoothing. Lower alpha = more smoothing. Only applied in policy
    # rollout (teleop_mode=False).
    action_ema_alpha: float | None = None
