from dataclasses import dataclass, field

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_teleop")
@dataclass
class PiperTeleoperatorConfig(TeleoperatorConfig):
    can_interface_left: str = "can_arm_left"
    can_interface_right: str = "can_arm_right"
    joint_names: list[str] = field(
        default_factory=lambda: [
            f"{side}_joint_{i + 1}" for side in ("left", "right") for i in range(6)
        ]
    )
