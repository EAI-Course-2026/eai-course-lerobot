"""Load per-robot joint ranges from LeRobot's local calibration file."""

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def load_calibrated_ranges(port: str, robot_id: str) -> dict[int, tuple[int, int]]:
    """Return servo ID -> raw minimum/maximum without opening the serial port."""
    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))
    if not robot.calibration:
        raise FileNotFoundError(f"No calibration found at {robot.calibration_fpath}")

    return {
        servo_id: (
            robot.calibration[joint].range_min,
            robot.calibration[joint].range_max,
        )
        for servo_id, joint in enumerate(JOINTS, start=1)
    }
