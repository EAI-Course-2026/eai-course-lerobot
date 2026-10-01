"""Record the current six-joint arm pose into poses.json."""

import argparse
import json
from pathlib import Path

from scservo_sdk import PacketHandler, PortHandler

from arm_config import JOINTS, load_calibrated_ranges

BAUDRATE = 1_000_000
PROTOCOL = 1

TORQUE_ENABLE_ADDRESS = 40
PRESENT_POSITION_ADDRESS = 56

POSES_PATH = Path(__file__).with_name("poses.json")


def set_torque(packet, port, ranges, enabled):
    """Enable or disable torque on all six servos."""
    value = 1 if enabled else 0
    for servo_id in ranges:
        packet.write1ByteTxRx(port, servo_id, TORQUE_ENABLE_ADDRESS, value)


def read_raw_positions(packet, port, ranges):
    """Read address 56 from IDs 1 to 6."""
    positions = []
    for servo_id in ranges:
        position, _, _ = packet.read2ByteTxRx(port, servo_id, PRESENT_POSITION_ADDRESS)
        positions.append(position)
    return positions


def raw_to_normalized(raw_positions, ranges):
    """Map the six raw positions into calibrated values between 0 and 1."""
    values = []
    for servo_id, position in enumerate(raw_positions, start=1):
        minimum, maximum = ranges[servo_id]
        value = (position - minimum) / (maximum - minimum)
        values.append(round(value, 4))
    return values


def save_pose(name, values):
    """Add or replace one named pose in poses.json."""
    if POSES_PATH.exists():
        poses = json.loads(POSES_PATH.read_text(encoding="utf-8"))
    else:
        poses = {}

    poses[name] = values
    POSES_PATH.write_text(
        json.dumps(poses, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Record one arm pose.")
    parser.add_argument("name", help="Pose name, for example stand, raise, left or right")
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--robot-id", default="scs215_com5")
    args = parser.parse_args()

    ranges = load_calibrated_ranges(args.port, args.robot_id)
    port = PortHandler(args.port)
    packet = PacketHandler(PROTOCOL)
    port.openPort()
    port.setBaudRate(BAUDRATE)

    # Torque must be off so the arm can be positioned by hand.
    set_torque(packet, port, ranges, False)
    input(f"Move the arm to pose '{args.name}', then press Enter to record it...")

    raw_positions = read_raw_positions(packet, port, ranges)
    normalized_values = raw_to_normalized(raw_positions, ranges)
    save_pose(args.name, normalized_values)

    print("Joint order:", " ".join(JOINTS))
    print("Raw positions:", raw_positions)
    print("Normalized pose:", normalized_values)
    print("Saved to:", POSES_PATH)

    port.closePort()


if __name__ == "__main__":
    main()
