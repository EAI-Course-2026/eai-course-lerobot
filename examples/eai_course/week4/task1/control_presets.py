"""Switch arm poses with keys and execute predefined action sequences."""

import argparse
import json
import time
from pathlib import Path

import msvcrt
from scservo_sdk import (
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
    SCS_HIBYTE,
    SCS_LOBYTE,
)

from arm_config import load_calibrated_ranges

BAUDRATE = 1_000_000
PROTOCOL = 1

TORQUE_ENABLE_ADDRESS = 40
GOAL_POSITION_ADDRESS = 42
PRESENT_POSITION_ADDRESS = 56

# Starting at address 42, each command contains three consecutive words:
# goal position (42), running time (44), and running speed (46).
SYNC_DATA_LENGTH = 6

POSES_PATH = Path(__file__).with_name("poses.json")

# Number keys switch directly to one saved pose.
POSE_KEYS = {
    "1": "stand",
    "2": "raise",
    "3": "left",
    "4": "right",
}

# Each action step contains:
#   pose: target name in poses.json
#   time_ms: requested transition time; use 0 when controlling by speed
#   speed: requested servo speed; use 0 when controlling by time
#   wait_s: delay before sending the next step
ACTION_KEYS = {
    "w": (
        {"pose": "raise", "time_ms": 1500, "speed": 0, "wait_s": 1.7},
        {"pose": "left", "time_ms": 1000, "speed": 0, "wait_s": 1.0},
        {"pose": "right", "time_ms": 1000, "speed": 0, "wait_s": 1.0},
        {"pose": "left", "time_ms": 1000, "speed": 0, "wait_s": 1.0},
        {"pose": "stand", "time_ms": 1500, "speed": 0, "wait_s": 1.5},
    ),
}


def load_poses():
    """Load all named normalized poses from poses.json."""
    return json.loads(POSES_PATH.read_text(encoding="utf-8"))


def normalized_to_raw(values, ranges):
    """Convert six 0..1 values into six calibrated raw positions."""
    positions = {}
    for servo_id, value in enumerate(values, start=1):
        minimum, maximum = ranges[servo_id]
        positions[servo_id] = round(minimum + value * (maximum - minimum))
    return positions


def word_bytes(value):
    """Encode one 16-bit number using the byte order selected by protocol 1."""
    return [SCS_LOBYTE(value), SCS_HIBYTE(value)]


def set_torque(packet, port, ranges, enabled):
    """Enable or disable all six servos."""
    value = 1 if enabled else 0
    for servo_id in ranges:
        packet.write1ByteTxRx(port, servo_id, TORQUE_ENABLE_ADDRESS, value)


def read_raw_positions(packet, port, ranges):
    """Read the actual position of all six servos."""
    positions = {}
    for servo_id in ranges:
        position, _, _ = packet.read2ByteTxRx(port, servo_id, PRESENT_POSITION_ADDRESS)
        positions[servo_id] = position
    return positions


def sync_write_targets(sync_writer, positions, time_ms=0, speed=0):
    """Send position, running time, and speed to all six servos at once."""
    sync_writer.clearParam()

    for servo_id, position in positions.items():
        data = word_bytes(position) + word_bytes(time_ms) + word_bytes(speed)
        sync_writer.addParam(servo_id, data)

    sync_writer.txPacket()


def send_pose(sync_writer, poses, pose_name, ranges, time_ms=1000, speed=0):
    """Convert one normalized pose and send it to the arm."""
    if pose_name not in poses:
        print(f"Pose '{pose_name}' has not been recorded.")
        return

    positions = normalized_to_raw(poses[pose_name], ranges)
    sync_write_targets(sync_writer, positions, time_ms, speed)
    print(
        f"Pose: {pose_name}, time={time_ms} ms, speed={speed}, "
        f"targets={list(positions.values())}"
    )


def run_action(sync_writer, poses, ranges, steps):
    """Execute several pose transitions in their configured order."""
    for step in steps:
        send_pose(
            sync_writer,
            poses,
            step["pose"],
            ranges,
            step["time_ms"],
            step["speed"],
        )
        time.sleep(step["wait_s"])


def print_controls():
    print("\nKeyboard controls")
    for key, pose_name in POSE_KEYS.items():
        print(f"  {key}: {pose_name}")
    print("  w: wave action")
    print("  q: quit")


def main():
    parser = argparse.ArgumentParser(description="Control saved SCS215 arm poses.")
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--robot-id", default="scs215_com5")
    args = parser.parse_args()

    poses = load_poses()
    ranges = load_calibrated_ranges(args.port, args.robot_id)

    port = PortHandler(args.port)
    packet = PacketHandler(PROTOCOL)
    port.openPort()
    port.setBaudRate(BAUDRATE)

    sync_writer = GroupSyncWrite(
        port,
        packet,
        GOAL_POSITION_ADDRESS,
        SYNC_DATA_LENGTH,
    )

    # Set the current pose as the goal before torque is enabled. This prevents
    # the arm from chasing a target left behind by an earlier program.
    current_positions = read_raw_positions(packet, port, ranges)
    sync_write_targets(sync_writer, current_positions, time_ms=0, speed=0)
    set_torque(packet, port, ranges, True)

    print_controls()

    try:
        while True:
            key = msvcrt.getwch().lower()

            if key == "q":
                break
            if key in POSE_KEYS:
                send_pose(sync_writer, poses, POSE_KEYS[key], ranges, time_ms=1000, speed=0)
            elif key in ACTION_KEYS:
                run_action(sync_writer, poses, ranges, ACTION_KEYS[key])
    finally:
        set_torque(packet, port, ranges, False)
        port.closePort()
        print("Torque disabled. Port closed.")


if __name__ == "__main__":
    main()
