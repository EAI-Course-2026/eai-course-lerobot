#!/usr/bin/env python
"""Record safe arm poses, collect synchronized samples, and solve hand-eye calibration."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageTk
from pynput import keyboard

from lerobot.cameras import ColorMode
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from calibrate_camera import open_camera
from camera_calibration import CameraCalibration, find_chessboard_corners
from hand_eye_calibration import (
    HAND_EYE_METHODS,
    HandEyeSample,
    canonicalize_target_frame,
    estimate_target_pose,
    load_dataset,
    prepare_hand_eye_samples,
    save_dataset,
    solve_all_methods,
    solve_hand_eye,
)


VISION_DIR = Path(__file__).resolve().parent
TASK2_DIR = VISION_DIR.parent / "week4" / "task2"
if str(TASK2_DIR) not in sys.path:
    sys.path.insert(0, str(TASK2_DIR))

from joint_mapping import ALL_MOTOR_NAMES, ARM_JOINT_NAMES, SO101JointMapper, URDF_LIMITS  # noqa: E402
from kinematics_backend import create_fk_solver  # noqa: E402
from run_steps3_to5 import arm_degrees_from_raw, raw_outside_calibration, read_raw_motors  # noqa: E402


DEFAULT_ROOT = VISION_DIR / "outputs" / "hand_eye"
DEFAULT_URDF = TASK2_DIR / "so101_new_calib.urdf"
DEFAULT_INTRINSICS = VISION_DIR / "outputs" / "camera_calibration" / "camera_intrinsics.json"


def make_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def backup_existing_file(path: Path, session_id: str) -> Path | None:
    """Copy an existing current result into the timestamped archive."""
    if not path.is_file():
        return None
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    candidate = archive_dir / f"{path.stem}_{session_id}{path.suffix}"
    suffix = 2
    while candidate.exists():
        candidate = archive_dir / f"{path.stem}_{session_id}_{suffix}{path.suffix}"
        suffix += 1
    shutil.copy2(path, candidate)
    return candidate


class EmergencyStop:
    def __init__(self) -> None:
        self.requested = threading.Event()

    def on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key == keyboard.Key.space:
            self.requested.set()


class SampleRejectedError(RuntimeError):
    """A pose-specific failure that may be skipped without ending the session."""


def load_mapper(robot: SO101Follower) -> SO101JointMapper:
    if not robot.calibration:
        raise FileNotFoundError(f"No robot calibration found at {robot.calibration_fpath}")
    mapper = SO101JointMapper(robot.calibration)
    errors = [issue for issue in mapper.validate() if issue.level == "ERROR"]
    if errors:
        details = "; ".join(f"{issue.joint}: {issue.message}" for issue in errors)
        raise RuntimeError(f"Robot calibration is invalid: {details}")
    return mapper


def load_recorded_poses(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported pose file: {path}")
    if tuple(payload.get("joint_order", [])) != ARM_JOINT_NAMES:
        raise ValueError("Pose joint order does not match the five-joint FK chain")
    poses = list(payload.get("poses", []))
    if not poses:
        raise ValueError(f"No poses in {path}")
    return poses


def save_recorded_poses(path: Path, poses: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "joint_order": list(ARM_JOINT_NAMES),
        "angle_unit": "degree",
        "poses": list(poses),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_pose(joint_degrees: np.ndarray, margin_deg: float) -> None:
    if joint_degrees.shape != (len(ARM_JOINT_NAMES),):
        raise ValueError("Each pose must contain five arm joint angles")
    for name, value in zip(ARM_JOINT_NAMES, joint_degrees, strict=True):
        limit = URDF_LIMITS[name]
        if not limit.lower_deg + margin_deg <= value <= limit.upper_deg - margin_deg:
            raise ValueError(
                f"{name}={value:.2f} deg is outside the safe range "
                f"{limit.lower_deg + margin_deg:.2f}..{limit.upper_deg - margin_deg:.2f}"
            )


class PoseRecordingPreview:
    """Show chessboard visibility while recording torque-off arm poses."""

    def __init__(
        self,
        camera: cv2.VideoCapture,
        robot: SO101Follower,
        mapper: SO101JointMapper,
        calibration: CameraCalibration,
        *,
        output: Path,
        target_count: int,
        joint_margin_deg: float,
        session_id: str,
        initial_poses: Sequence[dict[str, object]] = (),
    ) -> None:
        import tkinter as tk

        self.root = tk.Tk()
        self.root.title("SO-101 safe pose recording")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.help = tk.Label(
            self.root,
            text="Space/Enter: record pose   Q/Esc: finish",
            font=("Segoe UI", 11),
        )
        self.help.pack(pady=6)
        self.camera = camera
        self.robot = robot
        self.mapper = mapper
        self.calibration = calibration
        self.output = output
        self.target_count = target_count
        self.joint_margin_deg = joint_margin_deg
        self.session_id = session_id
        self.poses = [dict(pose) for pose in initial_poses]
        self.photo: ImageTk.PhotoImage | None = None
        self.record_requested = False
        self.running = True
        self.error: Exception | None = None
        self.output_backed_up = False

        self.root.bind("<space>", self.request_record)
        self.root.bind("<Return>", self.request_record)
        self.root.bind("q", self.close)
        self.root.bind("<Escape>", self.close)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def request_record(self, _event: object) -> None:
        self.record_requested = True

    def close(self, _event: object | None = None) -> None:
        if not self.running:
            return
        self.running = False
        self.root.destroy()

    def run(self) -> list[dict[str, object]]:
        self.root.after(0, self.update)
        self.root.mainloop()
        if self.error is not None:
            raise self.error
        return self.poses

    def update(self) -> None:
        if not self.running:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.error = RuntimeError("Camera frame read failed")
            self.close()
            return

        display = frame.copy()
        corners = find_chessboard_corners(frame, self.calibration.pattern_size)
        found = corners is not None
        if found:
            cv2.drawChessboardCorners(
                display, self.calibration.pattern_size, corners, True
            )

        status = (
            f"DETECTED  {len(self.poses)}/{self.target_count}"
            if found
            else f"NOT DETECTED  {len(self.poses)}/{self.target_count}"
        )
        color = (0, 180, 0) if found else (0, 0, 220)
        cv2.putText(
            display,
            status,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

        if self.record_requested:
            self.record_requested = False
            if not found:
                print("Chessboard is not fully detected; pose was not recorded.")
            else:
                try:
                    raw = read_raw_motors(self.robot, ARM_JOINT_NAMES)
                    outside = raw_outside_calibration(self.mapper, raw)
                    if outside:
                        raise RuntimeError(
                            f"Raw positions outside calibration: {', '.join(outside)}"
                        )
                    joint_degrees = arm_degrees_from_raw(self.mapper, raw)
                    validate_pose(joint_degrees, self.joint_margin_deg)
                    pose = {
                        "name": f"pose_{len(self.poses) + 1:02d}",
                        "joint_degrees": np.round(joint_degrees, 6).tolist(),
                    }
                    if not self.output_backed_up:
                        backup = backup_existing_file(self.output, self.session_id)
                        if backup is not None:
                            print(f"Previous poses archived: {backup}")
                        self.output_backed_up = True
                    self.poses.append(pose)
                    save_recorded_poses(self.output, self.poses)
                    print("Recorded:", np.round(joint_degrees, 2))
                    if len(self.poses) >= self.target_count:
                        print("Target pose count reached.")
                        self.root.after(250, self.close)
                except Exception as exc:
                    self.error = exc
                    self.close()
                    return

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.label.configure(image=self.photo)
        self.root.after(30, self.update)


class CapturePreview:
    """Display the camera's latest frame without consuming capture events."""

    def __init__(self, camera: object, emergency: EmergencyStop) -> None:
        import tkinter as tk

        self.root = tk.Tk()
        self.root.title("SO-101 hand-eye capture")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.status = tk.Label(self.root, text="Camera preview", font=("Segoe UI", 11))
        self.status.pack(pady=6)
        self.camera = camera
        self.emergency = emergency
        self.photo: ImageTk.PhotoImage | None = None
        self.next_requested = False
        self.last_frame_at = 0.0
        self.closed = False
        self.root.bind("<Return>", self.request_next)
        self.root.bind("<space>", self.stop)
        self.root.bind("q", self.stop)
        self.root.bind("<Escape>", self.stop)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

    def request_next(self, _event: object) -> None:
        self.next_requested = True

    def stop(self, _event: object | None = None) -> None:
        self.emergency.requested.set()

    def set_status(self, message: str) -> None:
        self.status.configure(text=message)

    def pump(self) -> None:
        if self.emergency.requested.is_set():
            raise KeyboardInterrupt("Emergency stop")
        now = time.monotonic()
        if now - self.last_frame_at >= 1 / 15:
            frame_rgb = self.camera.read_latest(max_age_ms=1000)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb.copy()))
            self.label.configure(image=self.photo)
            self.last_frame_at = now
        self.root.update()
        if self.emergency.requested.is_set():
            raise KeyboardInterrupt("Emergency stop")

    def wait_for_next(self, name: str, index: int, count: int) -> None:
        self.next_requested = False
        self.set_status(f"Ready for {name} ({index}/{count}). Press Enter here to move.")
        while not self.next_requested:
            self.pump()
            time.sleep(0.03)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.root.destroy()


def run_record_poses(args: argparse.Namespace) -> int:
    initial_poses: list[dict[str, object]] = []
    if args.output.is_file() and not args.overwrite:
        initial_poses = load_recorded_poses(args.output)
        for pose in initial_poses:
            joint_degrees = np.asarray(pose["joint_degrees"], dtype=float)
            validate_pose(joint_degrees, args.joint_margin_deg)
        if len(initial_poses) > args.count:
            raise ValueError(
                f"Existing file has {len(initial_poses)} poses, more than target "
                f"--count {args.count}. Increase --count or use --overwrite."
            )
        if len(initial_poses) == args.count:
            print(f"Target already reached: {len(initial_poses)}/{args.count} poses")
            print(f"Existing pose file was not changed: {args.output}")
            return 0
        print(
            f"Resuming {args.output}: {len(initial_poses)}/{args.count} poses already recorded."
        )
    elif args.output.is_file():
        print("Starting a new pose set; the old file will be archived after the first save.")

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    mapper = load_mapper(robot)
    calibration = CameraCalibration.load(args.intrinsics)
    session_id = make_session_id()
    if calibration.image_size != (args.width, args.height):
        raise ValueError(
            f"Camera calibration is {calibration.image_size}, requested "
            f"{(args.width, args.height)}"
        )
    camera = open_camera(args.camera_index, args.width, args.height, args.fps)
    try:
        robot.bus.connect(handshake=False)
        robot.bus.disable_torque(list(ARM_JOINT_NAMES), num_retry=5)
        print("Arm torque is OFF. Support the arm while positioning it.")
        print("Use large orientation changes around at least two different rotation axes.")
        print("Keep the complete chessboard visible; record in the preview with Space/Enter.")
        poses = PoseRecordingPreview(
            camera,
            robot,
            mapper,
            calibration,
            output=args.output,
            target_count=args.count,
            joint_margin_deg=args.joint_margin_deg,
            session_id=session_id,
            initial_poses=initial_poses,
        ).run()
        if len(poses) < args.count:
            print(f"Stopped early after {len(poses)}/{args.count} poses.")
        print(f"\nSaved {len(poses)} poses: {args.output}")
        return 0
    finally:
        camera.release()
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)


def measured_arm_degrees(robot: SO101Follower, mapper: SO101JointMapper) -> np.ndarray:
    raw = read_raw_motors(robot, ARM_JOINT_NAMES)
    outside = raw_outside_calibration(mapper, raw)
    if outside:
        raise RuntimeError(f"Raw positions outside calibration: {', '.join(outside)}")
    return arm_degrees_from_raw(mapper, raw)


def wait_for_target(
    robot: SO101Follower,
    mapper: SO101JointMapper,
    target: np.ndarray,
    emergency: EmergencyStop,
    *,
    tolerance_deg: float,
    timeout_s: float,
    progress: Callable[[], None] | None = None,
) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    while True:
        if progress is not None:
            progress()
        if emergency.requested.is_set():
            raise KeyboardInterrupt("Emergency stop")
        measured = measured_arm_degrees(robot, mapper)
        absolute_errors = np.abs(measured - target)
        worst_index = int(np.argmax(absolute_errors))
        error = float(absolute_errors[worst_index])
        if error <= tolerance_deg:
            return measured
        if time.monotonic() >= deadline:
            joint = ARM_JOINT_NAMES[worst_index]
            details = ", ".join(
                f"{name}={joint_error:.2f}"
                for name, joint_error in zip(
                    ARM_JOINT_NAMES, absolute_errors, strict=True
                )
            )
            raise SampleRejectedError(
                f"Pose timeout; {joint} has the largest error: target "
                f"{target[worst_index]:.2f} deg, measured "
                f"{measured[worst_index]:.2f} deg, error {error:.2f} deg. "
                f"All absolute errors (deg): {details}"
            )
        time.sleep(0.05)


def move_joints_smoothly(
    robot: SO101Follower,
    mapper: SO101JointMapper,
    start: np.ndarray,
    target: np.ndarray,
    gripper_position: float,
    emergency: EmergencyStop,
    *,
    speed_deg_s: float,
    control_hz: float,
    progress: Callable[[], None] | None = None,
) -> None:
    max_delta = float(np.max(np.abs(target - start)))
    steps = max(1, math.ceil(max_delta / (speed_deg_s / control_hz)))
    next_tick = time.perf_counter()
    for index in range(1, steps + 1):
        if progress is not None:
            progress()
        if emergency.requested.is_set():
            raise KeyboardInterrupt("Emergency stop")
        joint_degrees = start + (target - start) * (index / steps)
        action = {
            f"{name}.pos": mapper.urdf_degrees_to_normalized(name, value)
            for name, value in zip(ARM_JOINT_NAMES, joint_degrees, strict=True)
        }
        action["gripper.pos"] = gripper_position
        robot.send_action(action)
        next_tick = max(next_tick, time.perf_counter()) + 1.0 / control_hz
        delay = next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)


def wait_while_previewing(
    duration_s: float,
    emergency: EmergencyStop,
    progress: Callable[[], None] | None = None,
) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        if progress is not None:
            progress()
        if emergency.requested.is_set():
            raise KeyboardInterrupt("Emergency stop")
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


def capture_best_observation(
    camera: object,
    calibration: CameraCalibration,
    *,
    frame_count: int,
    timeout_ms: int,
    progress: Callable[[], None] | None = None,
) -> tuple[np.ndarray, object]:
    candidates: list[tuple[float, np.ndarray, object]] = []
    for _ in range(frame_count):
        if progress is not None:
            progress()
        frame_rgb = camera.async_read(timeout_ms=timeout_ms)
        observation = estimate_target_pose(frame_rgb, calibration)
        if progress is not None:
            progress()
        if observation is not None:
            candidates.append((observation.reprojection_error_px, frame_rgb.copy(), observation))
    if not candidates:
        raise SampleRejectedError("Chessboard was not detected in any capture frame")
    _, frame, observation = min(candidates, key=lambda item: item[0])
    return frame, observation


def save_sample_images(
    image_path: Path,
    axes_path: Path,
    frame_rgb: np.ndarray,
    observation: object,
    calibration: CameraCalibration,
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    axes_path.parent.mkdir(parents=True, exist_ok=True)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(image_path), frame_bgr):
        raise RuntimeError(f"Could not save {image_path}")

    annotated = frame_bgr.copy()
    cv2.drawChessboardCorners(
        annotated, calibration.pattern_size, observation.corners, True
    )
    rotation_vector, _ = cv2.Rodrigues(observation.camera_T_target[:3, :3])
    cv2.drawFrameAxes(
        annotated,
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        rotation_vector,
        observation.camera_T_target[:3, 3:4],
        calibration.square_size_mm / 1000.0 * 3.0,
        2,
    )
    if not cv2.imwrite(str(axes_path), annotated):
        raise RuntimeError(f"Could not save {axes_path}")


def run_capture(args: argparse.Namespace) -> int:
    poses = load_recorded_poses(args.poses)
    session_id = make_session_id()
    session_images_dir = args.images_dir / session_id
    session_axes_dir = args.axes_dir / session_id
    if len(poses) < args.minimum_samples:
        raise ValueError(
            f"Pose file has {len(poses)} poses; at least {args.minimum_samples} are required"
        )
    calibration = CameraCalibration.load(args.intrinsics)
    if args.square_size_mm is not None:
        calibration = replace(calibration, square_size_mm=args.square_size_mm)
    if calibration.image_size != (args.width, args.height):
        raise ValueError(
            f"Camera calibration is {calibration.image_size}, requested {(args.width, args.height)}"
        )
    fk, fk_backend = create_fk_solver(args.urdf)

    camera_config = OpenCVCameraConfig(
        index_or_path=args.camera_index,
        fps=args.fps,
        width=args.width,
        height=args.height,
        color_mode=ColorMode.RGB,
        warmup_s=1,
    )
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            cameras={"wrist": camera_config},
            disable_torque_on_disconnect=False,
            max_relative_target=None,
        )
    )
    mapper = load_mapper(robot)
    targets: list[tuple[str, np.ndarray]] = []
    for pose in poses:
        target = np.asarray(pose["joint_degrees"], dtype=float)
        validate_pose(target, args.joint_margin_deg)
        targets.append((str(pose["name"]), target))

    samples: list[HandEyeSample] = []
    recapture_names = tuple(dict.fromkeys(args.recapture)) if args.recapture else ()
    if recapture_names:
        available_pose_names = {name for name, _ in targets}
        unknown_poses = sorted(set(recapture_names) - available_pose_names)
        if unknown_poses:
            raise ValueError(f"Unknown recorded poses: {', '.join(unknown_poses)}")
        if not args.output.is_file():
            raise FileNotFoundError(
                f"Cannot recapture selected poses without an existing dataset: {args.output}"
            )
        existing_dataset, samples = load_dataset(args.output)
        existing_target = existing_dataset.get("target")
        if not isinstance(existing_target, dict):
            raise ValueError(
                "Existing dataset has no target metadata; run a full capture before selective recapture"
            )
        existing_corners = existing_target["inner_corners"]
        existing_pattern_size = (
            int(existing_corners["columns"]),
            int(existing_corners["rows"]),
        )
        existing_square_size_mm = float(existing_target["square_size_mm"])
        if existing_pattern_size != calibration.pattern_size or not np.isclose(
            existing_square_size_mm, calibration.square_size_mm
        ):
            raise ValueError(
                "Existing dataset target geometry does not match this capture: "
                f"existing {existing_pattern_size} at {existing_square_size_mm:.3f} mm, "
                f"requested {calibration.pattern_size} at {calibration.square_size_mm:.3f} mm"
            )
        if not existing_target.get("corner_order_canonicalized", False):
            raise ValueError(
                "Existing dataset corner order is not canonicalized; run a full capture first"
            )
        existing_sample_names = {sample.name for sample in samples}
        missing_samples = sorted(set(recapture_names) - existing_sample_names)
        if missing_samples:
            raise ValueError(
                f"Existing dataset has no samples to replace: {', '.join(missing_samples)}"
            )
        selected = set(recapture_names)
        targets = [(name, target) for name, target in targets if name in selected]

    emergency = EmergencyStop()
    listener = keyboard.Listener(on_press=emergency.on_press)
    torque_enabled = False
    camera = robot.cameras["wrist"]
    skipped: list[tuple[str, str]] = []
    replaced_names: list[str] = []
    output_backed_up = False
    preview: CapturePreview | None = None
    robot.bus.connect(handshake=False)
    try:
        robot.bus.port_handler.clearPort()
        time.sleep(0.1)
        current = measured_arm_degrees(robot, mapper)
        raw_all = read_raw_motors(robot, ALL_MOTOR_NAMES)
        current_goals = {
            name: mapper.raw_to_normalized(name, raw_all[name]) for name in ALL_MOTOR_NAMES
        }
        gripper_position = current_goals["gripper"]
        camera.connect()

        print(f"FK backend: {fk_backend}")
        print(f"Hand-eye target square size: {calibration.square_size_mm:.3f} mm")
        if recapture_names:
            print("Samples to replace:", ", ".join(recapture_names))
            print(
                "Movement and capture sequence:",
                " -> ".join(name for name, _ in targets),
            )
        else:
            print(f"Samples to collect: {len(targets)}")
        print("Space is the emergency stop. Keep the servo power switch within reach.")
        phrase = (
            f"START HAND EYE RECAPTURE {len(recapture_names)}"
            if recapture_names
            else f"START HAND EYE CAPTURE {len(targets)}"
        )
        if input(f'Type exactly "{phrase}" to enable torque: ').strip() != phrase:
            print("Confirmation did not match. No movement command was sent.")
            return 0

        if args.preview:
            preview = CapturePreview(camera, emergency)
            preview.pump()
        robot.bus.sync_write("Goal_Position", current_goals)
        robot.bus.enable_torque(list(ARM_JOINT_NAMES), num_retry=5)
        torque_enabled = True
        listener.start()

        for index, (name, target) in enumerate(targets, start=1):
            if not args.automatic:
                if preview is not None:
                    preview.wait_for_next(name, index, len(targets))
                else:
                    input(f"\nReady for {name} ({index}/{len(targets)}). Press ENTER to move...")
            try:
                if preview is not None:
                    preview.set_status(f"Moving to {name} ({index}/{len(targets)})")
                move_joints_smoothly(
                    robot,
                    mapper,
                    current,
                    target,
                    gripper_position,
                    emergency,
                    speed_deg_s=args.speed_deg_s,
                    control_hz=args.control_hz,
                    progress=preview.pump if preview is not None else None,
                )
                if preview is not None:
                    preview.set_status(f"Checking {name} ({index}/{len(targets)})")
                measured = wait_for_target(
                    robot,
                    mapper,
                    target,
                    emergency,
                    tolerance_deg=args.joint_tolerance_deg,
                    timeout_s=args.pose_timeout_s,
                    progress=preview.pump if preview is not None else None,
                )
                if preview is not None:
                    preview.set_status(f"Settling at {name} ({index}/{len(targets)})")
                wait_while_previewing(
                    args.settle_s,
                    emergency,
                    progress=preview.pump if preview is not None else None,
                )
                if preview is not None:
                    preview.set_status(f"Capturing {name} ({index}/{len(targets)})")
                frame_rgb, observation = capture_best_observation(
                    camera,
                    calibration,
                    frame_count=args.capture_frames,
                    timeout_ms=args.camera_timeout_ms,
                    progress=preview.pump if preview is not None else None,
                )
                if observation.reprojection_error_px > args.max_pnp_error_px:
                    raise SampleRejectedError(
                        f"PnP error is {observation.reprojection_error_px:.3f} px, "
                        f"above limit {args.max_pnp_error_px:.3f} px"
                    )
                measured = measured_arm_degrees(robot, mapper)
                base_T_gripper = fk.forward_kinematics(measured)
                canonical_target, corner_order_flipped = canonicalize_target_frame(
                    samples,
                    base_T_gripper,
                    observation.camera_T_target,
                    pattern_size=calibration.pattern_size,
                    square_size_mm=calibration.square_size_mm,
                )
                if corner_order_flipped:
                    observation = replace(
                        observation, camera_T_target=canonical_target
                    )
                    print(f"Canonicalized 180-degree corner order for {name}")
                if not output_backed_up:
                    backup = backup_existing_file(args.output, session_id)
                    if backup is not None:
                        print(f"Previous samples archived: {backup}")
                    output_backed_up = True
                image_path = session_images_dir / f"{name}.png"
                axes_path = session_axes_dir / f"{name}_axes.png"
                save_sample_images(
                    image_path, axes_path, frame_rgb, observation, calibration
                )
                new_sample = HandEyeSample(
                    name=name,
                    image=str(image_path.resolve()),
                    joint_degrees=tuple(float(value) for value in measured),
                    base_T_gripper=base_T_gripper,
                    camera_T_target=observation.camera_T_target,
                    pnp_reprojection_error_px=observation.reprojection_error_px,
                )
                if recapture_names:
                    sample_by_name = {sample.name: sample for sample in samples}
                    sample_by_name[name] = new_sample
                    pose_order = [str(pose["name"]) for pose in poses]
                    samples = [
                        sample_by_name[pose_name]
                        for pose_name in pose_order
                        if pose_name in sample_by_name
                    ]
                    replaced_names.append(name)
                else:
                    samples.append(new_sample)
                save_dataset(
                    args.output,
                    samples,
                    camera_calibration_path=args.intrinsics,
                    urdf_path=args.urdf,
                    target_pattern_size=calibration.pattern_size,
                    target_square_size_mm=calibration.square_size_mm,
                    corner_order_canonicalized=True,
                )
                tracking_errors = np.abs(measured - target)
                worst_index = int(np.argmax(tracking_errors))
                print(
                    f"Captured {name}: PnP error "
                    f"{observation.reprojection_error_px:.3f} px; max joint error "
                    f"{tracking_errors[worst_index]:.2f} deg "
                    f"({ARM_JOINT_NAMES[worst_index]})"
                )
                if preview is not None:
                    preview.set_status(
                        f"Captured {name}: PnP {observation.reprojection_error_px:.2f} px"
                    )
                current = measured
            except SampleRejectedError as exc:
                skipped.append((name, str(exc)))
                print(f"SKIPPED {name}: {exc}", file=sys.stderr)
                if preview is not None:
                    preview.set_status(f"Skipped {name}; see terminal for details")
                current = measured_arm_degrees(robot, mapper)
                continue

        completed_count = len(replaced_names) if recapture_names else len(samples)
        action = "replaced" if recapture_names else "captured"
        print(f"\nCapture complete: {completed_count} {action}, {len(skipped)} skipped.")
        for name, reason in skipped:
            print(f"  SKIPPED {name}: {reason}")
        if completed_count:
            print(f"Saved synchronized samples: {args.output}")
        else:
            print("No synchronized sample was changed.")
        if len(samples) < args.minimum_samples:
            print(
                f"WARNING: only {len(samples)} valid samples; at least "
                f"{args.minimum_samples} are required before solve."
            )
        return 0
    finally:
        if torque_enabled and robot.bus.is_connected:
            try:
                robot.bus.disable_torque(list(ARM_JOINT_NAMES), num_retry=5)
                print("Arm torque disabled.")
            except Exception as exc:
                print(f"WARNING: torque disable failed: {exc}", file=sys.stderr)
        if listener.is_alive():
            listener.stop()
            listener.join(timeout=1.0)
        try:
            if camera.is_connected:
                camera.disconnect()
        finally:
            if robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=False)
            if preview is not None:
                preview.close()


def create_summary_image(
    path: Path,
    result: object,
    actual_origin_mm: np.ndarray | None,
) -> None:
    canvas = np.full((620, 1080, 3), 255, dtype=np.uint8)
    lines = [
        "SO-101 Eye-in-Hand Calibration",
        f"Method: OpenCV {result.method.upper()}    Samples: {result.sample_count}",
        "gripper_T_camera: camera origin expressed in gripper frame",
        "Translation xyz (mm): "
        + np.array2string(result.camera_translation_in_gripper_mm, precision=2),
        "Rotation rpy XYZ (deg): "
        + np.array2string(result.camera_rpy_in_gripper_deg, precision=2),
        "Estimated board origin in base xyz (mm): "
        + np.array2string(result.base_T_target[:3, 3] * 1000.0, precision=2),
        "Board-origin RMSE xyz (mm): "
        + np.array2string(result.board_origin_rmse_mm, precision=2),
        "Board-origin max abs xyz (mm): "
        + np.array2string(result.board_origin_max_abs_mm, precision=2),
        f"Board rotation RMSE (deg): {result.board_rotation_rmse_deg:.3f}",
    ]
    if actual_origin_mm is not None:
        error = result.base_T_target[:3, 3] * 1000.0 - actual_origin_mm
        lines.extend(
            [
                "Measured board origin xyz (mm): "
                + np.array2string(actual_origin_mm, precision=2),
                "Estimated - measured xyz (mm): " + np.array2string(error, precision=2),
                "5 cm per-axis check: " + ("PASS" if np.all(np.abs(error) <= 50.0) else "FAIL"),
            ]
        )
    for index, line in enumerate(lines):
        scale = 0.9 if index == 0 else 0.62
        thickness = 2 if index == 0 else 1
        cv2.putText(
            canvas,
            line,
            (35, 55 + index * 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not save {path}")


def run_solve(args: argparse.Namespace) -> int:
    dataset, samples = load_dataset(args.dataset)
    if len(samples) < args.minimum_samples:
        raise ValueError(
            f"Dataset has {len(samples)} samples; at least {args.minimum_samples} are required"
        )
    target = dataset.get("target")
    if isinstance(target, dict):
        corners = target["inner_corners"]
        pattern_size = (int(corners["columns"]), int(corners["rows"]))
        recorded_square_size_mm = float(target["square_size_mm"])
        already_canonicalized = bool(target.get("corner_order_canonicalized", False))
    else:
        calibration_path = Path(str(dataset["camera_calibration"]))
        calibration = CameraCalibration.load(calibration_path)
        pattern_size = calibration.pattern_size
        recorded_square_size_mm = calibration.square_size_mm
        already_canonicalized = False

    pnp_square_size_mm = (
        args.pnp_square_size_mm
        if args.pnp_square_size_mm is not None
        else recorded_square_size_mm
    )
    actual_square_size_mm = (
        args.square_size_mm
        if args.square_size_mm is not None
        else recorded_square_size_mm
    )
    canonicalize_corners = not args.no_canonicalize_corners and not already_canonicalized
    samples, flipped_names = prepare_hand_eye_samples(
        samples,
        pattern_size=pattern_size,
        pnp_square_size_mm=pnp_square_size_mm,
        actual_square_size_mm=actual_square_size_mm,
        canonicalize_corners=canonicalize_corners,
    )
    print(
        "Target geometry: "
        f"{pattern_size[0]}x{pattern_size[1]} inner corners, "
        f"PnP square {pnp_square_size_mm:.3f} mm, "
        f"actual square {actual_square_size_mm:.3f} mm"
    )
    if flipped_names:
        print("Canonicalized 180-degree corner order:", ", ".join(flipped_names))
    elif already_canonicalized:
        print("Corner order was canonicalized during capture.")
    elif canonicalize_corners:
        print("No 180-degree corner-order changes were required.")
    all_results = solve_all_methods(samples)
    if not all_results:
        raise RuntimeError("All OpenCV hand-eye methods failed")

    print("Method comparison (fixed-board consistency):")
    for name, candidate in all_results.items():
        print(
            f"  {name:<10} origin RMSE xyz mm="
            f"{np.round(candidate.board_origin_rmse_mm, 2)}  "
            f"rotation RMS={candidate.board_rotation_rmse_deg:.3f} deg"
        )
    if args.method not in all_results:
        raise RuntimeError(f"Requested method {args.method} failed")
    result = solve_hand_eye(samples, args.method)

    actual_origin = (
        np.asarray(args.measured_board_origin_mm, dtype=float)
        if args.measured_board_origin_mm is not None
        else None
    )
    payload = result.to_json()
    payload["preprocessing"] = {
        "source_dataset": str(args.dataset.resolve()),
        "pattern_inner_corners": {
            "columns": pattern_size[0],
            "rows": pattern_size[1],
        },
        "pnp_square_size_mm": pnp_square_size_mm,
        "actual_square_size_mm": actual_square_size_mm,
        "translation_scale": actual_square_size_mm / pnp_square_size_mm,
        "corner_order_canonicalized": canonicalize_corners or already_canonicalized,
        "flipped_samples": flipped_names,
    }
    if actual_origin is not None:
        axis_error = result.base_T_target[:3, 3] * 1000.0 - actual_origin
        payload["measured_board_origin_in_base_mm"] = actual_origin.tolist()
        payload["board_origin_error_xyz_mm"] = axis_error.tolist()
        payload["within_50_mm_each_axis"] = bool(np.all(np.abs(axis_error) <= 50.0))
    session_id = make_session_id()
    result_backup = backup_existing_file(args.output, session_id)
    summary_backup = backup_existing_file(args.summary_image, session_id)
    if result_backup is not None:
        print(f"Previous result archived: {result_backup}")
    if summary_backup is not None:
        print(f"Previous summary archived: {summary_backup}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    create_summary_image(args.summary_image, result, actual_origin)

    print("\nSelected result:")
    print(f"  Method: {result.method}")
    print(
        "  Camera origin in gripper xyz (mm):",
        np.round(result.camera_translation_in_gripper_mm, 2),
    )
    print("  Camera orientation in gripper rpy XYZ (deg):", np.round(result.camera_rpy_in_gripper_deg, 2))
    print("  Estimated board origin in base xyz (mm):", np.round(result.base_T_target[:3, 3] * 1000.0, 2))
    print("  Board-origin RMSE xyz (mm):", np.round(result.board_origin_rmse_mm, 2))
    if actual_origin is not None:
        error = result.base_T_target[:3, 3] * 1000.0 - actual_origin
        print("  Estimated - measured xyz (mm):", np.round(error, 2))
        print("  5 cm per-axis check:", "PASS" if np.all(np.abs(error) <= 50.0) else "FAIL")
    else:
        print(
            "  Accuracy check: NOT RUN (provide three measured millimetre values, "
            "for example --measured-board-origin-mm 300 -50 20)"
        )
    print(f"Saved result: {args.output}")
    print(f"Saved screenshot: {args.summary_image}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SO-101 eye-in-hand calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record-poses", help="Record manually verified safe poses")
    record.add_argument("--port", default="COM5")
    record.add_argument("--robot-id", default="scs215_com5")
    record.add_argument("--camera-index", type=int, default=1)
    record.add_argument("--width", type=int, default=640)
    record.add_argument("--height", type=int, default=480)
    record.add_argument("--fps", type=int, default=15)
    record.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    record.add_argument(
        "--count", type=int, default=15, help="Final total pose count, including resumed poses"
    )
    record.add_argument("--joint-margin-deg", type=float, default=4.0)
    record.add_argument("--output", type=Path, default=DEFAULT_ROOT / "poses.json")
    record.add_argument(
        "--overwrite",
        action="store_true",
        help="Start from pose 01 instead of resuming the existing output file",
    )

    capture = subparsers.add_parser("capture", help="Replay poses and capture synchronized samples")
    capture.add_argument("--port", default="COM5")
    capture.add_argument("--robot-id", default="scs215_com5")
    capture.add_argument("--camera-index", type=int, default=1)
    capture.add_argument("--width", type=int, default=640)
    capture.add_argument("--height", type=int, default=480)
    capture.add_argument("--fps", type=int, default=15)
    capture.add_argument("--camera-timeout-ms", type=int, default=5000)
    capture.add_argument("--poses", type=Path, default=DEFAULT_ROOT / "poses.json")
    capture.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    capture.add_argument(
        "--square-size-mm",
        type=float,
        help="Physical hand-eye target square size; defaults to the intrinsics JSON value",
    )
    capture.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    capture.add_argument("--output", type=Path, default=DEFAULT_ROOT / "samples.json")
    capture.add_argument(
        "--recapture",
        nargs="+",
        metavar="POSE_NAME",
        help="Recapture named poses and replace only those samples in an existing dataset",
    )
    capture.add_argument("--images-dir", type=Path, default=DEFAULT_ROOT / "images")
    capture.add_argument("--axes-dir", type=Path, default=DEFAULT_ROOT / "axes")
    capture.add_argument("--minimum-samples", type=int, default=12)
    capture.add_argument("--speed-deg-s", type=float, default=12.0)
    capture.add_argument("--control-hz", type=float, default=20.0)
    capture.add_argument("--joint-margin-deg", type=float, default=4.0)
    capture.add_argument(
        "--joint-tolerance-deg",
        type=float,
        default=10.0,
        help="Maximum commanded-vs-measured error before capture (default: 10 deg for SCS215)",
    )
    capture.add_argument("--pose-timeout-s", type=float, default=8.0)
    capture.add_argument("--settle-s", type=float, default=1.5)
    capture.add_argument("--capture-frames", type=int, default=8)
    capture.add_argument("--max-pnp-error-px", type=float, default=1.5)
    capture.add_argument("--automatic", action="store_true")
    capture.add_argument(
        "--no-preview", dest="preview", action="store_false",
        help="Disable the live camera window (for headless sessions)",
    )

    solve = subparsers.add_parser("solve", help="Solve and evaluate gripper_T_camera")
    solve.add_argument("--dataset", type=Path, default=DEFAULT_ROOT / "samples.json")
    solve.add_argument("--method", choices=tuple(HAND_EYE_METHODS), default="park")
    solve.add_argument("--minimum-samples", type=int, default=12)
    solve.add_argument(
        "--square-size-mm",
        type=float,
        help="Actual target square size; rescales legacy PnP translations",
    )
    solve.add_argument(
        "--pnp-square-size-mm",
        type=float,
        help="Square size originally used by PnP; normally read from dataset metadata",
    )
    solve.add_argument(
        "--no-canonicalize-corners",
        action="store_true",
        help="Disable FK-based correction of symmetric-board 180-degree corner reversal",
    )
    solve.add_argument(
        "--measured-board-origin-mm",
        nargs=3,
        type=float,
        metavar=("X_MM", "Y_MM", "Z_MM"),
        help="Measured first inner-corner position in the robot base frame, in millimetres",
    )
    solve.add_argument("--output", type=Path, default=DEFAULT_ROOT / "hand_eye_result.json")
    solve.add_argument("--summary-image", type=Path, default=DEFAULT_ROOT / "hand_eye_summary.png")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "record-poses":
        if args.count < 12:
            raise ValueError("Record at least 12 poses")
        return run_record_poses(args)
    if args.command == "capture":
        return run_capture(args)
    return run_solve(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nEmergency stop or interruption requested.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
