#!/usr/bin/env python
"""Capture chessboards, solve intrinsics, and verify OpenCV calibration."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from camera_calibration import (
    CameraCalibration,
    calibrate_images,
    find_chessboard_corners,
)


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "outputs" / "camera_calibration"


def make_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def backup_existing_file(path: Path, session_id: str) -> Path | None:
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


def archive_capture_images(path: Path, session_id: str) -> Path | None:
    if not path.is_dir() or not any(
        item.suffix.lower() in {".png", ".jpg", ".jpeg"} for item in path.iterdir()
    ):
        return None
    destination = path.parent / "archive" / f"capture_{session_id}" / "images"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    return destination


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        camera = cv2.VideoCapture(index, backend)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_FPS, fps)
        if camera.isOpened():
            ok, _ = camera.read()
            if ok:
                actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Camera {index}: {actual_width}x{actual_height}")
                return camera
        camera.release()
    raise RuntimeError(
        f"Cannot open camera {index}. Close other camera programs and run "
        "lerobot-find-cameras opencv."
    )


class CameraPreview:
    """Tk preview compatible with the repository's headless OpenCV wheel."""

    def __init__(
        self,
        camera: cv2.VideoCapture,
        *,
        mode: str,
        pattern_size: tuple[int, int] | None = None,
        output_dir: Path | None = None,
        target_count: int = 20,
        calibration: CameraCalibration | None = None,
        alpha: float = 0.0,
    ) -> None:
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("OpenCV camera calibration")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.help = tk.Label(
            self.root,
            text="Space: capture   Q/Esc: finish" if mode == "capture" else "Q/Esc: finish",
            font=("Segoe UI", 11),
        )
        self.help.pack(pady=6)
        self.camera = camera
        self.mode = mode
        self.pattern_size = pattern_size
        self.output_dir = output_dir
        self.target_count = target_count
        self.calibration = calibration
        self.alpha = alpha
        self.latest_frame: np.ndarray | None = None
        self.latest_corners: np.ndarray | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.capture_requested = False
        existing_indices = []
        if output_dir is not None:
            for path in output_dir.glob("calib_*.png"):
                try:
                    existing_indices.append(int(path.stem.rsplit("_", 1)[1]))
                except (IndexError, ValueError):
                    continue
        self.next_image_index = max(existing_indices, default=0)
        self.session_count = 0
        self.running = True

        self.root.bind("<space>", self.request_capture)
        self.root.bind("q", self.close)
        self.root.bind("<Escape>", self.close)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def request_capture(self, _event: object) -> None:
        if self.mode == "capture":
            self.capture_requested = True

    def close(self, _event: object | None = None) -> None:
        self.running = False
        self.root.destroy()

    def run(self) -> None:
        self.root.after(0, self.update)
        self.root.mainloop()

    def update(self) -> None:
        if not self.running:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.close()
            raise RuntimeError("Camera frame read failed")
        self.latest_frame = frame

        if self.mode == "capture":
            assert self.pattern_size is not None
            display = frame.copy()
            self.latest_corners = find_chessboard_corners(frame, self.pattern_size)
            found = self.latest_corners is not None
            if found:
                cv2.drawChessboardCorners(
                    display, self.pattern_size, self.latest_corners, True
                )
            status = (
                f"DETECTED  {self.session_count}/{self.target_count}"
                if found
                else f"NOT DETECTED  {self.session_count}/{self.target_count}"
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
            if self.capture_requested:
                self.capture_requested = False
                if found:
                    assert self.output_dir is not None
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    self.next_image_index += 1
                    self.session_count += 1
                    path = self.output_dir / f"calib_{self.next_image_index:03d}.png"
                    if not cv2.imwrite(str(path), frame):
                        raise RuntimeError(f"Failed to save {path}")
                    print(f"Saved {path.name}")
                    if self.session_count >= self.target_count:
                        print("Target image count reached.")
                        self.running = False
                        self.root.after(250, self.close)
                else:
                    print("Pattern not found; image was not saved.")
        else:
            assert self.calibration is not None
            corrected = self.calibration.undistort(frame, alpha=self.alpha)
            cv2.putText(
                frame,
                "RAW",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                corrected,
                "UNDISTORTED",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 180, 0),
                2,
                cv2.LINE_AA,
            )
            display = np.hstack((frame, corrected))

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image=image)
        self.label.configure(image=self.photo)
        self.root.after(30, self.update)


def add_camera_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenCV chessboard intrinsic camera calibration"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Capture chessboard images")
    add_camera_arguments(capture)
    capture.add_argument("--columns", type=int, default=8, help="Inner-corner columns")
    capture.add_argument("--rows", type=int, default=6, help="Inner-corner rows")
    capture.add_argument("--target-count", type=int, default=20)
    capture.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "images")
    capture.add_argument(
        "--append",
        action="store_true",
        help="Add images to the current batch instead of archiving it",
    )

    solve = subparsers.add_parser("solve", help="Calculate and save camera intrinsics")
    solve.add_argument("--columns", type=int, default=8, help="Inner-corner columns")
    solve.add_argument("--rows", type=int, default=6, help="Inner-corner rows")
    solve.add_argument("--square-size-mm", type=float, required=True)
    solve.add_argument("--minimum-views", type=int, default=10)
    solve.add_argument("--images", type=Path, default=DEFAULT_ROOT / "images")
    solve.add_argument("--output", type=Path, default=DEFAULT_ROOT / "camera_intrinsics.json")
    solve.add_argument("--debug-dir", type=Path, default=DEFAULT_ROOT / "debug")

    verify = subparsers.add_parser("verify", help="Compare raw and undistorted live video")
    add_camera_arguments(verify)
    verify.add_argument(
        "--calibration", type=Path, default=DEFAULT_ROOT / "camera_intrinsics.json"
    )
    verify.add_argument("--alpha", type=float, default=0.0, choices=(0.0, 1.0))
    return parser


def run_capture(args: argparse.Namespace) -> int:
    camera = open_camera(args.camera_index, args.width, args.height, args.fps)
    try:
        if not args.append:
            archived = archive_capture_images(args.output_dir, make_session_id())
            if archived is not None:
                print(f"Previous image batch archived: {archived}")
        preview = CameraPreview(
            camera,
            mode="capture",
            pattern_size=(args.columns, args.rows),
            output_dir=args.output_dir,
            target_count=args.target_count,
        )
        preview.run()
    finally:
        camera.release()
    print(f"Captured images: {args.output_dir}")
    return 0


def run_solve(args: argparse.Namespace) -> int:
    image_paths = sorted(args.images.glob("*.png")) + sorted(args.images.glob("*.jpg"))
    calibration, accepted = calibrate_images(
        image_paths,
        pattern_size=(args.columns, args.rows),
        square_size_mm=args.square_size_mm,
        minimum_views=args.minimum_views,
        debug_dir=args.debug_dir,
    )
    backup = backup_existing_file(args.output, make_session_id())
    if backup is not None:
        print(f"Previous intrinsics archived: {backup}")
    calibration.save(args.output)

    print(f"\nAccepted views: {len(accepted)}/{len(image_paths)}")
    print(f"RMS: {calibration.rms:.4f} px")
    print(f"Mean reprojection error: {calibration.mean_reprojection_error_px:.4f} px")
    print("Camera matrix:")
    print(calibration.camera_matrix)
    print("Distortion coefficients:")
    print(calibration.distortion_coefficients.reshape(-1))
    print(f"Saved: {args.output}")
    worst = max(enumerate(calibration.per_view_errors_px), key=lambda item: item[1])
    print(f"Worst view: {accepted[worst[0]].name} ({worst[1]:.4f} px)")
    if calibration.mean_reprojection_error_px > 1.0:
        print("WARNING: error is above 1 px; recapture sharper, more varied views.")
    return 0


def run_verify(args: argparse.Namespace) -> int:
    calibration = CameraCalibration.load(args.calibration)
    if calibration.image_size != (args.width, args.height):
        raise ValueError(
            f"Requested {args.width}x{args.height}, but calibration is "
            f"{calibration.image_size[0]}x{calibration.image_size[1]}"
        )
    camera = open_camera(args.camera_index, args.width, args.height, args.fps)
    try:
        CameraPreview(
            camera,
            mode="verify",
            calibration=calibration,
            alpha=args.alpha,
        ).run()
    finally:
        camera.release()
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "capture":
        return run_capture(args)
    if args.command == "solve":
        return run_solve(args)
    return run_verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
