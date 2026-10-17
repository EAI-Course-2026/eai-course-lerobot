#!/usr/bin/env python

import unittest
from unittest.mock import Mock, patch

import numpy as np

from calibrate_hand_eye import (
    CapturePreview,
    EmergencyStop,
    build_parser,
    capture_best_observation,
    move_joints_smoothly,
)


class CapturePreviewTest(unittest.TestCase):
    def test_capture_parser_accepts_named_recapture(self) -> None:
        args = build_parser().parse_args(
            ["capture", "--recapture", "pose_03", "pose_07", "pose_11"]
        )
        self.assertEqual(args.recapture, ["pose_03", "pose_07", "pose_11"])

    def test_preview_reads_latest_without_consuming_capture_event(self) -> None:
        preview = CapturePreview.__new__(CapturePreview)
        preview.camera = Mock()
        preview.camera.read_latest.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
        preview.emergency = EmergencyStop()
        preview.root = Mock()
        preview.label = Mock()
        preview.last_frame_at = -1.0

        with patch("calibrate_hand_eye.ImageTk.PhotoImage", return_value=object()):
            preview.pump()

        preview.camera.read_latest.assert_called_once_with(max_age_ms=1000)
        preview.camera.async_read.assert_not_called()
        preview.root.update.assert_called_once()

    def test_emergency_prevents_next_motion_command(self) -> None:
        emergency = EmergencyStop()
        robot = Mock()

        with self.assertRaises(KeyboardInterrupt):
            move_joints_smoothly(
                robot,
                Mock(),
                np.zeros(5),
                np.ones(5),
                50.0,
                emergency,
                speed_deg_s=12.0,
                control_hz=20.0,
                progress=emergency.requested.set,
            )

        robot.send_action.assert_not_called()

    def test_closing_preview_requests_emergency(self) -> None:
        preview = CapturePreview.__new__(CapturePreview)
        preview.emergency = EmergencyStop()
        preview.stop()

        self.assertTrue(preview.emergency.requested.is_set())
        with self.assertRaises(KeyboardInterrupt):
            preview.pump()

    def test_capture_still_chooses_lowest_reprojection_error(self) -> None:
        camera = Mock()
        frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (1, 2)]
        camera.async_read.side_effect = frames
        observations = [
            Mock(reprojection_error_px=0.8),
            Mock(reprojection_error_px=0.3),
        ]
        progress = Mock()

        with patch("calibrate_hand_eye.estimate_target_pose", side_effect=observations):
            frame, observation = capture_best_observation(
                camera,
                Mock(),
                frame_count=2,
                timeout_ms=500,
                progress=progress,
            )

        np.testing.assert_array_equal(frame, frames[1])
        self.assertIs(observation, observations[1])
        self.assertEqual(progress.call_count, 4)


if __name__ == "__main__":
    unittest.main()
