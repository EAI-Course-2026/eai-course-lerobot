# Vision calibration results

This directory contains small, reviewable calibration results that are safe to
version. Raw frames, synchronized samples, local camera captures, and archived
runs remain under `../outputs/` and are intentionally ignored by Git.

## Current Eye-in-Hand result

`hand_eye_result_2026-10-17.json` was calculated from 15 synchronized samples
with OpenCV's Park method. The transform convention is
`destination_T_source`, so the saved `gripper_T_camera` is applied as:

```python
point_gripper = gripper_T_camera @ point_camera
```

The result is **provisional**. Its fixed-board origin RMSE is
`[9.84, 19.58, 74.09] mm`; the Z value is too high for precision control. Keep
the file for reproducibility and comparison, but repeat calibration after
improving multi-axis pose coverage and verifying the raw-to-URDF joint mapping.

The setup used an 8 x 6 inner-corner chessboard displayed on a fixed screen.
Five adjacent cells measured 98 mm, giving a physical cell size of 19.6 mm.
The camera was rigidly mounted on the gripper (Eye-in-Hand).
