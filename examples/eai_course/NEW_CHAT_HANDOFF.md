# 新对话交接卡：EAI Course SO-101

截至 2026-10-17。请从本仓库现有实现继续开发。团队仓库：
<https://github.com/EAI-Course-2026/eai-course-lerobot>。
当前工作分支为 `feature/visual-closed-loop`。

## 1. 硬件和本机环境

| 项目 | 当前配置 |
| --- | --- |
| 机械臂 | SO-ARM101 follower；6 个 Feetech **SCS215** 舵机，不是原生 STS3215 |
| 舵机 ID | 1 shoulder_pan；2 shoulder_lift；3 elbow_flex；4 wrist_flex；5 wrist_roll；6 gripper |
| 总线 | Feetech SCS protocol 1，1,000,000 baud，1024 个编码器位置 |
| 本机串口 | 最近手眼采集使用 `COM6`；端口会变化，以设备管理器/探测结果为准 |
| LeRobot 标定 ID | `scs215_com5`；ID 是文件名，不要求与 COM 端口字符串同步改变 |
| 相机 | 安装在机械臂末端的 USB 相机；本机通常 `--camera-index 1`，640x480、15 FPS；`0` 曾是笔记本摄像头 |
| 标定板 | 8 列 x 6 行**内角点**；当前屏幕棋盘实测连续 5 格为 98 mm，即单格 19.6 mm |
| 系统 | Windows 11，Miniconda `lerobot`，Python 3.12.14 |
| GPU | NVIDIA RTX 5060 Laptop；上次验证 PyTorch `2.11.0+cu128`、CUDA 可用 |

团队目前共用一台实体机械臂。仓库中的参考标定
`examples/eai_course/calibration/scs215_com5.reference.json` 对应这台机器；
本机运行时 LeRobot 读取
`%USERPROFILE%\.cache\huggingface\lerobot\calibration\robots\so_follower\scs215_com5.json`。
舵盘、连杆或舵机更换后需重新标定。参考 raw 安全范围：

| ID | 关节 | raw min..max |
| ---: | --- | ---: |
| 1 | shoulder_pan | 183..897 |
| 2 | shoulder_lift | 64..737 |
| 3 | elbow_flex | 196..769 |
| 4 | wrist_flex | 70..681 |
| 5 | wrist_roll | 47..974 |
| 6 | gripper | 237..644 |

## 2. 仓库与目录

```text
lerobot/
|-- src/lerobot/
|   |-- motors/feetech/{feetech.py,tables.py}       SCS215 协议/寄存器适配
|   |-- motors/motors_bus.py                         电机总线适配
|   `-- robots/so_follower/{config_so_follower.py,so_follower.py}
|                                                  六舵机配置与 LeRobot 动作接口
|-- examples/eai_course/
|   |-- README.md, HANDOFF.md, setup_windows.cmd    安装与团队说明
|   |-- calibration/                                共享机械臂参考标定及安装说明
|   |-- week4/task1/                                六关节预设姿态与动作序列
|   |   |-- arm_config.py, record_pose.py, control_presets.py, poses.json
|   |-- week4/task2/                                关节映射、FK/IK、直线规划与键盘控制
|   |   |-- joint_mapping.py, urdf_fk.py, kinematics_backend.py
|   |   |-- position_ik.py, cartesian_planner.py
|   |   |-- run_steps3_to5.py, keyboard_control.py
|   |   |-- check_step1.py, check_step2_fk.py, reindex_id5.py
|   |   `-- so101_new_calib.urdf, test_*.py
|   |-- vision/                                     相机/手眼标定与黑球视觉闭环
|   |   |-- calibrate_camera.py, camera_calibration.py
|   |   |-- calibrate_hand_eye.py, hand_eye_calibration.py
|   |   |-- chessboard_8x6_25mm.svg, test_camera_calibration.py
|   |   |-- test_hand_eye_calibration.py, test_capture_preview.py
|   |   |-- results/                                已提交的标定结果摘要，不含原始图片
|   |   |-- visual_closed_loop.py, ball_detector.py, visual_math.py
|   |   |-- black_ball_target.html, test_vision.py, README.md, REPORT.md
|   |   `-- outputs/                                 本机运行数据，Git 忽略
|   `-- voice/README.md                             语音模块尚未实现
`-- pyproject.toml                                    LeRobot 及可选依赖
```

本地 Git 远端名需要注意：`course` 指团队 GitHub 仓库，`origin` 指
`huggingface/lerobot` 上游。新克隆的仓库可能使用不同远端名，先运行 `git remote -v`。

## 3. 代码接口与数据约定

- 硬件入口是 `SO101Follower(SO101FollowerConfig(port=..., id="scs215_com5"))`。
  关节指令使用 `robot.send_action({"shoulder_pan.pos": value, ...})`；
  其中适配后的手臂关节 LeRobot 值为 `-100..100`，夹爪为 `0..100`。
- `week4/task2/joint_mapping.py` 的 `SO101JointMapper` 负责
  raw `0..1023`、LeRobot 归一化值、URDF 角度之间的转换。
  `ARM_JOINT_NAMES` 仅含 ID 1-5；`ALL_MOTOR_NAMES` 还含 ID 6。
- `create_fk_solver(urdf_path)` 返回 `(solver, backend_name)`；
  `solver.forward_kinematics(five_joint_degrees)` 返回 4x4 的
  `base_T_gripper`。Windows 无 Placo 时使用 `urdf_fk.py` 的 NumPy FK。
- `position_ik.py` 的 `DampedLeastSquaresIK.solve(target_xyz_m, seed_deg)`
  返回 `IKResult`；只约束末端 **XYZ 位置**，不约束姿态，ID 6 不在 IK 链中。
- `CartesianLinePlanner.plan(start_five_joint_deg, target_xyz_m)` 返回
  `CartesianPlan`，包括 `waypoints`、`reached_target`、`ik_failed`；
  遇无解位置时保留原方向上尽可能远的可行点。
- `week4/task1` 保存六关节 `0..1` 预设姿态，按键触发同步运动或动作序列；
  与 Task 2 的 URDF 角度表示不同。复用时先在接口边界做转换。
- `BlackBallDetector.detect(frame_rgb)` 返回 `(BallDetection | None, mask)`；
  检测结果含中心、半径、置信度。`visual_math.py` 根据局部图像雅可比求小步位移。
- `visual_closed_loop.py` 的 `VisionArmRuntime` 已封装
  `connect(camera_only=...)`、`read_frame()`、`move_to(xyz_m)`、
  `move_delta(delta_xyz_m)`、`measured_joint_degrees()`、`shutdown()`；
  它目前仍属于黑球实验脚本，**还没有**供视觉和语音共用的稳定控制 API。
- `CameraCalibration.load(path)` 读取相机内参 JSON；`undistort(frame)` 可选。
  手眼采集使用原图角点及畸变系数做 PnP，未先去畸变。
- `hand_eye_calibration.py` 的 `HandEyeSample` 保存同步的
  `base_T_gripper` 与 `camera_T_target`。矩阵命名采用
  `destination_T_source`；`solve_hand_eye(samples, method="park")`
  返回 `gripper_T_camera`、估计的固定板 `base_T_target` 和一致性误差。
- 当前配置是 **Eye-in-Hand**：相机刚性安装在末端，棋盘与机械臂基座固定。
  `gripper_T_camera` 将相机坐标转换到末端坐标，即
  `point_gripper = gripper_T_camera @ point_camera`。

## 4. 当前状态与本机数据

- 电机适配、预设动作、位置 IK、笛卡尔直线规划、键盘末端控制和黑球视觉闭环已有代码。
- `vision/outputs/camera_calibration/camera_intrinsics.json` 是本机 640x480、
  8x6 棋盘的内参结果：RMS 约 `0.943 px`，平均重投影误差约 `0.728 px`。
  本机结果没有推送到 GitHub；换相机、分辨率或焦距时需重新标定。
- `calibrate_hand_eye.py` 已实现 `record-poses`、`capture`、`solve`，采集时默认
  显示实时相机画面。关闭窗口或按 Space/Q/Esc 会请求急停；逐姿态模式在预览窗口
  按 Enter 后才运动。`--recapture pose_03 ...` 只移动和替换指定样本，失败样本保留
  原值；样本会记录棋盘尺寸并在采集时统一对称棋盘的 180 度角点方向。
- 当前本机 `vision/outputs/hand_eye/poses.json` 和 `samples.json` 都有 15 个姿态/样本。
  Park 求得的 Eye-in-Hand `gripper_T_camera` 平移为
  `[44.57, 57.69, -116.40] mm`，XYZ 欧拉角为 `[32.05, 4.27, -21.05] deg`。
  完整矩阵和指标见 `vision/results/hand_eye_result_2026-10-17.json`。
- **手眼外参仍是暂定结果，尚未验收**。固定棋盘原点一致性 RMSE 为
  `[9.84, 19.58, 74.09] mm`，Z 方向偏高；不要直接用于高精度控制。后续应改善
  非平行旋转轴的姿态覆盖、核对关节角到 URDF 的物理映射，并验证相机安装刚性。
- `visual_closed_loop.py` 使用在线图像雅可比闭环，当前不依赖上述手眼矩阵。
  其默认串口是 `COM7`；连接本机机械臂时显式传 `--port COM5`。
- `voice/` 只有接入说明，尚无语音识别、命令解析或调度实现。

`vision/outputs/`、缓存和本机采集图像被 Git 忽略；新克隆不会包含原始内参、姿态
或同步样本。`vision/results/` 只保存经过筛选的结果及质量说明。参考舵机标定 JSON
已提交。

## 5. 接手时可运行的检查

从仓库根目录，在 Miniconda Prompt 中：

```cmd
conda activate lerobot
lerobot-info
python -m unittest discover -v -s examples\eai_course\week4\task2 -p "test_*.py"
cd examples\eai_course\vision
python -m unittest discover -v -s . -p "test_*.py"
```

这些测试不连接硬件。视觉目录最近一次的 21 项测试通过。具体使用命令、
确认短语和参数见 `week4/task2/README.md`、`vision/README.md`。
手眼标定入口示例（在 `vision/` 目录）：

```cmd
python calibrate_hand_eye.py record-poses --port COM6 --camera-index 1 --count 15
python calibrate_hand_eye.py capture --port COM6 --camera-index 1 --square-size-mm 19.6
python calibrate_hand_eye.py solve --method park --square-size-mm 19.6
```

`record-poses` 会关闭舵机扭矩，需支撑机械臂；`capture` 会在精确确认后让机械臂
逐姿态运动。控制程序运行时 COM 端口只能由一个进程占用。下一步增加视觉或语音
功能时，优先建立唯一持有机器人串口的控制运行时，把识别结果转换成有限、可验证
的动作命令；急停应能中断普通动作。
