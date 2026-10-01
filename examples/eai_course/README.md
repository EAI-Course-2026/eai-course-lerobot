# EAI Course LeRobot

本目录保存 EAI Course 2026 的 SO-ARM101（SCS215 舵机）控制代码。仓库本身是
LeRobot 的课程分支，因此 SCS215 通信适配、课程程序和后续视觉/语音模块可以在
同一个环境中安装和运行。

## 项目结构

```text
examples/eai_course/
|-- week4/
|   |-- task1/       # 预设姿态、键盘切换和动作序列
|   `-- task2/       # FK、IK、直线轨迹和末端键盘控制
|-- vision/          # 后续 OpenCV 感知模块
|-- voice/           # 后续语音命令模块
`-- setup_windows.cmd
```

SCS215 适配位于 LeRobot 源码中，主要改动为：

- 注册 `scs215` 型号、1024 分辨率、协议 1 和正确的控制表。
- 对协议 1 使用逐舵机读取，因为 SCS215 不支持 `GroupSyncRead`。
- 跳过 SCS215 手册中未定义的寄存器，并按实际安全范围进行标定。
- SO follower 的 1 至 6 号电机改用 SCS215 配置。

对应提交为 `62754eb8 feat(hardware): add Feetech SCS215 support`。

## Windows 环境

首次安装建议在 Miniconda Prompt 中逐行执行：

```cmd
git clone git@github.com:EAI-Course-2026/eai-course-lerobot.git
cd /d eai-course-lerobot
conda create -n lerobot python=3.12 -y
conda activate lerobot
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
examples\eai_course\setup_windows.cmd
```

已经有 `lerobot` 环境时，从仓库根目录执行：

```cmd
conda activate lerobot
examples\eai_course\setup_windows.cmd
```

安装脚本以 editable 模式安装当前仓库的 `feetech` 和 `hardware` 依赖，并运行
离线测试。以后执行 `git pull` 后不需要重新复制源码；只有依赖发生变化时才需要
再次运行安装脚本。

每台机械臂都必须单独编号和标定。标定文件位于用户缓存目录，不应提交到 Git：

```cmd
lerobot-find-port
lerobot-calibrate --robot.type=so_follower --robot.port=COM5 --robot.id=scs215_com5
```

## 运行控制程序

预设姿态和动作序列：

```cmd
cd /d examples\eai_course\week4\task1
python record_pose.py stand --port COM5 --robot-id scs215_com5
python control_presets.py --port COM5 --robot-id scs215_com5
```

检查映射、FK 和 IK：

```cmd
cd /d ..\task2
python check_step1.py --port COM5 --robot-id scs215_com5
python check_step2_fk.py --hardware --port COM5 --robot-id scs215_com5
python run_steps3_to5.py --hardware --delta-mm 0 0 10 --execute --port COM5 --robot-id scs215_com5
```

末端直线键盘控制：

```cmd
python keyboard_control.py --port COM5 --robot-id scs215_com5 --speed-mm-s 10 --control-hz 20
```

运动前应支撑机械臂、确认没有其他程序占用串口，并先使用只读或离线命令检查
当前标定。Task 2 的 `README.md` 记录了各步骤的安全确认和按键说明。

## 团队协作

每项功能使用独立分支，不直接在 `main` 上试验：

```cmd
git switch main
git pull --ff-only
git switch -c feature/vision-marker-tracking

git add examples/eai_course/vision
git commit -m "feat(vision): add marker tracking"
git push -u origin feature/vision-marker-tracking
```

语音功能可使用 `feature/voice-command-control`，修复可使用 `fix/...`。合并前运行：

```cmd
python -m unittest discover -v -s examples\eai_course\week4\task2 -p "test_*.py"
```

不要提交 Hugging Face 缓存中的标定 JSON、Conda 环境、摄像头录制数据或
`__pycache__`。
