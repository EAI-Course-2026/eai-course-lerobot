# Vision

本目录包含两组相互独立的功能：

- `calibrate_camera.py` 和 `camera_calibration.py`：以 OpenCV 官方实现为准的相机
  内参、畸变标定工具，不连接机械臂串口。
- `visual_closed_loop.py` 等：队友实现的黑球视觉闭环实验。

## 相机内参标定

实现依据：

- [OpenCV 官方 Python 标定示例](https://github.com/opencv/opencv/blob/4.x/samples/python/calibrate.py)
- [OpenCV 官方 Camera Calibration 教程](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

核心流程与官方代码一致：建立棋盘格三维点，使用 `findChessboardCorners` 和
`cornerSubPix` 获取图像点，再调用 `calibrateCamera` 求相机矩阵和畸变系数，最后
用 `projectPoints` 计算重投影误差。

仓库已有 `opencv-python-headless`。本工具使用 Tk/Pillow 显示画面，不需要卸载它，
也不要同时安装 `opencv-python`。

### 1. 准备棋盘格

打印 `chessboard_8x6_25mm.svg`，打印设置选择“实际大小”或 100%，不要选择“适合
页面”。打印后用尺测量方格边长应为 25 mm。图案有 **8 列、6 行内角点**；命令中
的 columns/rows 指内角点，不是黑白方格数量。

如果使用其他棋盘格，将下面命令的 `--columns`、`--rows` 和
`--square-size-mm` 改成实物参数。

### 2. 采集图片

在仓库根目录执行：

```cmd
conda activate lerobot
cd examples\eai_course\vision
python calibrate_camera.py capture --camera-index 1 --width 640 --height 480 --columns 8 --rows 6 --target-count 20
```

预览中检测到角点后按空格保存；按 `Q` 或 `Esc` 结束。采集 15 至 25 张清晰图片：
让棋盘分别覆盖画面中心、四角和边缘，并改变距离与倾角。不要只在同一位置连续
拍摄。图片保存到被 Git 忽略的 `outputs/camera_calibration/images`。

每次运行 `capture` 默认开始一个新批次，原图片会移入
`outputs/camera_calibration/archive/capture_时间戳/images`，避免换相机后混用图片。
仅在确认相机、分辨率和镜头设置完全相同并需要补拍时增加 `--append`。

若相机编号不是 1，先运行：

```cmd
lerobot-find-cameras opencv
```

### 3. 计算参数

```cmd
python calibrate_camera.py solve --columns 8 --rows 6 --square-size-mm 25
```

结果保存为 `outputs/camera_calibration/camera_intrinsics.json`，角点检查图保存在
`outputs/camera_calibration/debug`。应逐张查看角点是否正确。平均重投影误差越接近
0 越好；一般可先以小于 0.5 px 为目标，超过 1 px 时建议重新采集模糊或姿态单一
的图片。重新求解时，旧内参会先备份到
`outputs/camera_calibration/archive/camera_intrinsics_时间戳.json`。

### 4. 验证去畸变

```cmd
python calibrate_camera.py verify --camera-index 1 --width 640 --height 480
```

窗口左侧为原始画面，右侧为去畸变画面。用棋盘、门框等直线检查画面边缘。
标定结果与具体相机、分辨率、焦距/对焦状态绑定；改变分辨率或镜头设置后应重新
标定。

`CameraCalibration.load(path).undistort(frame)` 可供后续视觉代码直接调用。这里得到
的是相机内参和镜头畸变，并不等于相机坐标系到机械臂 base 坐标系的外参；固定
相机或手眼系统还需要单独完成外参/手眼标定。

## Eye-in-hand 手眼标定

相机安装在末端、棋盘固定不动的配置使用 `calibrate_hand_eye.py`。代码遵循 OpenCV
`calibrateHandEye` 的坐标定义，并复用 Week 4 的关节映射和 FK。采集期间不会对
画面做去畸变；`solvePnP` 直接接收原图角点、内参和畸变系数。

### 1. 记录安全姿态

关闭扭矩后手动摆出 15 个姿态。程序会同时打开相机预览；画面显示绿色
`DETECTED`、棋盘格完整可见时，在预览窗口按 `Space` 或 `Enter` 保存当前姿态：

```cmd
python calibrate_hand_eye.py record-poses --port COM6 --camera-index 1 --count 15
```

姿态必须位于关节范围内部，并保证之后按记录顺序插值运动不会碰撞。红色
`NOT DETECTED` 时程序不会保存姿态。除了平移，还要让相机绕至少两个不平行轴产生
明显旋转；不能只改变底座转角。从记录安全姿态开始就固定标定板，直到正式采集和
求解全部完成都不要移动机械臂基座、相机安装座或标定板。

默认采用断点续录：如果 `poses.json` 已有 8 个姿态，再运行相同命令会从 `8/15`
继续，下一个保存为 `pose_09`；`--count` 表示包含已有姿态在内的最终总数。每次保存
使用临时文件原子替换，避免写入中断损坏 JSON。需要彻底从 `pose_01` 重录时使用：

```cmd
python calibrate_hand_eye.py record-poses --port COM6 --camera-index 1 --count 15 --overwrite
```

首次成功保存新姿态前，旧文件会先复制到
`outputs/hand_eye/archive/poses_时间戳.json`；如果未保存就退出，旧文件不会改变。

### 2. 自动移动并同步采集

将棋盘固定，使每个已记录姿态都能完整看到棋盘，然后运行：

```cmd
python calibrate_hand_eye.py capture --port COM6 --camera-index 1 --square-size-mm 19.6
```

当前棋盘显示在笔记本屏幕上，实测连续 5 格为 98 mm，因此单格为 19.6 mm；该值
只适用于保持相同浏览器缩放和显示设置的这块屏幕。若改用打印板，应改回打印后
实测的方格尺寸。相机内参不因这里的尺寸变化而改变，但 PnP 平移尺度会随之变化。

采集时会打开实时相机预览，显示当前姿态的移动、到位和拍摄状态。默认逐姿态确认：
在**预览窗口**按 Enter 才开始移动到下一姿态。若已确认所有记录姿态之间的路径
安全，可加 `--automatic` 连续移动和采集；不要在首次验证路径时使用自动模式。
关闭预览窗口或在其中按 `Space`、`Q`、`Esc` 会请求急停。无桌面图形界面时可加
`--no-preview`，此时逐姿态确认仍在终端按 Enter。预览使用相机的最新帧，不消耗
正式采样读取的帧；保存的原图仍是机械臂稳定后选出的同步采集帧。

连续自动采集命令：

```cmd
python calibrate_hand_eye.py capture --port COM6 --camera-index 1 --square-size-mm 19.6 --automatic
```

程序只有在输入精确确认短语后才启用 ID 1 至 5 的扭矩。按空格会请求急停并关闭
扭矩。每次稳定后保存以下配对数据：

```text
base_T_gripper   FK 得到的末端到基座变换
camera_T_target  solvePnP 得到的棋盘到相机变换
```

若求解诊断确认只有少数姿态误差较大，可以只重采这些姿态并合并回现有数据集：

```cmd
python calibrate_hand_eye.py capture --port COM6 --camera-index 1 --square-size-mm 19.6 --recapture pose_03 pose_07 pose_11 --speed-deg-s 8 --settle-s 3
```

该模式要求现有 `samples.json` 已包含相同棋盘尺寸和这些姿态。每个成功姿态替换
同名旧样本，其他样本保持不变；失败或跳过的姿态继续保留旧样本。首次成功替换前
会备份原 JSON，新图片保存在新的时间戳目录。程序只会依次移动并采集
`pose_03 -> pose_07 -> pose_11`，不会经过其他记录姿态。由于三个姿态之间是直接
运动，首次运行应在预览窗口逐次按 Enter，并确认路径不会碰撞后再考虑追加
`--automatic`。本例的扭矩确认短语为 `START HAND EYE RECAPTURE 3`。

SCS215 在重力、齿隙和线缆拉力作用下通常不会精确停在指令角度，因此默认到位容差
为 10 度。程序拍照后会重新读取编码器，并用实测关节角而非指令角计算 FK，所以这项
容差不会把目标角冒充成真实位姿。若仍然超时，错误信息会列出五个关节的偏差；应
先检查偏差最大的舵机是否受阻或负载过大，不要继续盲目放宽容差。

原图和带坐标轴的检查图分别进入 `outputs/hand_eye/images/时间戳` 和
`outputs/hand_eye/axes/时间戳`，`samples.json` 始终指向最近一次采集。重新采集时
旧 JSON 会先进入 `outputs/hand_eye/archive`，而旧图片目录不会被覆盖。单个姿态
到位超时、看不全棋盘或 PnP 重投影误差超过 1.5 px 时会打印 `SKIPPED` 并继续下一
个姿态；串口、相机读取或文件保存等系统级故障仍会停止。全部完成后会汇总成功和
跳过项；有效样本不足 12 个时需要补录姿态后重新采集，不能用低质量数据凑数量。

### 3. 求手眼矩阵和检查误差

有效样本达到 12 个后，可以先不提供尺量坐标，只求手眼矩阵和内部一致性：

```cmd
python calibrate_hand_eye.py solve --method park --square-size-mm 19.6
```

对于旧版以 25 mm 计算、但实际屏幕方格为 19.6 mm 的 `samples.json`，上述命令会
缩放已保存的 PnP 平移，并用 FK 相对旋转自动统一对称棋盘可能发生的 180 度角点
顺序翻转；原始数据不会被改写。新版 `capture` 会在保存样本前完成角点顺序统一，
并把实际方格尺寸写入数据集元数据。

若还要完成 5 cm 的外部误差检查，先测量棋盘 **第一个内角点** 在机器人 base
坐标系中的实际 X/Y/Z，单位为毫米，再把三个实测数值写进命令。下面的
`300 -50 20` 只是格式示例，不是本机械臂的真实坐标：

```cmd
python calibrate_hand_eye.py solve --method park --square-size-mm 19.6 --measured-board-origin-mm 300 -50 20
```

OpenCV 返回 `gripper_T_camera`：相机坐标中的点乘该矩阵后进入末端坐标系；其中平移
列就是“相机光心在末端坐标系中的位置”。程序还会对每组数据计算：

```text
base_T_target = base_T_gripper @ gripper_T_camera @ camera_T_target
```

由于棋盘固定，所有 `base_T_target` 应一致。程序输出棋盘原点逐轴 RMSE、最大偏差，
以及估计原点与尺量原点的 X/Y/Z 误差；三个绝对误差均不超过 50 mm 才通过作业的
5 cm 指标。结果保存在 `hand_eye_result.json`，可截图的摘要为
`hand_eye_summary.png`，每张 `axes/*_axes.png` 用于确认棋盘坐标轴方向。
再次求解前，旧结果 JSON 和摘要图同样会自动复制到 `outputs/hand_eye/archive`。

棋盘是对称图案，采集时不要让相机相对棋盘翻转 180 度，否则角点顺序可能反向。
建议在作为原点的物理角旁做清晰标记，并始终确认检查图中的红色 X 轴、绿色 Y 轴
从同一个物理角点出发。

完整接入顺序和安全约束见 [项目交接文档](../HANDOFF.md#8-视觉功能的建议边界)。
