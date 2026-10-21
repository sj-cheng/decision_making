# MCTS 项目迁移：当前机器人平台现场分析

检测时间：2026-10-13  
检测主机：`robot-nuc-35`  
结论依据：本机命令、设备节点、源码、配置文件、Docker 镜像。未在运行的 ROS 图上观测到的话题，均来自源码与配置，不代表检测当时已有发布者。

检测时 ROS master（`http://localhost:11311`）未启动，底盘节点与定位栈均未运行。

---

## 平台身份（必须先固定）

| 来源 | 值 |
|---|---|
| hostname | `robot-nuc-35` |
| 无线网 `wlp0s20f3` | `192.168.30.35/24` |
| `~/.bashrc` | `MACHINE_ID=35`，`ROBOT_TEAM=RED`，`ROBOT_ID=5` |
| `/etc/robot.conf`（Ansible 生成） | `MACHINE_ID=25`，`ROBOT_TEAM=RED`，`ROBOT_ID=1` |
| 检测时进程环境 | `MACHINE_ID=25`，`ROBOT_ID=1` |
| 定位 launch | `/nfs/SYL/RSG_Localization/src/rsg_localization/coordinate_change/launch/red5_locate.launch`，`/listen/drone_id=red5` |

`/nfs/env_setup.sh` 会 source `/etc/robot.conf`，交互式 bash 又会被 `~/.bashrc` 覆盖成 RED 5。两套身份同时存在。

与 hostname、无线 IP、`red5_locate.launch` 对齐的是 **RED 5 / machine 35**。迁移脚本必须显式 export，不能依赖默认环境。

---

## 1. 计算资源

| 项目 | 实测 |
|---|---|
| 整机 | Intel NUC11PHi7，主板 `NUC11PHBi7`，BIOS `PHTGL579.0079.2024.1014.1949` |
| OS | Ubuntu 20.04.6 LTS，内核 `5.15.0-139-generic`，x86_64 |
| CPU | 11th Gen Intel Core i7-1165G7 @ 2.80 GHz，1 socket，4 核 8 线程，最大 4.7 GHz，AVX-512。当前 governor：`powersave` |
| 内存 | 15 GiB；检测时 available 约 12 GiB；swap 2 GiB，未使用 |
| 系统盘 | ZHITAI Ti600 1 TB NVMe。根分区 915 G，已用 374 G，剩余 495 G |
| NFS | `192.168.30.254:/nfs` 挂载于 `/nfs`（796 G，剩余 602 G） |
| 独显 | NVIDIA GeForce RTX 2060，6 GiB，compute capability **7.5**，驱动 `535.183.01`，驱动报告 CUDA **12.2**。检测时显存 187 MiB，仅桌面占用 |
| 核显 | PCI `8086:9a49`（Tiger Lake GT2 / Iris Xe），`/dev/dri/card0` 与 `renderD128` |
| NPU | PCI `00:08.0` = `8086:9a11`（Tiger Lake GNA）。无对应内核模块，无 `/dev/accel`。没有可用 NPU 运行时 |
| 主机 CUDA 工具链 | 无 `nvcc`。`ldconfig` 中无 `libcudart`、`libnvinfer`。主机 Python 无 PyTorch、TensorRT |
| Docker GPU | Docker 28.1.1，runtime 含 `nvidia`。镜像 `192.168.30.254/library/nv_ros:yolox` 内 `torch 2.1.0a0+fe05266`，`--gpus all` 下 `torch.cuda.is_available()==True`，设备名 RTX 2060。同镜像 OpenCV 4.8.1 |

### 对 MCTS 的含义

- 经典 MCTS 是 CPU 搜索，i7-1165G7 可以运行。
- 神经网络价值/策略模型可以放在 RTX 2060 上。6 GB 适合小模型。
- 主机没有 TensorRT。推理使用 `nv_ros:yolox`，或另行安装 CUDA toolkit。
- 底盘命令超过 1 s 未更新会被置零。搜索周期必须短于该超时，并持续重发。

---

## 2. 底盘与动作接口

底盘类型：**四电机全向速度底盘**。控制量是车体坐标系 `vx`、`vy`、`vw`。

轮径、轮距、减速比在协议结构 `cmd_chassis_param` 中存在，源码标记为未实现。这些几何参数 **不确定**。

电机不由 NUC 直接 PWM。链路：

```text
geometry_msgs/Twist
  -> rsg_control_node
  -> RoboMaster 串口协议
  -> /dev/serial_sdk
  -> STM32
  -> 四个电机
```

### 串口

| 项目 | 值 |
|---|---|
| 设备 | `/dev/serial_sdk` → `ttyACM0` |
| USB | `0483:5740`，STMicroelectronics STM32 Virtual COM Port |
| 序列号 | `385A31653439` |
| 波特率 | `921600`（`roborts_sdk` 写死） |
| SocketCAN | 无。NUC 上没有 `can0`。CAN 在 STM32 一侧 |

### 真实控制入口

使用镜像 `192.168.30.254/library/rsg_control:250818-1552` 中的 `rsg_control_node`。

对应源码：

- `/nfs/CGY/test_rsg_control/src/rsg_control_src`
- `/nfs/CGY/arena_pilot/base_ws/src/arena_pilot_base`

`MotionCtrl` 的 NodeHandle 固定为 `"control"`。

**MCTS 输出动作时发布：**

```text
话题: /control/cmd_vel
类型: geometry_msgs/Twist
linear.x   车体前向速度，m/s
linear.y   车体左向速度，m/s
angular.z  航向角速度，rad/s
```

节点内部将 `angular.z * 180/π` 转为 deg/s，再按 `hardware_config.h` 限幅：

| 量 | 软件限幅 |
|---|---|
| `vx`、`vy` | ±1.5 m/s |
| `vw` | ±300 deg/s（约 ±5.24 rad/s） |

控制环 50 Hz（`MOTION_CTRL_TIMER_PERIOD = 0.02`）。`/control/cmd_vel` 超过 **1.0 s** 没有新消息时，`vx/vy/vw` 被置 0。必须周期重发。

最终下发是多路相加：

```text
/control/cmd_vel
+ /control/cmd_vel_aux
+ 遥控器
+ 云台协调跟随
+ 摇摆
```

`/control/cmd_vel_acc`（`rsg_msgs/TwistAccel`）的回调只取 `twist`。加速度字段没有传给底盘。

### 不要使用的旧入口

`/nfs/HJT/roborts_ws` 的 `roborts_base_node` 订阅全局 `/cmd_vel`，且 `CMakeLists.txt` 编译的是 `chassis_for_cars.cpp`。那不是这台车当前的执行入口。

已对准新入口的本地代码：

- `/home/robomaster/SJC/src/robo_square`：订阅 `/odom`，发布 `/control/cmd_vel`
- `/home/robomaster/CGY/arena_small_20261009`：发布 `/control/cmd_vel`

### 反馈话题

| 话题 | 类型 | 内容 |
|---|---|---|
| `/odom` | `nav_msgs/Odometry` | `odom` → `base_link`。米、m/s、rad/s。来自 STM32 的 `position_*_mm`、`v_*_mm` 和陀螺。不是地图坐标 |
| `/control/chassis_info` | `rsg_msgs/ChassisInfo` | `motor1..4_online`、目标速度、yaw、电池 |
| `/control/motion_ctrl_info` | `rsg_msgs/MotionCtrlInfo` | 合成后的实际速度指令 |
| `/control/virt_odom` | `nav_msgs/Odometry` | 虚拟底盘位姿 |

### 启动

容器 `rsg_control:250818-1552` 或 `:latest`，`--net host --privileged -v /dev:/dev`，然后：

```bash
roslaunch rsg_control base.launch
```

配置：`serial_port: /dev/serial_sdk`，模块 `chassis, gimbal, referee_system`。root 下节点会设置 `SCHED_FIFO` 并绑定 CPU0。

检测时 `rsg_control_node` 未运行。节点未启动时，发布 `/control/cmd_vel` 不会驱动车辆。

---

## 3. 总线

| 总线 | 检测结果 |
|---|---|
| USB | STM32 CDC（底盘）；海康工业相机 |
| Ethernet `enp88s0` | `192.168.1.50/24`。对端雷达 `192.168.1.160`，ping 0.27–0.64 ms |
| Wi-Fi `wlp0s20f3` | `192.168.30.35/24`。网关 `192.168.30.1`。NFS `192.168.30.254` |
| ZeroTier | `zt3jn4ybpt` = `192.168.196.119/24` |
| UART | 仅上述 ACM。`/dev/ttyS*` 是板载串口节点，没有接到机器人的证据 |
| CAN | 无接口 |
| SPI | 无 `spidev` |
| I2C / GPIO | 有 `/dev/i2c-*`、`/dev/gpiochip0`，属于 NUC/核显，不是车体总线 |
| ROS | ROS1 Noetic。`/opt/ros` 下没有 ROS2。检测时 master 不可达，无节点 |

---

## 4. 传感器与 MCTS 状态

### 4.1 激光：Livox MID-360

配置与在线地址一致。

| 项 | 值 |
|---|---|
| `lidar_type` | `8`（MID-360） |
| 雷达 IP | `192.168.1.160`（ping 通） |
| 主机 IP | `192.168.1.50` |
| `pcl_data_type` | `1` = `kLivoxLidarCartesianCoordinateHighData` |
| `pattern_mode` | `0` = `kLivoxLidarScanPatternNoneRepetive` |

配置文件：

- `/home/robomaster/Documents/MID360_config.json`
- `/nfs/SYL/RSG_Localization/src/livox_ros_driver2/config/MID360_config.json`

驱动 launch：`msg_MID360.launch`。`xfer_format=1`（`livox_ros_driver2/CustomMsg`），`publish_freq=10.0`。

| 话题 | 类型 | 说明 |
|---|---|---|
| `/livox/lidar` | `livox_ros_driver2/CustomMsg` | 10 Hz |
| `/livox/imu` | `sensor_msgs/Imu` | 雷达内置 IMU |
| `/livox/points` | `livox_repub_online` 从 `/livox/lidar` 转出 | |
| `/Odometry` | `nav_msgs/Odometry` | Faster-LIO，`laser_mapping.cc` 发布 |
| `/cloud_registered` | `sensor_msgs/PointCloud2` | 世界系点云 |
| `/cloud_registered_body` | `sensor_msgs/PointCloud2` | 车体系点云 |
| `/path` | `nav_msgs/Path` | Faster-LIO 轨迹 |
| `/robot_pose` | `nav_msgs/Odometry` | 地图/世界系位姿和速度 |

`red5_locate.launch` 串联 MID-360、Faster-LIO、`multi_vehicle_localization_yaml`。

Faster-LIO 加载的是 `avia.yaml`，文件内数值如下（不是本次标定测量值）：

- `scan_line: 6`
- `fov_degree: 90`
- `det_range: 450`
- `blind: 4`
- `extrinsic_T: [0.04165, 0.02326, -0.0284]`

MID-360 使用 Avia 参数。迁移时要单独核对。

旧 bringup 中的北洋雷达 `192.168.0.10` 和 `/dev/rplidar` 当前都不存在。

### 4.2 相机：海康 MV-CS016-10UC

| 项 | 值 |
|---|---|
| 接口 | USB3，MVS SDK（`/opt/MVS`） |
| 序列号 | `DA5624287` |
| 固件 | `V2.2.0 240702` |
| V4L2 | 无 `/dev/video*` |

打开设备、未取流时读到：

| 参数 | 值 |
|---|---|
| `WidthMax` × `HeightMax` | 1440 × 1080 |
| 当前 ROI | 1280 × 960，`OffsetX=80`，`OffsetY=60` |
| `PixelFormat` | `0x01080009` = BayerRG8 |
| `PayloadSize` | 1228800 = 1280×960 |
| `ExposureTime` | 3000 µs |
| `AcquisitionFrameRateEnable` | False |
| `AcquisitionFrameRate` | 165.9 |
| `ResultingFrameRate` | 185.0 |

`AcquisitionFrameRate` 与 `ResultingFrameRate` 是打开瞬间的 GenICam 读数，不是实测推流 FPS。持续取流帧率 **不确定**。

`base.launch` 静态 TF：`gimbal` → `camera`，平移 `x=0.1 m`，RPY `(-90°, 0°, -90°)`。

### 4.3 定位与其它

| 传感器 | 状态 |
|---|---|
| 底盘 IMU | STM32 推送 yaw / yaw rate。无独立 IMU USB |
| 雷达 IMU | `/livox/imu` |
| 编码器 | 只有底盘积分位姿。无原始编码器话题 |
| GPS / RTK | 未检测到设备，定位包中无对应 launch |
| 超声波 | 未检测到 |
| 深度相机 | 主机有 `.realsense-config.json` 和 realsense udev 规则。USB 树上没有 RealSense |

### 4.4 MCTS 状态与动作

定位栈当前未运行。栈启动后：

```text
位置、速度   /robot_pose                 地图系
             /odom                       底盘积分，不是场地坐标
环境         /livox/lidar 或 /cloud_registered
目标         传感器不提供，由决策/上层给定
动作         /control/cmd_vel            连续 (vx, vy, wz)
```

---

## 5. 软件环境

### 主机已安装

- ROS Noetic desktop-full，Gazebo 11
- Python 3.8.10，pip 24.1.2
- gcc/g++ 9.4.0，CMake 3.16.3
- OpenCV 4.2（`python3-opencv` 同版本），PCL 1.10，Eigen 3.3.7
- `ros-noetic-librealsense2`、`realsense2-camera`、`cv-bridge`、`pcl-ros`、`diff-drive-controller`
- Livox SDK2：`/home/robomaster/Livox-SDK2`
- 海康 MVS：`/opt/MVS`（检测时 `MvLogServer` 在运行）
- Docker + nvidia container runtime
- Ceres 2.0.0：`/home/robomaster/ceres-solver-2.0.0`

### 主机未安装

PyTorch、TensorRT、OpenVINO、`python-can`、`pyserial`、ROS2。

### 工作空间与镜像

| 用途 | 路径 / 镜像 |
|---|---|
| 底盘 | `/nfs/CGY/test_rsg_control`、`/nfs/CGY/arena_pilot/base_ws`；镜像 `rsg_control` |
| 定位 | `/nfs/SYL/RSG_Localization`；镜像 `rsg_localization` |
| 规划 | `/nfs/HJT/planning_ws`（ego-planner 车端 remap 到 `/control/cmd_vel_acc`）；镜像 `rsg_planning` |
| 本机运动示例 | `/home/robomaster/SJC` 的 `robo_square` |
| 决策 | 镜像 `rsg_bt`、`nv_ros:yolox`、`decision4:zyc20250925` |
| 近期本地镜像 | `robo_square:latest`（无 torch）、`arena-e2e-slam:noetic`（无 torch） |

检测时没有 ROS 节点。已有 Docker 容器均为退出状态。

---

## 6. Migration Compatibility Analysis

### 6.1 硬件能否运行 MCTS

能。CPU 适合树搜索。RTX 2060 6 GB 能跑小网络，已在 `nv_ros:yolox` 中验证 CUDA。不适合大模型或 TensorRT 部署。实时性取决于搜索能否在 1 s 命令超时内持续发出 `/control/cmd_vel`。建议 10–50 Hz 重发。

### 6.2 可以直接迁移的模块

MCTS 搜索循环、reward、状态评估。前提是它们不依赖仿真器内部 API。输入输出改为下面的 ROS 类型。

### 6.3 必须修改的模块

- **状态输入**：接 `/robot_pose`、`/odom`、`/livox/lidar` 或 `/cloud_registered`。定位使用 `red5_locate.launch`，并先解决 RED 5 / RED 1 身份冲突。
- **动作空间**：连续 `(vx, vy, wz)`，限幅 ±1.5 m/s、±300 deg/s。前进、后退、平移、转向都编码进这一条 Twist。
- **运动模型**：全向速度模型，`vy` 有效。
- **控制执行**：只发 `/control/cmd_vel`。不发全局 `/cmd_vel`。加速度话题当前不会作用到底盘。
- **神经网络**：放进带 GPU 的容器，不装在主机 Python 3.8 上。
- **Faster-LIO**：`avia.yaml` 与 MID-360 不一致，外参和 `scan_line` / `fov` 要重核。

### 6.4 需要重新接上的接口

1. 启动 `rsg_control_node`（串口 `/dev/serial_sdk`）。
2. 启动 MID-360 + Faster-LIO + `red5` 坐标变换。
3. MCTS 节点与它们共用同一个 ROS master。
4. 命令超时 1 s。mux 会叠加遥控器、`cmd_vel_aux`、云台跟随。

### 6.5 缺少的硬件与软件能力

未检测到：GPS/RTK、超声波、深度相机、可用 NPU、主机 TensorRT、原始编码器话题、轮式几何参数。

地图坐标在定位栈启动前不存在。相机当前 ROI 为 1280×960 BayerRG8，推流帧率未测。

### 6.6 迁移难度

中等。车端执行和定位栈已经存在。缺口是把 MCTS 的状态和动作接到这两个栈上，并处理身份冲突和 Avia 配置。

### 6.7 推荐迁移步骤

1. 固定身份。按 hostname 和 `red5_locate.launch`，在启动脚本中显式设置 `ROBOT_TEAM=RED`、`ROBOT_ID=5`、`MACHINE_ID=35`。确认 `/etc/robot.conf` 不会在后面把它改回 RED 1。
2. 启动 `rsg_control`。确认 `/dev/serial_sdk` 打开，`/control/chassis_info` 中四个电机在线，`/odom` 有数据。
3. 启动 `red5_locate.launch`。确认 `/livox/lidar`、`/livox/imu`、`/Odometry`、`/robot_pose`。核对 MID-360 外参。
4. 用 10 Hz 发布器向 `/control/cmd_vel` 发小速度，确认 1 s 超时和 ±1.5 m/s 限幅，再接入 MCTS。
5. 将 MCTS 环境模型改为全向 `(vx, vy, wz)`。状态从 `/robot_pose` 和点云读取。神经网络放进 `nv_ros:yolox`，使用 `--gpus all`。
6. 若与现有规划共存，避开 `/control/cmd_vel_aux` 和 ego-planner 的 `/control/cmd_vel_acc`，否则速度会被相加。

---

## 给服务器端 AI 的落地接口摘要

```text
动作发布
  话题 /control/cmd_vel
  类型 geometry_msgs/Twist
  linear.x  m/s   限幅 ±1.5
  linear.y  m/s   限幅 ±1.5
  angular.z rad/s 限幅 ±300 deg/s
  周期      建议 10–50 Hz，超过 1.0 s 未更新则底盘速度被置 0

地图系状态
  /robot_pose          nav_msgs/Odometry
  /cloud_registered    sensor_msgs/PointCloud2
  /livox/lidar         livox_ros_driver2/CustomMsg   10 Hz
  /livox/imu           sensor_msgs/Imu

底盘积分（非场地坐标）
  /odom                nav_msgs/Odometry
  /control/chassis_info

前置进程
  rsg_control_node     串口 /dev/serial_sdk  921600
  red5_locate.launch   身份 RED / 5 / machine 35

推理
  镜像 192.168.30.254/library/nv_ros:yolox
  torch 2.1.0a0  CUDA 可用（RTX 2060，6 GB，cc 7.5）
```
