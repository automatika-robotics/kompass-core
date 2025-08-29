# Kompass Core（核心库）

[![English Version][en-badge]][en-url]
[![ドキュメント-日本語][jp-badge]][jp-url]
[![PyPI][pypi-badge]][pypi-url]
[![MIT 许可][mit-badge]][mit-url]
[![Python 版本][python-badge]][python-url]

[en-badge]: https://img.shields.io/badge/Documentation-English-green.svg
[en-url]: ../README.md
[jp-badge]: https://img.shields.io/badge/ドキュメント-日本語-red.svg
[jp-url]: README.ja.md
[pypi-badge]: https://img.shields.io/pypi/v/kompass-core.svg
[pypi-url]: https://pypi.org/project/kompass-core/
[mit-badge]: https://img.shields.io/pypi/l/kompass-core.svg
[mit-url]: https://github.com/automatika-robotics/kompass-core/LICENSE
[python-badge]: https://img.shields.io/pypi/pyversions/kompass-core.svg
[python-url]: https://www.python.org/downloads/

Kompass Core 是一个高性能、支持 GPU 加速的库，用于机器人导航系统中的运动规划、建图与控制。核心算法采用 C++ 实现，并通过无缝 Python 绑定提供接口。它还集成了第三方库 [OMPL](https://ompl.kavrakilab.org/) 和 [FCL](https://github.com/flexible-collision-library/fcl)。Kompass 的设计哲学是极致高速与高度可靠，通过 GPGPU 支持的并行化算法实现与底层硬件无关。因此，Kompass Core 可运行于多种厂商的 CPU 或 GPU 上，便于机器人硬件厂商在不更换软件栈的情况下切换计算平台。

本软件包可与 [Kompass](https://github.com/automatika-robotics/kompass) 一起使用，用于在 [ROS2](https://docs.ros.org/en/rolling/index.html) 中构建导航栈。详细使用文档请见 [Kompass 文档](https://automatika-robotics.github.io/kompass/)。

- [**安装指南**](#安装指南) Kompass Core 🛠️
- 查看 [**软件包概览**](#-软件包概览)
- 若需在 ROS2 中使用 Kompass Core，请访问 [**Kompass 框架**](https://automatika-robotics.github.io/kompass) 🚀

# 安装指南

## 安装支持 GPU 的版本（推荐）

- 要在任何基于 Ubuntu 20+（包括 Jetpack）系统的机器上安装支持 GPU 的 kompass-core，只需运行以下命令：

```bash
curl https://raw.githubusercontent.com/automatika-robotics/kompass-core/refs/heads/main/build_dependencies/install_gpu.sh | bash
```

该脚本将安装所有相关的依赖项，包括 [AdaptiveCPP](https://github.com/AdaptiveCpp/AdaptiveCpp)，并从源代码安装最新版的 kompass-core。建议您先阅读该[脚本](https://github.com/automatika-robotics/kompass-core/blob/main/build_dependencies/install_gpu.sh)。

## 使用 pip 安装（仅支持 CPU）

- 在 Ubuntu 22.04 或更高版本上，请先运行以下命令安装依赖项：

```bash
sudo apt-get install libompl-dev libfcl-dev
```

- 然后按如下方式安装 kompass-core：
```bash
pip install kompass-core
```

适用于 linux x86_64 和 aarch64 架构的安装包已发布在 Pypi 上。请注意，Pypi 上提供的版本暂不支持 GPU 加速。

## 安装内容

安装 kompass-core 后，将提供以下三个软件包：

- `kompass_core`：主 Python API，包含用于 2D 空间导航的运动规划与控制的所有封装器与工具。
- `kompass_cpp`：Kompass 核心 C++ 库的 Python 绑定，包含路径跟踪与运动控制算法的实现。
- `omplpy`：为 OMPL（开源运动规划库）量身定制的 Python 绑定。

# 📦 软件包概览

本仓库包含以下模块：

- `kompass_cpp/` — 规划、控制、碰撞检测与建图算法的核心 C++ 实现。

- `kompass_core/` — Python 实现与前端配置类，支持高级逻辑开发。

## `kompass_cpp` 模块概览

`kompass_cpp/` 是一个包含建图、控制、轨迹规划与基于视觉的跟踪算法的 C++ 软件包，支持 **GPU 加速**，并通过 `nanobind` 提供 Python 绑定。

### 1. 建图
- 实现高效的局部建图算法。
- 支持 **GPU 加速**，实现实时性能。
- 核心类：`LocalMapper`, `LocalMapperGPU`

### 2. 控制与轨迹规划
- 支持多种控制策略，包括 PID、Stanley、动态窗口法（DWA）与视觉引导控制器。
- 支持 **GPU 加速** 的轨迹采样与代价评估，可自定义权重。
- 核心类：`Controller`, `PID`, `Stanley`, `DWA`, `VisionDWA`, `TrajectorySampler`, `CostEvaluator`

### 3. 碰撞检测与关键区域检测
- 提供碰撞检测工具与关键区域识别功能，保障导航安全。
- 提供 CPU 与 GPU 两种实现。
- 核心类：`CollisionChecker`, `CriticalZoneChecker`, `CriticalZoneCheckerGPU`

### 4. 视觉与跟踪
- 基于特征的边框跟踪与深度检测，增强感知能力。
- 支持鲁棒的视觉导航算法。
- 核心类：`FeatureBasedBboxTracker`, `DepthDetector`

### 5. 实用工具
- 多线程池用于高效的并发处理。
- 日志工具支持运行时诊断。
- 基于线性状态空间的卡尔曼滤波器用于状态估计。
- `tk` 命名空间下的样条插值工具。

### 6. 数据类型与参数
- 提供丰富的数据结构，用于表示路径、轨迹、控制、速度与边框等。
- 强类型参数与配置类，支持灵活调参。

### 7. Python 绑定
- 使用 `nanobind` 构建的全面 Python 绑定，实现与 Python 工作流的无缝集成。
- 绑定涵盖建图、控制、视觉与工具等核心功能。

## `kompass_core` 模块概览

- `kompass_core.calibration` - 机器人运动模型的校准、拟合与仿真模块。

- `kompass_core.control` - 多种控制策略及配置，包含 C++ 控制器的 Python 包装类：

| 算法名称                             | 描述                                              |
| ------------------------------------ | ------------------------------------------------- |
| **Stanley**                          | 具备鲁棒收敛性的路径跟踪                         |
| **DWA（动态窗口法）**              | 基于速度空间的采样与优化                         |
| **DVZ**                              | 使用可变形区域的反应式避障                       |
| **VisionRGBFollower**               | 基于 RGB 图像跟踪视觉目标                        |
| **VisionRGBDFollower**              | 基于 RGBD 图像（含深度）跟踪视觉目标             |

- `kompass_core.datatypes` - 标准化的机器人与传感器数据格式。

- `kompass_core.mapping` - 局部建图与占据栅格生成，支持多种激光模型与分辨率配置。

- `kompass_core.models` - 机器人模型与运动学，支持差动式、全向轮式与 Ackermann 结构；包含几何定义、控制限制与仿真状态表示。

- `kompass_core.motion_cost` - Python 中用于轨迹评估的代价模型，支持包括碰撞概率、参考轨迹跟踪、动态/静态障碍物等多种代价函数。

- `kompass_core.performance` - 用于算法性能评估的模块。

- `kompass_core.py_path_tools` - 路径插值与执行工具。

- `kompass_core.simulation` - 用于机器人运动仿真与路径可行性评估的工具。

- `kompass_core.third_party` - 与外部规划与碰撞库的封装与集成：

    - FCL（灵活碰撞库）

    - OMPL（开源运动规划库）

- `kompass_core.utils` - 通用工具函数集。

## 版权信息

本发行版中的源代码（除非另有说明）版权归 Automatika Robotics 所有 © 2024。

Kompass Core 遵循 MIT 开源许可证。详细信息请见 [LICENSE](LICENSE) 文件。

## 贡献说明

Kompass Core 由 [Automatika Robotics](https://automatikarobotics.com/) 与 [Inria](https://inria.fr/) 合作开发，欢迎社区开发者贡献代码与文档。
