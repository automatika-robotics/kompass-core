# Kompass Core

[![中文版本][cn-badge]][cn-url]
[![ドキュメント-日本語][jp-badge]][jp-url]
[![PyPI][pypi-badge]][pypi-url]
[![MIT licensed][mit-badge]][mit-url]
[![Python Version][python-badge]][python-url]

[cn-badge]: https://img.shields.io/badge/文档-中文-blue.svg
[cn-url]: docs/README.zh.md
[jp-badge]: https://img.shields.io/badge/ドキュメント-日本語-red.svg
[jp-url]: docs/README.ja.md
[pypi-badge]: https://img.shields.io/pypi/v/kompass-core.svg
[pypi-url]: https://pypi.org/project/kompass-core/
[mit-badge]: https://img.shields.io/pypi/l/kompass-core.svg
[mit-url]: https://github.com/automatika-robotics/kompass-core/LICENSE
[python-badge]: https://img.shields.io/pypi/pyversions/kompass-core.svg
[python-url]: https://www.python.org/downloads/

Kompass Core is a high-performance, GPU-accelerated library for motion planning, mapping, and control in robot navigation systems. The core algorithms are implemented in C++ with seamless Python bindings. It also implements third party integrations with [OMPL](https://ompl.kavrakilab.org/) and [FCL](https://github.com/flexible-collision-library/fcl). The Kompass philosophy is to be blazzingly fast and highly reliable, by implementing GPGPU supported parallelized algorithms which are agnostic to underlying hardware. Thus Kompass Core can be run on CPUs or GPUs from a wide variety of vendors, making it easy for robot hardware manufacturers to switch underlying compute architecture without overhauling their software stack.

This package is developed to be used with [Kompass](https://github.com/automatika-robotics/kompass) for creating navigation stacks in [ROS2](https://docs.ros.org/en/rolling/index.html). For detailed usage documentation, check Kompass [docs](https://automatika-robotics.github.io/kompass/).


- [**Install**](#installation) Kompass Core 🛠️
- Check the [**Package Overview**](#-package-overview)
- To use Kompass Core on your robot with ROS2, check the [**Kompass**](https://automatika-robotics.github.io/kompass) framework 🚀


# Installation

## Install with GPU Support (Recommended)

- To install kompass-core with GPU support, on any Ubuntu 20+ (including Jetpack) based machine, you can simply run the following:

```bash
curl https://raw.githubusercontent.com/automatika-robotics/kompass-core/refs/heads/main/build_dependencies/install_gpu.sh | bash
```

This script will install all relevant dependencies, including [AdaptiveCPP](https://github.com/AdaptiveCpp/AdaptiveCpp) and install the latest version of kompass-core from source. It is good practice to read the [script](https://github.com/automatika-robotics/kompass-core/blob/main/build_dependencies/install_gpu.sh) first.

## Installing with pip (CPU only)

- On Ubuntu versions >= 22.04, install dependencies by running the following:

```bash
sudo apt-get install libompl-dev libfcl-dev
```

- Then install kompass-core as follows:

```bash
pip install kompass-core
```

Wheels are available on Pypi for linux x86_64 and aarch64 architectures. Please note that the version available on Pypi does not support GPU acceleration yet.

## Installation Contents

The following three packages will become available once kompass-core is installed.

- `kompass_cpp`: Core C++ library for control, collision checking, and mapping algorithms.
- `kompass_core`: Python bindings for Kompass core C++ library with front-end classes for configuration and high-level logic.
- `omplpy`: Bespoke python bindings for the Open Motion Planning Library (OMPL).


# 📦 Package Overview

The package includes modules for mapping, control, trajectory planning, and vision-based tracking algorithms, with **GPU acceleration** support and Python bindings via `nanobind`.


### Control Module
- Includes a rich set of optimized C++ control strategies implementations and their python wrappers.
- Supports **GPU-accelerated** trajectory sampling and cost evaluation with customizable weights for sampling based controllers.
- Internally implements feature-based bounding box tracking and depth detection for enhanced vision-based tracking control.

| Algorithm                                   | Description                                        |
| ------------------------------------------- | -------------------------------------------------- |
| **Stanley**                   | Path tracking with robust convergence              |
| **DWA (Dynamic Window Approach)** | Velocity-space sampling and optimization           |
| **DVZ**                           | Reactive obstacle avoidance using deformable zones |
| **VisionRGBFollower**   | Follow visual targets using RGB images          |
| **VisionRGBDFollower**   | Follow visual targets using RGBD (depth) images          |

### Mapping Module
- Implements efficient local mapping and occupancy grid generation algorithms, with configuration support for various laser models and grid resolution settings.
- Supports **GPU-accelerated** mapping for real-time performance.


### Utilities Module
- Provides collision checking utilities and critical zone detection to ensure safe navigation, including both CPU and GPU implementations.
- Logger utilities for runtime diagnostics.
- Linear state-space Kalman filter implementation for state estimation (C++).
- Spline interpolation utilities for path control.

### Data Types and Models Modules
- Rich set of data types to represent paths, trajectories, controls, velocities, bounding boxes and various sensor data.
- Strongly-typed parameters and configuration classes to enable flexible tuning.
- Robot models and motion kinematics, supporting differential, omni-directional, and Ackermann robots. Along with geometry definitions, control limits and simulation-ready state representations.

### Third Party Modules
Includes wrappers and integrations with external planning and collision libraries:

- FCL (Flexible Collision Library)

- OMPL (Open Motion Planning Library)


## Copyright

The code in this distribution is Copyright (c) 2024 Automatika Robotics unless explicitly indicated otherwise.

Kompass Core is made available under the MIT license. Details can be found in the [LICENSE](LICENSE) file.

## Contributions

Kompass Core has been developed in collaboration between [Automatika Robotics](https://automatikarobotics.com/) and [Inria](https://inria.fr/). Contributions from the community are most welcome.
