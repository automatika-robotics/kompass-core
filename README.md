# Kompass Core

[![PyPI][pypi-badge]][pypi-url]
[![MIT licensed][mit-badge]][mit-url]
[![Python Version][python-badge]][python-url]

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
- [**Copyright**](#copyright) and [**Contributions**](#contributions)
- To use Kompass Core on your robot with ROS2, check the [**Kompass**](https://automatika-robotics.github.io/kompass) framework 🚀


# Installation

## Install with GPU Support (Recommended)

To install kompass-core with GPU support, on any Ubuntu 20+ (including Jetpack) based machine, you can simply run the following:

- `curl https://raw.githubusercontent.com/automatika-robotics/kompass-core/refs/heads/main/build_dependencies/install_gpu.sh | bash`

This script will install all relevant dependencies, including [AdaptiveCPP](https://github.com/AdaptiveCpp/AdaptiveCpp) and install the latest version of kompass-core from source. It is good practice to read the [script](https://github.com/automatika-robotics/kompass-core/blob/main/build_dependencies/install_gpu.sh) first.

## Installing with pip (CPU only)

On Ubuntu versions >= 22.04, install dependencies by running the following:

- `sudo apt-get install libompl-dev libfcl-dev libpcl-dev`

Then install kompass-core as follows:

- `pip install kompass-core`

Wheels are available on Pypi for linux x86_64 and aarch64 architectures. Please note that the version available on Pypi does not support GPU acceleration yet.

## Installation Contents

The following three packages will become available once kompass-core is installed.

- `kompass_core`: The main Python API containing all the wrappers and utilities for motion planning and control for navigation in 2D spaces.
- `kompass_cpp`: Python bindings for Kompass core C++ library containing the algorithms implementation for path tracking and motion control.
- `omplpy`: Bespoke python bindings for the Open Motion Planning Library (OMPL).


# 📦 Package Overview

This repository consists the following modules:

- `kompass_cpp/` — Core C++ implementation of planning, control, collision checking, and mapping algorithms.

- `kompass_core/` — Python implementations and front-end classes for configuration and high-level logic.

## `kompass_cpp` Overview

`kompass_cpp/` contains the C++ package which includes mapping, control, trajectory planning, and vision-based tracking algorithms, with **GPU acceleration** support and Python bindings via `nanobind`.

### 1. Mapping
- Implements efficient local mapping algorithms.
- Supports **GPU-accelerated** mapping for real-time performance.
- Core classes: `LocalMapper`, `LocalMapperGPU`

### 2. Control & Trajectory Planning
- Includes multiple control strategies such as PID, Stanley, Dynamic Window Approach (DWA), and vision-guided controllers.
- Supports **GPU-accelerated** trajectory sampling and cost evaluation with customizable weights.
- Core classes: `Controller`, `PID`, `Stanley`, `DWA`, `VisionDWA`, `TrajectorySampler`, `CostEvaluator`

### 3. Collision and Critical Zone Checking
- Provides collision checking utilities and critical zone detection to ensure safe navigation.
- Includes both CPU and GPU implementations.
- Core classes: `CollisionChecker`, `CriticalZoneChecker`, `CriticalZoneCheckerGPU`

### 4. Vision and Tracking
- Feature-based bounding box tracking and depth detection for enhanced perception.
- Supports robust vision-based navigation algorithms.
- Core classes: `FeatureBasedBboxTracker`, `DepthDetector`

### 5. Utilities
- Thread pooling for efficient multi-threaded operations.
- Logger utilities for runtime diagnostics.
- Linear state-space Kalman filter implementation for state estimation.
- Spline interpolation utilities in the `tk` namespace.

### 6. Data Types and Parameters
- Rich set of data types to represent paths, trajectories, controls, velocities, and bounding boxes.
- Strongly-typed parameters and configuration classes to enable flexible tuning.

### 7. Python Bindings
- Comprehensive Python bindings built with `nanobind` to enable seamless integration with Python workflows.
- Bindings cover core functionalities across mapping, control, vision, and utilities.



## `kompass_core` Overview

- `kompass_core.calibration` - Modules for robot motion model calibration, fitting and robot simulation.

- `kompass_core.control` - A rich set of control strategies and configurations. Include the wrapper python classes for the C++ implementations:

| Algorithm                                   | Description                                        |
| ------------------------------------------- | -------------------------------------------------- |
| **Stanley**                   | Path tracking with robust convergence              |
| **DWA (Dynamic Window Approach)** | Velocity-space sampling and optimization           |
| **DVZ**                           | Reactive obstacle avoidance using deformable zones |
| **VisionRGBFollower**   | Follow visual targets using RGB images          |
| **VisionRGBDFollower**   | Follow visual targets using RGBD (depth) images          |

- `kompass_core.datatypes` - Standardized message/data formats for various robot and sensor data.


- `kompass_core.mapping` - Local mapping and occupancy grid generation, with configuration support for various laser models and grid resolution settings.

- `kompass_core.models` - Robot models and motion kinematics, supporting differential, omni-directional, and Ackermann robots. Along with geometry definitions, control limits and simulation-ready state representations.

- `kompass_core.motion_cost` - Cost models for trajectory evaluation in Python with various costs including collision probabilities, reference tracking and dynamic/static obstacle handling.

- `kompass_core.performance` - Modules for evaluating algorithms performance.

- `kompass_core.py_path_tools` - Path interpolation and execution tools.

- `kompass_core.simulation` - Tools for simulating robot motion and evaluating path feasibility.

- `kompass_core.third_party` - Wrappers and integrations with external planning and collision libraries:

    - FCL (Flexible Collision Library)

    - OMPL (Open Motion Planning Library)

- `kompass_core.utils` - General utilities.


## Copyright

The code in this distribution is Copyright (c) 2024 Automatika Robotics unless explicitly indicated otherwise.

Kompass Core is made available under the MIT license. Details can be found in the [LICENSE](LICENSE) file.

## Contributions

Kompass Core has been developed in collaboration between [Automatika Robotics](https://automatikarobotics.com/) and [Inria](https://inria.fr/). Contributions from the community are most welcome.
