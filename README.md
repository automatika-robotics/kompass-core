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

Kompass Core is a fast, GPU powered motion planning and control package for robot navigation. The package contains C++ implementation for core algorithms along with Python wrappers. It also implements third party integrations with [OMPL](https://ompl.kavrakilab.org/) and [FCL](https://github.com/flexible-collision-library/fcl). The Kompass philosophy is to be blazzingly fast and highly reliable, by implementing parallelized algorithms which are agnostic to underlying hardware. Thus Kompass Core can be run on CPUs, GPUs, NPUs or FPGAs from a wide variety of vendors, making it easy for robot manufacturers to switch underlying compute architecture.

This package is developed to be used with [Kompass](https://github.com/automatika-robotics/kompass) for creating navigation stacks in [ROS2](https://docs.ros.org/en/rolling/index.html). For detailed usage documentation, check Kompass [docs](https://automatika-robotics.github.io/kompass/).

## Installation

### Install with GPU Support (Recommended)

To install kompass-core with GPU support, on any Ubuntu (including Jetpack) based machine, you can simply run the following:

- `curl https://raw.githubusercontent.com/automatika-robotics/kompass-core/refs/heads/main/build_dependencies/install_gpu.sh | bash`

This script will install all relevant dependencies, including [AdaptiveCPP](https://github.com/AdaptiveCpp/AdaptiveCpp) and install the latest version of kompass-core from source. It is good practice to read the [script](https://github.com/automatika-robotics/kompass-core/blob/main/build_dependencies/install_gpu.sh) first.

### Installing with pip

On Ubuntu versions >= 22.04, install dependencies by running the following:

- `sudo apt-get install libompl-dev libfcl-dev libpcl-dev`

Then install kompass-core as follows:

- `pip install kompass-core`

Wheels are available on Pypi for linux x86_64 and aarch64 architectures. Please note that the version available on Pypi does not support GPU acceleration yet.

### Installation Contents

The following three packages will become available once kompass-core is installed.

- `kompass_core`: The main Python API containing all the wrappers and utilities for motion planning and control for navigation in 2D spaces.
- `kompass_cpp`: Python bindings for Kompass core C++ library containing the algorithms implementation for path tracking and motion control.
- `omplpy`: Bespoke python bindings for the Open Motion Planning Library (OMPL).

## Testing

### Run Planning Test

- `cd tests`
- `python3 test_ompl.py`

To test path planning using OMPL bindings a reference planning problem is provided using Turtlebot3 Waffle map and fixed start and end position. The test will simulate the planning problem for all geometric planners for the number of desired repetitions to get average values.

### Run Controllers Test

- `cd tests`
- `python3 test_controllers.py`

The test will simulate path tracking using a reference global path. The results plot for each available controller will be generated in tests/resources/control

## Usage Example

```python
from kompass_core.control import DVZ

from kompass_core.models import (
    AngularCtrlLimits,
    LinearCtrlLimits,
    Robot,
    RobotCtrlLimits,
    RobotGeometry,
    RobotType,
)

# Setup the robot
my_robot = Robot(
        robot_type=RobotType.ACKERMANN,
        geometry_type=RobotGeometry.Type.CYLINDER,
        geometry_params=np.array([0.1, 0.4]),
    )

# Set the robot control limits
robot_ctr_limits = RobotCtrlLimits(
    vx_limits=LinearCtrlLimits(max_vel=1.0, max_acc=5.0, max_decel=10.0),
    omega_limits=AngularCtrlLimits(
        max_vel=2.0, max_acc=3.0, max_decel=3.0, max_steer=np.pi
    ),
)

# Set the control time step (s)
control_time_step = 0.1     # seconds

# Initialize the controller
dvz = DVZ(
        robot=my_robot,
        ctrl_limits=robot_ctr_limits,
        control_time_step=control_time_step,
    )
```

## Copyright

The code in this distribution is Copyright (c) 2024 Automatika Robotics unless explicitly indicated otherwise.

Kompass Core is made available under the MIT license. Details can be found in the [LICENSE](LICENSE) file.

## Contributions

Kompass Core has been developed in collaboration between [Automatika Robotics](https://automatikarobotics.com/) and [Inria](https://inria.fr/). Contributions from the community are most welcome.
