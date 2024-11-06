# Kompass Core

Kompass Core is a motion planning and control package for robot navigation. The package contains C++ implementation for core algorithms along with Python wrappers. It also implements third party integrations with [OMPL](https://ompl.kavrakilab.org/) and [FCL](https://github.com/flexible-collision-library/fcl). This package is developed to be used with [Kompass](https://github.com/automatika-robotics/kompass) package for creating navigation stacks in [ROS2](https://docs.ros.org/en/rolling/index.html). For detailed usage documentation, check Kompass [docs](https://automatika-robotics.github.io/kompass/).

## Installation

### Install Dependencies

- `sudo apt-get install libompl-dev libfcl-dev libpcl-dev`

### Install with pip

Wheels are available on Pypi for linux x86_64 and aarch64 architectures.

- `pip install kompass-core`

### Install from Source

- `git clone https://github.com/automatika-robotics/kompass-core`
- `cd kompass-core`
- `pip install .`

Installation with pip will install three Python package:

- `kompass_core`: The main Python API containing all the wrappers and utilities for motion planning and control for navigation in 2D spaces.
- `kompass_cpp`: Pybind11 python bindings for Kompass core C++ library containing the algorithms implementation for path tracking and motion control.
- `ompl`: Bespoke Pybind11 python bindings for the Open Motion Planning Library (OMPL).

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
from nav_msgs.msg import Path

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

# Set the reference path
global_path : Path = Path()

# Set the reference path for the motion control
dvz.set_path(global_path)

# Get the sensor data
laser_scan = LaserScanData()

# At each control step run
dvz.loop_step(current_state=robot.state, laser_scan=laser_scan)
```

## Copyright

The code in this distribution is Copyright (c) 2024 Automatika Robotics unless explicitly indicated otherwise.

Kompass Core is made available under the MIT license. Details can be found in the [LICENSE](LICENSE) file.

## Contributions

Kompass Core has been developed in collaboration between [Automatika Robotics](https://automatikarobotics.com/) and [Inria](https://inria.fr/). Contributions from the community are most welcome.
