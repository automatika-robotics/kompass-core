from typing import Optional

import fcl
import numpy as np

from ...models import RobotState

from .config import FCLConfig, fcl_object_geometry


class FCL:
    def __init__(
        self, config: Optional[FCLConfig] = None, map_3d: Optional[np.ndarray] = None
    ) -> None:
        """
        Setup a handler using the Flexible Collision Library (FCL)

        :param config: FCL configuration parameters
        :type config: FCLConfig

        :param map_3d: 3D array for map PointCloud data, defaults to None
        :type map_3d: np.ndarray | None, optional

        :raises Exception: FCL exceptions
        """
        try:
            if config:
                self._config = config
            else:
                self._config = FCLConfig()

            self._setup_from_config()

            if map_3d is not None:
                self.set_map(map_3d)
        except Exception as e:
            raise Exception(f"FCL geometry setup error: {e}") from e

    def _setup_from_config(self):
        """
        Setup fcl from provided config
        """
        self._robot_geometry = fcl_object_geometry[self._config.robot_geometry_type](
            *tuple(self._config.robot_geometry_params)
        )
        self._map_resolution = self._config.map_resolution

        self._collision_manager = fcl.DynamicAABBTreeCollisionManager()

        self.got_map = False

    def configure(self, yaml_file: str, root_name: Optional[str] = None):
        """
        Load configuration from yaml

        :param yaml_file: Path to config file (.yaml)
        :type yaml_file: str
        :param root_name: Parent root name of the config in the file 'Parent.fcl' - config must be under 'fcl', defaults to None
        :type root_name: str | None, optional
        """
        if root_name:
            nested_root_name = root_name + ".fcl"
        else:
            nested_root_name = "fcl"
        self._config.from_yaml(yaml_file, nested_root_name)
        self._setup_from_config()

    def update_state(self, robot_state: RobotState):
        """
        Update robot collision object from new robot state

        :param robot_state: Robot pose (x, y, yaw)
        :type robot_state: RobotState
        """
        self._robot_object = fcl.CollisionObject(self._robot_geometry)

        # Get translation and rotation from robot state
        translation = np.array([robot_state.x, robot_state.y, 0.0])
        c_t = np.cos(robot_state.yaw)
        s_t = np.sin(robot_state.yaw)
        R = np.array([[c_t, -s_t, 0.0], [s_t, c_t, 0.0], [0.0, 0.0, 1.0]])
        transform = fcl.Transform(R, translation)

        # Set transform to robot object
        self._robot_object.setTransform(transform)

    def set_map(self, map_3d: np.ndarray):
        """
        Set new map to the collision manager

        :param map_3d: 3D array (PointCloud data)
        :type map_3d: np.ndarray
        """
        map_octree = fcl.OcTree(self._map_resolution, points=map_3d)
        self._map_object = fcl.CollisionObject(map_octree)
        self.got_map = True

    def check_collision(self) -> bool:
        """
        Check collision between the robot and the map obstacles

        :return: If any collisions are found
        :rtype: bool
        """
        # Clear olf robot pose and old map data from the collision manager
        self._collision_manager.clear()

        # Register up to dat data to manager
        self._collision_manager.registerObjects([self._map_object, self._robot_object])
        self._collision_manager.setup()

        # Get collision between objects registered in the manager
        collision_data = fcl.CollisionData()
        self._collision_manager.collide(collision_data, fcl.defaultCollisionCallback)
        return collision_data.result.is_collision
