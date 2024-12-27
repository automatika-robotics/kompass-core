from enum import IntEnum
from typing import Any, Tuple

from .pose import PoseData


class OCCUPANCY_TYPE(IntEnum):
    """
    Integer Enum to represent the occupancy status on a grid
    """

    UNEXPLORED = -1
    EMPTY = 0
    OCCUPIED = 100


class OBSTACLE_TYPE(IntEnum):
    """
    Integer Enum to represent obstacle types

    Types:
    Scan        Obstacle obtained from laser scan data
    SEMANTIC    Obstacle obtained from semantic segmentation model from RGB input
    """

    SCAN = 1
    SEMANTIC = 2


class ObstaclesData:
    """
    Obstacles detected and mapped in grid of size width/height in the current surrounding of the robot
    the full width and height in meter is width * resolution and height * resolution
    attributes:
    resolution      float       resolution of the grid where the obstacles are occupied in meter/cell
    width           int         width of the map (1st dimension) in number of cell in the vertical direction
    height          int         height of the map (2nd dimension) in number of cell in the horizontal direction
    origin_pose     PoseData    Pose in 3D space referenced by the global frame for the upper left corner, being the (0,0) point on the grid
    robot_pose      PoseData    Pose in 3D space referenced by the global frame for the robot

    obstacle_type   int         type of obstacle, categorized by the sensor. Set to 1 for obstacles from laser scan and 2 from detections
    x_global        float       x coordinate of the obstacle in the global frame
    y_global        float       y coordinate of the obstacle in the global frame
    x_local         float       x coordinate of the obstacle in the local frame, referenced by the robot frame
    y_local         float       y coordinate of the obstacle in the local frame, referenced by the robot frame
    i_grid          int         vertical position of the obstacle on the grid
    j_grid          int         horizontal position of the obstacle on the grid
    occupied_zone   float       zone occupied by the obstacle in meter. It's radius that surround the central (i_grid, j_grid)

    class_id        int         class id in the YOLO definition of classes

    object_id       int         id for tracking the object across frames
    vx              float       velocity of the obstacle in the x-direction in the global frame
    vy              float       velocity of the obstacle in the y-direction in the global frame
    """

    def __init__(self):
        """
        initialize a Obstacles instance
        """
        # metadata
        self.resolution = 1.0
        self.width = 0
        self.height = 0
        self.origin_pose = PoseData()
        self.robot_pose = PoseData()

        # occupancy info
        self.obstacle_type = []
        self.x_global = []
        self.y_global = []
        self.x_local = []
        self.y_local = []
        self.i_grid = []
        self.j_grid = []
        self.occupied_zone = []

        # semantics info
        self.class_id = []
        # self.class_name = []

        # tracking info
        self.object_id = []
        self.vx = []
        self.vy = []

    def get_length(self) -> int:
        """
        get the length of obstacles

        :return: length of obstacles
        :rtype: int
        """
        self.check_attributes_equal_length()
        return len(self.obstacle_type)

    def check_attributes_equal_length(self):
        """
        check that all attributes has equal length
        """
        expected_length = len(self.obstacle_type)
        assert (
            len(self.x_global) == expected_length
            and len(self.y_global) == expected_length
            and len(self.x_local) == expected_length
            and len(self.y_local) == expected_length
            and len(self.i_grid) == expected_length
            and len(self.j_grid) == expected_length
            and len(self.occupied_zone) == expected_length
            and len(self.class_id) == expected_length
            and len(self.object_id) == expected_length
            and len(self.vx) == expected_length
            and len(self.vy) == expected_length
        )

    def add_obstacle(
        self,
        obstacle_type: int,
        x_global: float,
        y_global: float,
        x_local: float,
        y_local: float,
        i_grid: int,
        j_grid: int,
        occupied_zone: float,
        class_id: int,
        object_id: int,
        vx: float,
        vy: float,
    ):
        """
        add an obstacle by providing its information

        :param      obstacle_type:      type of obstacle, categorized by the sensor. Set to 1 for obstacles from laser scan and 2 from detections
        :type       obstacle_type:      int
        :param      x_global:           x coordinate of the obstacle in the global frame
        :type       x_global:           float
        :param      y_global:           y coordinate of the obstacle in the global frame
        :type       y_global:           float
        :param      x_local:            x coordinate of the obstacle in the local frame, referenced by the robot frame
        :type       x_local:            float
        :param      y_local:            y coordinate of the obstacle in the local frame, referenced by the robot frame
        :type       y_local:            float
        :param      i_grid:             vertical position of the obstacle on the grid
        :type       i_grid:             int
        :param      j_grid:             horizontal position of the obstacle on the grid
        :type       j_grid:             int
        :param      occupied_zone:      zone occupied by the obstacle in meter. It's radius that surround the central (i_grid, j_grid)
        :type       occupied_zone:      float
        :param      class_id:           class id in the YOLO definition of classes
        :type       class_id:           int
        :param      object_id:          id for tracking the object across frames
        :type       object_id:          int
        :param      vx:                 velocity of the obstacle in the x-direction in the global frame
        :type       vx:                 float
        :param      vy:                 velocity of the obstacle in the y-direction in the global frame
        :type       vy:                 float
        """
        # occupancy info
        self.obstacle_type.append(obstacle_type)
        self.x_global.append(x_global)
        self.y_global.append(y_global)
        self.x_local.append(x_local)
        self.y_local.append(y_local)
        self.i_grid.append(i_grid)
        self.j_grid.append(j_grid)
        self.occupied_zone.append(occupied_zone)

        # semantic info
        self.class_id.append(class_id)

        # tracking info
        self.object_id.append(object_id)
        self.vx.append(vx)
        self.vy.append(vy)

    def merge_obstacles(self, obstacles: Any):
        """
        Merge the current existing obstacles with new obstacles

        :param obstacles: obstacles to merge with the existing obstacles
        :type obstacles: Obstacles
        """
        self.obstacle_type.extend(obstacles.obstacle_type)
        self.x_global.extend(obstacles.x_global)
        self.y_global.extend(obstacles.y_global)
        self.x_local.extend(obstacles.x_local)
        self.y_local.extend(obstacles.y_local)
        self.i_grid.extend(obstacles.i_grid)
        self.j_grid.extend(obstacles.j_grid)
        self.occupied_zone.extend(obstacles.occupied_zone)
        self.class_id.extend(obstacles.class_id)
        self.object_id.extend(obstacles.object_id)
        self.vx.extend(obstacles.vx)
        self.vy.extend(obstacles.vy)

    def update_metadata(
        self,
        resolution: float,
        width: int,
        height: int,
        origin_pose: PoseData,
        robot_pose: PoseData,
    ):
        """
        update grid metadata

        :param      resolution:         resolution of the grid where the obstacles are occupied in meter/cell
        :type       resolution:         float
        :param      width:              width of the map (1st dimension) in number of cell in the vertical direction
        :type       width:              int
        :param      height:             height of the map (2nd dimension) in number of cell in the horizontal direction
        :type       height:             int
        :param      origin_pose:        Pose in 3D space referenced by the global frame for the upper left corner, being the (0,0) point on the grid
        :type       origin_pose:        PoseData
        :param      robot_pose:         Pose in 3D space referenced by the global frame for the robot
        :type       robot_pose:         PoseData
        """
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_pose = origin_pose
        self.robot_pose = robot_pose

    def remove_obstacle_by_index(self, obstacle_index: int):
        """
        Remove obstacle from the data using its index in the lists - CURRENTLY NOT USED

        :param obstacle_index: obstacle index in the lists
        :type obstacle_index: int
        """
        raise (NotImplementedError)

    def remove_obstacle_by_object_id(self, object_id: int):
        """
        Remove obstacle from the data using its object id - CURRENTLY NOT USED

        :param object_id: object tracking id
        :type object_id: int
        """
        raise (NotImplementedError)


def split_obstacles_by_type(
    obstacles: ObstaclesData,
) -> Tuple[ObstaclesData, ObstaclesData]:
    """
    split obstacles by obstacle_type into new instances

    :param obstacles: Obstacles to split
    :type obstacles: ObstaclesData
    :return: Obstacles instance each contains specific obstacle_type (currently supports scan and semantic types)
    :rtype: Tuple[ObstaclesData, ObstaclesData]
    """
    scan_obstacles = ObstaclesData()
    semantic_obstacles = ObstaclesData()

    scan_obstacles.width = obstacles.width
    scan_obstacles.height = obstacles.height
    scan_obstacles.resolution = obstacles.resolution
    scan_obstacles.origin_pose = obstacles.origin_pose
    scan_obstacles.robot_pose = obstacles.robot_pose

    semantic_obstacles.width = obstacles.width
    semantic_obstacles.height = obstacles.height
    semantic_obstacles.resolution = obstacles.resolution
    semantic_obstacles.origin_pose = obstacles.origin_pose
    semantic_obstacles.robot_pose = obstacles.robot_pose

    len_obstacles = len(obstacles.obstacle_type)
    for i in range(len_obstacles):
        if obstacles.obstacle_type[i] == OBSTACLE_TYPE.SCAN:
            scan_obstacles.add_obstacle(
                obstacle_type=obstacles.obstacle_type[i],
                x_global=obstacles.x_global[i],
                y_global=obstacles.y_global[i],
                x_local=obstacles.x_local[i],
                y_local=obstacles.y_local[i],
                i_grid=obstacles.i_grid[i],
                j_grid=obstacles.j_grid[i],
                occupied_zone=obstacles.occupied_zone[i],
                class_id=obstacles.class_id[i],
                object_id=obstacles.object_id[i],
                vx=obstacles.vx[i],
                vy=obstacles.vy[i],
            )

        elif obstacles.obstacle_type[i] == OBSTACLE_TYPE.SEMANTIC:
            semantic_obstacles.add_obstacle(
                obstacle_type=obstacles.obstacle_type[i],
                x_global=obstacles.x_global[i],
                y_global=obstacles.y_global[i],
                x_local=obstacles.x_local[i],
                y_local=obstacles.y_local[i],
                i_grid=obstacles.i_grid[i],
                j_grid=obstacles.j_grid[i],
                occupied_zone=obstacles.occupied_zone[i],
                class_id=obstacles.class_id[i],
                object_id=obstacles.object_id[i],
                vx=obstacles.vx[i],
                vy=obstacles.vy[i],
            )

    assert len_obstacles == (
        len(scan_obstacles.obstacle_type) + len(semantic_obstacles.obstacle_type)
    )

    return scan_obstacles, semantic_obstacles
