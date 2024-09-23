from typing import List, Optional
import numpy as np
import cv2
import matplotlib.colors as PltColors
import matplotlib.markers as PltMarkers
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..datatypes.obstacles import ObstaclesData, OCCUPANCY_TYPE
from ..datatypes.path import PathPoint, PathSample


def plt_map_obstacles(map: ObstaclesData, ax: Axes = None):
    """
    Plot a given map's obstacles as circles

    :param map: Map containing a set of obstacles
    :type map: ObstaclesData
    :param ax: Plotting Axes, defaults to None
    :type ax: Axes, optional
    """
    if not ax:
        ax = plt.gca()

    for idx, obs_x in enumerate(map.x_global):
        obs_y = map.y_global[idx]
        circle = plt.Circle((obs_x, obs_y), map.occupied_zone[idx], color="k")
        ax.add_patch(circle)


def plt_path_sample(
    ref_path: PathSample, label="", color="blue", marker="", ax: Axes = None
):
    """
    Plot a given path sample as a line

    :param ref_path: Path points
    :type ref_path: PathSample
    :param label: Line label, defaults to ''
    :type label: str, optional
    :param color: Line color, defaults to 'blue'
    :type color: str, optional
    :param marker: Line marker, defaults to ''
    :type marker: str, optional
    :param ax: Plotting Axes, defaults to None
    :type ax: Axes, optional
    """
    if color not in PltColors.cnames.keys():
        color = "blue"

    if marker not in PltMarkers.MarkerStyle.markers.keys():
        marker = ""

    if not ax:
        ax = plt.gca()

    ax.plot(
        ref_path.x_points,
        ref_path.y_points,
        color=color,
        label=label,
        marker=marker,
    )


def plt_path_points_list(
    ref_path: List[PathPoint], label="", color="blue", marker="", ax: Axes = None
):
    """
    Plot a given list of path points as a line

    :param ref_path: Path points
    :type ref_path: List[PathPoint]
    :param label: Line label, defaults to ''
    :type label: str, optional
    :param color: Line color, defaults to 'blue'
    :type color: str, optional
    :param marker: Line marker, defaults to ''
    :type marker: str, optional
    :param ax: Plotting Axes, defaults to None
    :type ax: Axes, optional
    """
    if color not in PltColors.cnames.keys():
        color = "blue"

    if marker not in PltMarkers.MarkerStyle.markers.keys():
        marker = ""

    if not ax:
        ax = plt.gca()

    x_points = []
    y_points = []
    for i in range(len(ref_path)):
        x_points.append(ref_path[i].x)
        y_points.append(ref_path[i].y)

    ax.plot(x_points, y_points, color=color, label=label, marker=marker)


def _resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """
    resize a given image using a given scale

    :param      image:      image to rescale
    :type       image:      np.ndarray
    :param      scale:      scale factor  1.0 means the same size
    :type       scale:      float

    :return:    _description_
    :rtype:     np.ndarray
    """

    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)

    # resize image
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized_image


_COLORS_DICT = {
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "cyan": [0.0, 1.0, 1.0],
    "yellow": [1.0, 1.0, 0.0],
    "magenta": [1.0, 0.0, 1.0],
    "orange": [1.0, 0.644, 0],
    "white": [1.0, 1.0, 1.0],
    "black": [0.0, 0.0, 0.0],
    "dark_grey": [0.2, 0.2, 0.2],
    "grey": [0.5, 0.5, 0.5],
    "light_grey": [0.7, 0.7, 0.7],
}


def get_color(color_name: str, normalized: bool = False) -> List:
    """
    Given a color_name, the RGB values is returned

    :param      color_name:     the name of the color requested
    :type       color_name:     str
    :param      normalized:     whether to scale the values between [0-1] or [0-255], defaults to False
    :type       normalized:     bool, optional

    :return:    RGB values of a color
    :rtype:     List
    """
    color = _COLORS_DICT[color_name]
    if normalized:
        return color

    return (np.array(color) * 255).tolist()


# this tensor will map any HxWx1 grid (where the 3rd channel is between 1 to 256) to HxWx3 RGB-colored image
MAPPING_GRID_TO_COLOR = np.full((256, 3), 150, dtype=np.uint8)
# MAPPING_GRID_TO_COLOR[10:] = cv2.applyColorMap(
#     np.arange(246, dtype=np.uint8), cv2.COLORMAP_JET
# ).squeeze(1)[:, ::-1]
MAPPING_GRID_TO_COLOR[OCCUPANCY_TYPE.UNEXPLORED] = get_color("dark_grey")
MAPPING_GRID_TO_COLOR[OCCUPANCY_TYPE.EMPTY] = get_color("grey")
MAPPING_GRID_TO_COLOR[OCCUPANCY_TYPE.OCCUPIED] = get_color("black")

MAPPING_GRID_TO_COLOR[1] = get_color("red")
MAPPING_GRID_TO_COLOR[2] = get_color("green")
MAPPING_GRID_TO_COLOR[3] = get_color("blue")
MAPPING_GRID_TO_COLOR[4] = get_color("cyan")
MAPPING_GRID_TO_COLOR[5] = get_color("yellow")
MAPPING_GRID_TO_COLOR[6] = get_color("magenta")
MAPPING_GRID_TO_COLOR[7] = get_color("orange")
MAPPING_GRID_TO_COLOR[8] = get_color("white")
MAPPING_GRID_TO_COLOR[9] = get_color("light_grey")


def visualize_grid(
    grid_data: np.ndarray,
    scale: float = 0.0,
    show_image: bool = False,
    save_file: Optional[str] = None,
):
    """
    visualize a 2D grid as an image

    :param      grid_data:      2D grid to visualize
    :type       grid_data:      np.ndarray
    :param      scale:          scale the grid by a factor , defaults to 0.0
    :type       scale:          float, optional
    :param      show_image:     interactive visualize, defaults to False
    :type       show_image:     bool, optional
    :param      save_file:      path to save file, defaults to "" (don't save!)
    :type       save_file:      str, optional

    :return:    grid image
    :rtype:     np.ndarray
    """
    grid_image = MAPPING_GRID_TO_COLOR[
        grid_data
    ]  # map HxWx1 grid to HxWx3 RGB colored grid as image
    if show_image or save_file:
        grid_image = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)

    if scale > 0:
        grid_image = _resize_image(grid_image, scale)

    if save_file:
        file_name = save_file.split("/")[-1].split(".")[0]
        cv2.imwrite(save_file, grid_image)

    if show_image:
        window_name = "grid" if not save_file else file_name
        cv2.imshow(window_name, grid_image)
        cv2.waitkey(0)

    if show_image or save_file:
        grid_image = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)

    return grid_image
