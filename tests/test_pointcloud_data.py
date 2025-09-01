import os
import json
import numpy as np
from kompass_cpp.utils import pointcloud_to_laserscan_from_raw
from kompass_core.datatypes import PointCloudData


def plot_ranges_angles(angles: list, ranges: list, output_image_path: str):
    """Plots and saves LaserScan data: 'ranges' and 'angles' from a JSON file

    :param angles: List of angles in radians
    :type angles: list
    :param ranges: List of ranges corresponding to the angles (m)
    :type ranges: list
    :param output_image_path: Path to save the output image
    :type output_image_path: str
    :raises ValueError: If 'ranges' and 'angles' do not have the same length
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Matplotlib is not installed. Test figures will not be generated. To generate figures run 'pip install matplotlib'"
        )
        return

    if len(ranges) != len(angles):
        raise ValueError("'ranges' and 'angles' must have the same length.")

    # Create polar plot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(angles, ranges, marker="o")

    ax.set_theta_zero_location("N")  # 0° at top
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_title("Ranges vs Angles", va="bottom")

    # Save plot to file
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pointcloud_from_json(file_path: str, output_image_path: str) -> PointCloudData:
    """Plots and saves a 3D point cloud from a JSON file

    :param file_path: Path to the JSON file containing point cloud data
    :type file_path: str
    :param output_image_path: Path to save the output HTML figure
    :type output_image_path: str
    :return: PointCloudData object containing the parsed data
    :rtype: PointCloudData
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(
            "Plotly is not installed. 3D pointcloud figures will not be generated. To generate figures run 'pip install plotly'"
        )
        return
    # Load JSON
    with open(file_path, "r") as f:
        pc_json = json.load(f)

    # Read data similar to callback in ROS2
    pc = PointCloudData(
            point_step=pc_json["point_step"],
            row_step=pc_json["row_step"],
            data=np.array(pc_json["data"], dtype=np.int8),
            height=pc_json["height"],
            width=pc_json["width"]
        )

    # Extract raw byte data
    data = pc_json["data"]
    point_step = pc_json["point_step"]
    fields = pc_json["fields"]
    width = pc_json["width"]
    height = pc_json["height"]

    # Find offsets for x, y, z
    offset_map = {f["name"]: f["offset"] for f in fields}
    offset_x = offset_map.get("x")
    pc.x_offset = offset_x
    offset_y = offset_map.get("y")
    pc.y_offset = offset_y
    offset_z = offset_map.get("z")
    pc.z_offset = offset_z
    if offset_x is None or offset_y is None or offset_z is None:
        raise ValueError("JSON missing x, y, or z fields")

    # Convert raw bytes to numpy array
    buffer = np.array(data, dtype=np.uint8)
    num_points = width * height

    # Extract coordinates
    points = []
    for i in range(num_points):
        base = i * point_step
        x = np.frombuffer(
            buffer[base + offset_x : base + offset_x + 4], dtype=np.float32
        )[0]
        y = np.frombuffer(
            buffer[base + offset_y : base + offset_y + 4], dtype=np.float32
        )[0]
        z = np.frombuffer(
            buffer[base + offset_z : base + offset_z + 4], dtype=np.float32
        )[0]
        points.append((x, y, z))

    points = np.array(points)

    # Plot using Plotly
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={
                    "size": 2,
                    "color": points[:, 2],  # color by Z value
                    "colorscale": "Viridis",
                    "opacity": 0.8,
                },
            )
        ]
    )

    fig.update_layout(
        scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
        title="PointCloud2 3D Plot",
    )
    fig.write_html(output_image_path, include_plotlyjs="cdn")
    return pc


def main():
    """Main function to test point cloud to LaserScan conversion
    """
    max_range = 10.0
    min_z = 1.6
    max_z = 1.8
    angle_step = 0.05

    dir_name = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(dir_name, "resources/mapping")

    data = plot_pointcloud_from_json(file_path=os.path.join(resources_path, "livox_pointcloud_sample_1.json"),
                                     output_image_path=os.path.join(resources_path, "pointcloud_plot.html"))
    ranges, angles = pointcloud_to_laserscan_from_raw(
        data=data.data,
        point_step=data.point_step,
        row_step=data.row_step,
        height=data.height,
        width=data.width,
        x_offset=data.x_offset,
        y_offset=data.y_offset,
        z_offset=data.z_offset,
        max_range=max_range,
        min_z=min_z,
        max_z=max_z,
        angle_step=angle_step,
    )
    plot_ranges_angles(angles, ranges, os.path.join(resources_path, "pointcloud_to_laserscan.png"))


if __name__ == "__main__":
    main()
