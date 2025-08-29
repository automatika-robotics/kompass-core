import json
import argparse
import numpy as np


def plot_ranges_angles(json_file_path: str, output_image_path: str):
    """Plots and saves LaserScan data: 'ranges' and 'angles' from a JSON file

    :param json_file_path: Path to the JSON file containg the data
    :type json_file_path: str
    :param output_image_path: Ouutput path for the saved image
    :type output_image_path: str
    :raises ValueError: If the JSON does not contain 'ranges' or 'angles' or if they are empty
    :raises ValueError: If 'ranges' and 'angles' do not have the same length
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Matplotlib is not installed. Test figures will not be generated. To generate figures run 'pip install matplotlib'"
        )
        return
    # Load JSON data
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Extract ranges and angles
    ranges = data.get("ranges", [])
    angles = data.get("angles", [])

    if not ranges or not angles:
        raise ValueError("JSON must contain non-empty 'ranges' and 'angles' arrays.")
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


def plot_pointcloud_from_bin(file_path: str, output_image_path: str):
    """Plots and saves a 3D point cloud from a binary file

    :param file_path: Path to the binary file containing point cloud data
    :type file_path: str
    :param output_image_path: Output path for the saved HTML figure
    :type output_image_path: str
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(
            "Plotly is not installed. 3D pointcloud figures will not be generated. To generate figures run 'pip install plotly'"
        )
        return
    pts = np.fromfile(file_path, dtype=np.float32)
    if pts.size % 3 != 0:
        pts = pts[: pts.size - (pts.size % 3)]
    pts = pts.reshape(-1, 3)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=2, color=pts[:, 2], colorscale="Viridis"),
            )
        ]
    )
    fig.update_layout(
        scene_aspectmode="data", template="plotly_white", title="Sphere Point Cloud"
    )
    fig.write_html(output_image_path, include_plotlyjs="cdn")


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process some strings.")

    # Define the expected arguments
    parser.add_argument(
        "--laserscan",
        type=str,
        required=False,
        help="Laserscan JSON file name in the current directory",
    )

    parser.add_argument(
        "--pointcloud",
        type=str,
        required=False,
        help="PointCloud file name in the current directory",
    )

    # Parse the arguments
    args = parser.parse_args()

    try:
        # plot ranges and angles if the file is provided
        laserscan_file = args.laserscan

        plot_ranges_angles(f"{laserscan_file}.json", f"{laserscan_file}.png")
    except Exception:
        pass

    try:
        # Plot point cloud if the file is provided
        pointcloud_file = args.pointcloud
        plot_pointcloud_from_bin(f"{pointcloud_file}.bin", f"{pointcloud_file}.html")
    except Exception:
        pass


if __name__ == "__main__":
    main()
