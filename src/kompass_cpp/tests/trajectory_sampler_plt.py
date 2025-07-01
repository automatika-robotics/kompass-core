import json
import argparse


def read_trajectories_from_json(filename: str):
    """Read JSON trajectories saved by the trajectory sampler tester

    :param filename: JSON file name
    :type filename: str

    :return: Trajectories data
    :rtype: _type_
    """
    try:
        with open(filename, "r") as file:
            data = json.load(file)
        return data["paths"]
    except Exception as e:
        print(f"Read file error: {e}")


def read_path_from_json(filename: str):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Read file error: {e}")


def plot_samples(figure_name, trajectories, reference=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Matplotlib is not installed. Test figures will not be generated. To generate figures run 'pip install matplotlib'"
        )
        return

    if reference:
        # Plot reference
        ref_path = reference["points"]
        x_coords = [point["x"] for point in ref_path]
        y_coords = [point["y"] for point in ref_path]
        plt.plot(x_coords, y_coords, "--b", linewidth=7.0, label="reference")

    for traj in trajectories:
        path = traj["points"]
        x_coords = [point["x"] for point in path]
        y_coords = [point["y"] for point in path]
        plt.plot(x_coords, y_coords, marker="o")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(figure_name[figure_name.rindex("/") + 1 :])
    plt.grid(True)
    plt.savefig(f"./{figure_name[figure_name.rindex('/') + 1 :]}.png")


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process some strings.")

    # Define the expected arguments
    parser.add_argument(
        "--samples",
        type=str,
        required=True,
        help="Samples JSON file name in the current directory",
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=False,
        help="Reference path JSON file name in the current directory",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    samples_file = args.samples
    reference_file = args.reference

    trajectories = read_trajectories_from_json(f"{samples_file}.json")

    reference = (
        read_path_from_json(f"{reference_file}.json") if reference_file else None
    )

    # print(trajectories)
    plot_samples(samples_file, trajectories, reference)


if __name__ == "__main__":
    main()
