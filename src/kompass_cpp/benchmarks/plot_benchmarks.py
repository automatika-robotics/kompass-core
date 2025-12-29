import json
import matplotlib.pyplot as plt
import glob
import sys

# --- Configuration ---
OUTPUT_IMAGE = "benchmark_comparison.png"

# Chic Color Palette
COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1"]


def load_data():
    all_benchmarks = {}  # Key: Test Name, Value: List of results

    # Get all json files
    json_files = glob.glob("*.json")
    if not json_files:
        print("[Error] No .json files found in the current directory.")
        sys.exit(1)

    print(f"[Info] Found {len(json_files)} JSON files. parsing...")

    for filename in json_files:
        try:
            with open(filename, "r") as f:
                data = json.load(f)

                # Validation
                if "platform" not in data or "benchmarks" not in data:
                    print(f"  [Skip] {filename} (Not a benchmark file)")
                    continue

                platform = data["platform"]
                print(f"  [Load] {filename} -> Platform: {platform}")

                for bench in data["benchmarks"]:
                    test_name = bench["test_name"]
                    if test_name not in all_benchmarks:
                        all_benchmarks[test_name] = []

                    all_benchmarks[test_name].append({
                        "platform": platform,
                        "mean": bench["mean_ms"],
                        "std_dev": bench["std_dev_ms"],
                    })
        except Exception as e:
            print(f"  [Error] Failed to read {filename}: {e}")

    return all_benchmarks


def plot_benchmarks(data_map):
    if not data_map:
        print("[Error] No valid benchmark data found to plot!")
        return

    test_names = list(data_map.keys())
    num_tests = len(test_names)

    # Setup Figure with transparent background
    fig, axes = plt.subplots(nrows=num_tests, ncols=1, figsize=(10, 5 * num_tests))
    if num_tests == 1:
        axes = [axes]

    # Make the figure background transparent
    fig.patch.set_alpha(0.0)

    for i, test_name in enumerate(test_names):
        ax = axes[i]
        results = data_map[test_name]

        # Sort: CPU platforms first (0), then others (1), then alphabetical
        results.sort(key=lambda x: ("CPU" not in x["platform"], x["platform"]))

        platforms = [r["platform"] for r in results]
        means = [r["mean"] for r in results]
        errors = [r["std_dev"] for r in results]

        # --- Styling ---
        # Make axes background transparent
        ax.patch.set_alpha(0.0)

        # Remove top and right borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

        # Horizontal grid, dotted, behind bars
        ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.3)
        ax.set_axisbelow(True)

        # Bar Plot
        x_pos = range(len(platforms))  # Create explicit x-positions
        bars = ax.bar(
            x_pos,
            means,
            yerr=errors,
            capsize=6,
            color=COLORS[: len(platforms)],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            error_kw={"lw": 1.5, "capthick": 1.5, "ecolor": "#444444"},
        )

        # Titles and Labels (Bold)
        ax.set_title(
            f"Benchmark: {test_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
            color="#333333",
        )
        ax.set_ylabel(
            "Time (ms)", fontsize=12, fontweight="bold", labelpad=10, color="#333333"
        )

        # Set the ticks first, THEN the labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(platforms, fontsize=11, fontweight="500", color="#333333")

        # Add Value Labels on top of bars
        for bar in bars:
            height = bar.get_height()
            label_text = f"{height:.2f} ms"

            # place above bar
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (max(means) * 0.02),
                label_text,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#222222",
            )

    plt.tight_layout()

    # Save with transparent background
    plt.savefig(OUTPUT_IMAGE, dpi=300, transparent=True, bbox_inches="tight")
    print(f"\n[Success] Generated chic benchmark chart: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    data = load_data()
    plot_benchmarks(data)
