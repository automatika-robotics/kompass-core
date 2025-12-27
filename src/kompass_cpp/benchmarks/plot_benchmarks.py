import json
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# filenames expected for benchmarking data
EXPECTED_FILES = [
    "benchmark_cpu_native.json",
    "benchmark_cpu_omp.json",
    "benchmark_cuda.json",
    "benchmark_rocm.json",
]

OUTPUT_IMAGE = "benchmark_comparison.png"


def load_data():
    all_benchmarks = {}  # Key: Test Name, Value: List of results

    for filename in EXPECTED_FILES:
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Skipping.")
            continue

        with open(filename, "r") as f:
            data = json.load(f)
            platform = data["platform"]

            for bench in data["benchmarks"]:
                test_name = bench["test_name"]
                if test_name not in all_benchmarks:
                    all_benchmarks[test_name] = []

                all_benchmarks[test_name].append({
                    "platform": platform,
                    "mean": bench["mean_ms"],
                    "std_dev": bench["std_dev_ms"],
                })
    return all_benchmarks


def plot_benchmarks(data_map):
    if not data_map:
        print("No data found to plot!")
        return

    test_names = list(data_map.keys())
    num_tests = len(test_names)

    # Create subplots (one row per test case)
    _, axes = plt.subplots(nrows=num_tests, ncols=1, figsize=(10, 4 * num_tests))
    if num_tests == 1:
        axes = [axes]  # Handle single case

    # Colors for platforms
    colors = ["#3498db", "#9b59b6", "#2ecc71", "#e74c3c"]

    for i, test_name in enumerate(test_names):
        ax = axes[i]
        results = data_map[test_name]

        # Sort results to keep platforms consistent (optional)
        # results.sort(key=lambda x: x['platform'])

        platforms = [r["platform"] for r in results]
        means = [r["mean"] for r in results]
        errors = [r["std_dev"] for r in results]

        # Create Bar Chart
        bars = ax.bar(
            platforms,
            means,
            yerr=errors,
            capsize=5,
            color=colors[: len(platforms)],
            alpha=0.8,
        )

        # Formatting
        ax.set_title(f"Benchmark: {test_name}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Time (ms) - Lower is Better", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        # Add numeric labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                1.02 * height,
                f"{height:.3f} ms",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Successfully generated {OUTPUT_IMAGE}")


if __name__ == "__main__":
    data = load_data()
    plot_benchmarks(data)
