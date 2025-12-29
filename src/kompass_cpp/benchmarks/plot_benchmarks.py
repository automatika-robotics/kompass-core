import json
import matplotlib.pyplot as plt
import glob
import sys

# --- Configuration ---
BASELINE_PLATFORM = "Rockchip_CPU_Native"
COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1"]


def load_data():
    all_benchmarks = {}
    json_files = glob.glob("benchmark_files/*.json")
    if not json_files:
        print("[Error] No .json files found.")
        sys.exit(1)

    print(f"[Info] Found {len(json_files)} JSON files. parsing...")

    for filename in json_files:
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                if "platform" not in data or "benchmarks" not in data:
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
        except Exception:
            pass

    return all_benchmarks


def generate_chart(
    data_map, output_file, log_scale=False, baseline_name=BASELINE_PLATFORM
):
    if not data_map:
        return

    test_names = list(data_map.keys())
    num_tests = len(test_names)

    fig, axes = plt.subplots(nrows=num_tests, ncols=1, figsize=(10, 6 * num_tests))
    if num_tests == 1:
        axes = [axes]

    fig.patch.set_alpha(0.0)  # Transparent background

    for i, test_name in enumerate(test_names):
        ax = axes[i]
        results = data_map[test_name]

        # Sort: CPU first (0), then others (1)
        results.sort(key=lambda x: ("CPU" not in x["platform"], x["platform"]))

        platforms = [r["platform"] for r in results]
        means = [r["mean"] for r in results]
        errors = [r["std_dev"] for r in results]

        # --- Styling ---
        ax.patch.set_alpha(0.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

        # --- Scale Specific Logic ---
        if log_scale:
            ax.set_yscale("log")
            # Log Grid (Minor ticks are important here)
            ax.grid(True, which="major", linestyle="-", color="grey", alpha=0.3)
            ax.grid(True, which="minor", linestyle=":", color="grey", alpha=0.15)
            ylabel = "Time (ms) - Logarithmic Scale"
        else:
            # Linear Grid
            ax.grid(True, axis="y", linestyle="--", color="grey", alpha=0.3)
            ylabel = "Time (ms) - Absolute"

        ax.set_axisbelow(True)

        x_pos = range(len(platforms))
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

        ax.set_title(
            f"Benchmark: {test_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
            color="#333333",
        )
        ax.set_ylabel(
            ylabel, fontsize=12, fontweight="bold", labelpad=10, color="#333333"
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(platforms, fontsize=11, fontweight="500", color="#333333")

        # --- Label Placement ---
        for bar in bars:
            height = bar.get_height()
            label_text = f"{height:.2f} ms"

            # Position logic changes based on scale
            if log_scale:
                y_pos = height * 1.15  # Multiplicative spacing for log
            else:
                y_pos = height + (max(means) * 0.02)  # Additive spacing for linear

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                label_text,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#222222",
            )

        # --- Speedup Annotation ---
        baseline_time = next(
            (r["mean"] for r in results if r["platform"] == baseline_name), None
        )

        if baseline_time and len(means) > 1:
            fastest_time = min(means)
            if fastest_time > 0:
                speedup = baseline_time / fastest_time
                if speedup < 1.01:
                    annot_text = f"Baseline: {baseline_name}"
                else:
                    annot_text = f"Max Speedup vs CPU Native: {speedup:.1f}x"

                ax.text(
                    0.95,
                    0.9,
                    annot_text,
                    transform=ax.transAxes,
                    ha="right",
                    fontsize=12,
                    color="#333333",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True, bbox_inches="tight")
    print(f"[Success] Generated chart: {output_file}")


if __name__ == "__main__":
    data = load_data()

    # Generate Linear Scale Image
    generate_chart(data, "benchmark_comparison_absolute.png", log_scale=False)

    # Generate Log Scale Image
    generate_chart(data, "benchmark_comparison_log.png", log_scale=True)
