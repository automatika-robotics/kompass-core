import json
import matplotlib.pyplot as plt
import glob
import sys
import os

# --- Configuration ---
BASELINE_PLATFORM = "RK3588_CPU_Native"
COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../docs"))


def load_data():
    all_benchmarks = {}
    json_files = glob.glob("benchmark_files/*.json")
    if not json_files:
        print("[Error] No .json files found in benchmark_files/.")
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
    data_map,
    output_filename,
    log_scale=False,
    baseline_name=BASELINE_PLATFORM,
    theme="light",
):
    if not data_map:
        return

    # --- Theme Logic ---
    if theme == "dark":
        c_text = "#F0F0F0"  # White-ish text
        c_grid = "#666666"  # Lighter grid for visibility
        c_edge = "#DDDDDD"  # Light bar borders
        c_annot_bg = "#303030"  # Dark bg for annotation box
        c_error = "#FFFFFF"  # White error bars
    else:
        c_text = "#333333"  # Dark text
        c_grid = "grey"  # Standard grid
        c_edge = "black"  # Black bar borders
        c_annot_bg = "white"  # White bg for annotation box
        c_error = "#444444"  # Dark error bars

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

        # Color the spines (axes lines)
        ax.spines["left"].set_color(c_text)
        ax.spines["bottom"].set_color(c_text)
        ax.tick_params(axis="x", colors=c_text)
        ax.tick_params(axis="y", colors=c_text)

        # --- Scale Specific Logic ---
        if log_scale:
            ax.set_yscale("log")
            # Log Grid
            ax.grid(True, which="major", linestyle="-", color=c_grid, alpha=0.3)
            ax.grid(True, which="minor", linestyle=":", color=c_grid, alpha=0.15)
            ylabel = "Time (ms) - Logarithmic Scale"
        else:
            # Linear Grid
            ax.grid(True, axis="y", linestyle="--", color=c_grid, alpha=0.3)
            ylabel = "Time (ms) - Absolute"

        ax.set_axisbelow(True)

        x_pos = range(len(platforms))
        bars = ax.bar(
            x_pos,
            means,
            yerr=errors,
            capsize=6,
            color=COLORS[: len(platforms)],
            edgecolor=c_edge,
            linewidth=0.5,
            alpha=0.9,
            error_kw={"lw": 1.5, "capthick": 1.5, "ecolor": c_error},
        )

        ax.set_title(
            f"Benchmark: {test_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
            color=c_text,
        )
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold", labelpad=10, color=c_text)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(platforms, fontsize=11, fontweight="500", color=c_text)

        # --- Label Placement ---
        for bar in bars:
            height = bar.get_height()
            label_text = f"{height:.2f} ms"

            if log_scale:
                y_pos = height * 1.15
            else:
                y_pos = height + (max(means) * 0.02)

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                label_text,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=c_text,
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
                    color=c_text,
                    bbox={
                        "facecolor": c_annot_bg,
                        "alpha": 0.7,
                        "edgecolor": c_text,
                        "linewidth": 0.5,
                    },
                )

    plt.tight_layout()
    output_path = os.path.join(DOCS_DIR, output_filename)

    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    print(f"[Success] Generated chart ({theme}): {output_path}")


if __name__ == "__main__":
    data = load_data()

    # Generate 4 images total

    # 1. Linear Scale
    generate_chart(data, "benchmark_abs_light.png", log_scale=False, theme="light")
    generate_chart(data, "benchmark_abs_dark.png", log_scale=False, theme="dark")

    # 2. Log Scale
    generate_chart(data, "benchmark_log_light.png", log_scale=True, theme="light")
    generate_chart(data, "benchmark_log_dark.png", log_scale=True, theme="dark")
