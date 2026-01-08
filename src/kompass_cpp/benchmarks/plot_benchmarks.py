import json
import matplotlib.pyplot as plt
import glob
import sys
import os

# Configuration
BASELINE_PLATFORM = "RK3588_CPU"
COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1"]

# Determine the absolute path to the docs folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../docs"))

# Ensure docs directory exists
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)


def load_data():
    all_benchmarks = {}
    # Fallback to local dir if benchmark_files doesn't exist/empty
    search_path = "benchmark_files/*.json"
    if not glob.glob(search_path):
        search_path = "*.json"

    json_files = glob.glob(search_path)
    if not json_files:
        print("[Error] No .json files found.")
        sys.exit(1)

    print(f"[Info] Found {len(json_files)} JSON files. Parsing...")

    for filename in json_files:
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                if "platform" not in data or "benchmarks" not in data:
                    continue

                platform = data["platform"]
                for bench in data["benchmarks"]:
                    test_name = bench["test_name"]
                    if test_name not in all_benchmarks:
                        all_benchmarks[test_name] = []

                    entry = {
                        "platform": platform,
                        "mean": bench["mean_ms"],
                        "std_dev": bench["std_dev_ms"],
                    }

                    # Capture Power if available
                    if "avg_power_w" in bench:
                        entry["power"] = bench["avg_power_w"]

                    all_benchmarks[test_name].append(entry)
        except Exception as e:
            print(f"  [Warn] Failed to read {filename}: {e}")

    return all_benchmarks


def get_theme_colors(theme):
    if theme == "dark":
        return {
            "text": "#F0F0F0",
            "grid": "#666666",
            "edge": "#DDDDDD",
            "annot_bg": "#303030",
            "error": "#FFFFFF",
        }
    else:
        return {
            "text": "#333333",
            "grid": "grey",
            "edge": "black",
            "annot_bg": "white",
            "error": "#444444",
        }


def setup_ax_style(ax, theme_colors, log_scale=False):
    c = theme_colors
    ax.patch.set_alpha(0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.spines["left"].set_color(c["text"])
    ax.spines["bottom"].set_color(c["text"])
    ax.tick_params(axis="x", colors=c["text"])
    ax.tick_params(axis="y", colors=c["text"])

    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="-", color=c["grid"], alpha=0.3)
        ax.grid(True, which="minor", linestyle=":", color=c["grid"], alpha=0.15)
    else:
        ax.grid(True, axis="y", linestyle="--", color=c["grid"], alpha=0.3)

    ax.set_axisbelow(True)


def generate_perf_chart(
    data_map, output_filename, baseline_name=BASELINE_PLATFORM, theme="light"
):
    if not data_map:
        return

    c = get_theme_colors(theme)
    test_names = list(data_map.keys())
    num_tests = len(test_names)

    fig, axes = plt.subplots(nrows=num_tests, ncols=1, figsize=(10, 6 * num_tests))
    if num_tests == 1:
        axes = [axes]
    fig.patch.set_alpha(0.0)

    for i, test_name in enumerate(test_names):
        ax = axes[i]

        # Only include results that do not have the 'power' measurement
        results = [r for r in data_map[test_name] if "power" not in r]

        if not results:
            ax.text(
                0.5,
                0.5,
                "No 'pure' benchmark data found\n(All files contain power stats)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=c["text"],
            )
            continue

        results.sort(key=lambda x: ("CPU" not in x["platform"], x["platform"]))

        platforms = [r["platform"] for r in results]
        means = [r["mean"] for r in results]
        errors = [r["std_dev"] for r in results]

        setup_ax_style(ax, c, log_scale=True)

        x_pos = range(len(platforms))
        bars = ax.bar(
            x_pos,
            means,
            yerr=errors,
            capsize=6,
            color=COLORS[: len(platforms)],
            edgecolor=c["edge"],
            linewidth=0.5,
            alpha=0.9,
            error_kw={"lw": 1.5, "capthick": 1.5, "ecolor": c["error"]},
        )

        ax.set_title(
            f"Benchmark: {test_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
            color=c["text"],
        )
        ax.set_ylabel(
            "Time (ms) - Log Scale",
            fontsize=12,
            fontweight="bold",
            labelpad=10,
            color=c["text"],
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(platforms, fontsize=11, fontweight="500", color=c["text"])

        # Labels and speedup
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height * 1.15,
                f"{height:.2f} ms",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=c["text"],
            )

        # Baseline Logic
        baseline_record = next(
            (r for r in results if r["platform"] == baseline_name), None
        )
        if not baseline_record:  # Fallback
            baseline_record = next(
                (
                    r
                    for r in results
                    if "CPU" in r["platform"] and "Native" in r["platform"]
                ),
                None,
            )

        if baseline_record and len(means) > 1:
            baseline_time = baseline_record["mean"]
            real_base_name = baseline_record["platform"]
            fastest_time = min(means)
            if fastest_time > 0:
                speedup = baseline_time / fastest_time
                txt = (
                    f"Baseline: {real_base_name}"
                    if speedup < 1.05
                    else f"Max Speedup vs baseline: {speedup:.1f}x"
                )
                ax.text(
                    0.95,
                    0.9,
                    txt,
                    transform=ax.transAxes,
                    ha="right",
                    fontsize=12,
                    color=c["text"],
                    bbox={
                        "facecolor": c["annot_bg"],
                        "alpha": 0.7,
                        "edgecolor": c["text"],
                        "linewidth": 0.5,
                    },
                )

    plt.tight_layout()
    output_path = os.path.join(DOCS_DIR, output_filename)
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    print(f"[Success] Generated Perf Chart ({theme}): {output_path}")


def generate_power_chart(data_map, output_filename, theme="light"):
    if not data_map:
        return

    # Filter tests that have at least one platform with power data
    valid_tests = []
    for test, results in data_map.items():
        if any("power" in r for r in results):
            valid_tests.append(test)

    if not valid_tests:
        print(f"[Skip] No power data found for chart ({theme}).")
        return

    c = get_theme_colors(theme)
    num_tests = len(valid_tests)

    # Create 2 columns: Left=Power(W), Right=Perf/Watt
    fig, axes = plt.subplots(nrows=num_tests, ncols=2, figsize=(18, 6 * num_tests))
    if num_tests == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array access
    fig.patch.set_alpha(0.0)

    for i, test_name in enumerate(valid_tests):
        # Filter only results with power
        results = [r for r in data_map[test_name] if "power" in r]
        results.sort(key=lambda x: ("CPU" not in x["platform"], x["platform"]))

        platforms = [r["platform"] for r in results]
        power_vals = [r["power"] for r in results]

        # Efficiency Metric: (Runs per Second) / Watts
        # 1 run = mean_ms / 1000 seconds
        efficiency_vals = [(1000.0 / r["mean"]) / r["power"] for r in results]

        # Plot Power (Watts)
        ax_p = axes[i, 0]
        setup_ax_style(ax_p, c, log_scale=False)
        x_pos = range(len(platforms))
        bars_p = ax_p.bar(
            x_pos,
            power_vals,
            color=COLORS[: len(platforms)],
            edgecolor=c["edge"],
            linewidth=0.5,
            alpha=0.9,
        )

        ax_p.set_title(
            f"{test_name} - Consumption",
            fontsize=14,
            fontweight="bold",
            color=c["text"],
        )
        ax_p.set_ylabel(
            "Avg Power (Watts)", fontsize=11, fontweight="bold", color=c["text"]
        )
        ax_p.set_xticks(x_pos)
        ax_p.set_xticklabels(
            platforms,
            fontsize=10,
            fontweight="500",
            color=c["text"],
            rotation=15,
            ha="right",
        )

        for bar in bars_p:
            ax_p.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + (max(power_vals) * 0.02),
                f"{bar.get_height():.2f} W",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=c["text"],
            )

        # Plot Efficiency (Perf/Watt)
        ax_e = axes[i, 1]
        setup_ax_style(ax_e, c, log_scale=False)
        bars_e = ax_e.bar(
            x_pos,
            efficiency_vals,
            color=COLORS[: len(platforms)],
            edgecolor=c["edge"],
            linewidth=0.5,
            alpha=0.9,
        )

        ax_e.set_title(
            f"{test_name} - Efficiency", fontsize=14, fontweight="bold", color=c["text"]
        )
        ax_e.set_ylabel(
            "Perf / Watt (Runs/sec/W)", fontsize=11, fontweight="bold", color=c["text"]
        )
        ax_e.set_xticks(x_pos)
        ax_e.set_xticklabels(
            platforms,
            fontsize=10,
            fontweight="500",
            color=c["text"],
            rotation=15,
            ha="right",
        )

        for bar in bars_e:
            ax_e.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + (max(efficiency_vals) * 0.02),
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=c["text"],
            )

    plt.tight_layout()
    output_path = os.path.join(DOCS_DIR, output_filename)
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    print(f"[Success] Generated Power Chart ({theme}): {output_path}")


if __name__ == "__main__":
    data = load_data()

    # 1. Performance Charts (Only Pure Runs, Log Scale)
    generate_perf_chart(data, "benchmark_log_light.png", theme="light")
    generate_perf_chart(data, "benchmark_log_dark.png", theme="dark")

    # 2. Power and Efficiency Charts (Only Runs With Power)
    generate_power_chart(data, "benchmark_power_light.png", theme="light")
    generate_power_chart(data, "benchmark_power_dark.png", theme="dark")
