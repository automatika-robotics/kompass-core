"""CPU/GPU parity test for the cost evaluator.

Drives two builds of the C++ cost_evaluator_test (FORCE_CPU_BUILD=ON in
build_cpu/ and default GPU in build/), runs each with COST_PARITY_JSON
pointing at a tmp path, then compares the resulting JSON dumps test by
test. Any step that fails before both dumps exist causes pytest.skip; only
genuine numerical drift triggers a failure.

Since cost_evaluator.cpp (#ifndef GPU) and cost_evaluator_gpu.cpp (#ifdef
GPU) are compiled mutually exclusively, a single test binary can only ever
exercise one backend. This test is the only way to surface formula drift
between the two implementations.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Optional

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CPU_BUILD_DIR = REPO_ROOT / "build_cpu"
GPU_BUILD_DIR = REPO_ROOT / "build"
TEST_BIN_REL = pathlib.Path("src/kompass_cpp/tests/cost_evaluator_test")

# Relative tolerance applied per-cost. Tightenable once we're confident.
REL_TOL = 1e-4


def _say(msg: str) -> None:
    """Inline progress marker visible with pytest -s."""
    print(f"[cost_parity] {msg}", flush=True)


def _run_streaming(cmd: list[str],
                   env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run `cmd` streaming stdout/stderr to the parent so progress is live.

    With `pytest -s`, cmake/make output shows up as it's produced — avoids
    the test looking hung during a full build. Without `-s` pytest still
    captures it and shows the tail on failure.
    """
    _say("$ " + " ".join(cmd))
    return subprocess.run(cmd, env=env)


def _configure_and_build(build_dir: pathlib.Path, force_cpu: bool) -> Optional[str]:
    """Configure (if needed) and build cost_evaluator_test in `build_dir`.

    Returns None on success, or a reason string on failure (for skipping).
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    cache = build_dir / "CMakeCache.txt"
    backend = "CPU" if force_cpu else "GPU"
    if not cache.exists():
        _say(f"Configuring {backend} build in {build_dir}")
        configure = ["cmake", "-S", str(REPO_ROOT), "-B", str(build_dir)]
        if force_cpu:
            configure.append("-DFORCE_CPU_BUILD=ON")
        result = _run_streaming(configure)
        if result.returncode != 0:
            return (f"cmake configure failed for {build_dir.name} "
                    f"(exit {result.returncode}); see output above.")

    _say(f"Building cost_evaluator_test in {build_dir.name} ({backend})")
    result = _run_streaming(["cmake", "--build", str(build_dir),
                             "--target", "cost_evaluator_test", "--", "-j4"])
    if result.returncode != 0:
        return (f"cmake --build failed for {build_dir.name} "
                f"(exit {result.returncode}); see output above.")
    return None


def _run_test_binary(binary: pathlib.Path,
                     dump_path: pathlib.Path) -> Optional[str]:
    """Run the cost_evaluator_test binary asking it to dump parity JSON.

    Returns None if the dump was written (even if the process returned
    non-zero after writing — the GPU build currently aborts during SYCL
    runtime teardown, which is harmless here). Returns a reason string if
    the dump file wasn't produced.
    """
    if not binary.exists():
        return f"binary missing: {binary}"
    if dump_path.exists():
        dump_path.unlink()
    env = os.environ.copy()
    env["COST_PARITY_JSON"] = str(dump_path)
    _say(f"Running {binary} (COST_PARITY_JSON={dump_path})")
    _run_streaming([str(binary)], env=env)  # exit code ignored by design
    if not dump_path.exists():
        return f"binary did not produce dump file: {binary}"
    return None


def _load_dump(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def parity_dumps(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("cost_parity")
    cpu_dump = tmp / "cpu.json"
    gpu_dump = tmp / "gpu.json"

    # CPU
    reason = _configure_and_build(CPU_BUILD_DIR, force_cpu=True)
    if reason:
        pytest.skip(f"CPU build unavailable — {reason}")
    reason = _run_test_binary(CPU_BUILD_DIR / TEST_BIN_REL, cpu_dump)
    if reason:
        pytest.skip(f"CPU test run did not produce dump — {reason}")

    # GPU
    reason = _configure_and_build(GPU_BUILD_DIR, force_cpu=False)
    if reason:
        pytest.skip(f"GPU build unavailable (AdaptiveCpp likely missing) — "
                    f"{reason}")
    reason = _run_test_binary(GPU_BUILD_DIR / TEST_BIN_REL, gpu_dump)
    if reason:
        pytest.skip(f"GPU test run did not produce dump — {reason}")

    return {"cpu": _load_dump(cpu_dump), "gpu": _load_dump(gpu_dump)}


@pytest.mark.parity
def test_cost_parity(parity_dumps: dict, capsys: pytest.CaptureFixture) -> None:
    cpu = parity_dumps["cpu"]
    gpu = parity_dumps["gpu"]

    assert cpu.get("backend") == "cpu", (
        f"CPU dump reports backend={cpu.get('backend')!r} — did you run the "
        "right binary in build_cpu/?")
    assert gpu.get("backend") == "gpu", (
        f"GPU dump reports backend={gpu.get('backend')!r} — did you run the "
        "right binary in build/?")

    cpu_tests = cpu["tests"]
    gpu_tests = gpu["tests"]

    only_cpu = set(cpu_tests) - set(gpu_tests)
    only_gpu = set(gpu_tests) - set(cpu_tests)
    common = sorted(set(cpu_tests) & set(gpu_tests))

    # Build a per-(test, index) diff table up-front; print it on every run
    # so drift near the tolerance is always visible, not just on failure.
    rows = []
    length_mismatches = []
    for name in common:
        cpu_costs = cpu_tests[name]["costs"]
        gpu_costs = gpu_tests[name]["costs"]
        if len(cpu_costs) != len(gpu_costs):
            length_mismatches.append((name, len(cpu_costs), len(gpu_costs)))
            continue
        for i, (c, g) in enumerate(zip(cpu_costs, gpu_costs)):
            delta = abs(c - g)
            denom = max(abs(c), 1e-6)
            rel = delta / denom
            rows.append((name, i, c, g, delta, rel))

    rows.sort(key=lambda r: r[5], reverse=True)
    # Print via capsys-friendly stdout so `-s` shows it.
    print()
    print(f"{'Test':<45} {'Idx':>3} {'CPU':>14} {'GPU':>14} {'Δ':>12} "
          f"{'relΔ':>10}")
    print("-" * 102)
    for name, i, c, g, d, r in rows:
        flag = "" if r <= REL_TOL else "  <-- DRIFT"
        print(f"{name:<45} {i:>3} {c:>14.6f} {g:>14.6f} {d:>12.3e} "
              f"{r:>10.2e}{flag}")

    # Now assert.
    assert not only_cpu, f"Tests present only in CPU dump: {sorted(only_cpu)}"
    assert not only_gpu, f"Tests present only in GPU dump: {sorted(only_gpu)}"
    assert not length_mismatches, (
        "Per-test cost list length differs between CPU and GPU: "
        f"{length_mismatches}")

    failures = [r for r in rows if r[5] > REL_TOL]
    assert not failures, (
        f"CPU/GPU parity violated (rel tolerance {REL_TOL}). "
        f"{len(failures)} mismatch(es); see table above.")
