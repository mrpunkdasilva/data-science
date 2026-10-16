"""Latency and throughput benchmark for the Lab 7 cancer classification API.

Two paths are measured under increasing levels of concurrency:

* ``model (direct)``: in-process inference through the cached model service.
* ``api (/predict)``: the full request cycle through the FastAPI application.

The results are written to ``data/latency_benchmark.csv``, plotted to
``data/latency_benchmark.png`` and printed as Markdown tables so they can be
pasted straight into the README. Every scenario is measured ``--repeat`` times
and the median of each statistic is reported, which damps the scheduler noise
of a shared development machine. The script exits with status 1 when the p95
latency of any scenario breaks the 100 ms budget.

Usage::

    python benchmark.py
    python benchmark.py --levels 1 4 8 --repeat 1 --no-plot
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence, TypedDict

import httpx
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.datasets import load_breast_cancer

LAB_ROOT = Path(__file__).resolve().parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from src import config
from src.app import app
from src.model import clear_service_cache, get_service

DEFAULT_LEVELS = (1, 2, 4, 8, 16)
DEFAULT_REPEAT = 3
DIRECT_REQUESTS_PER_WORKER = 200
API_REQUESTS_PER_WORKER = 100
WARMUP_CALLS = 20
CSV_PATH = config.DATA_DIR / "latency_benchmark.csv"
PLOT_PATH = config.DATA_DIR / "latency_benchmark.png"
MARKDOWN_PATH = config.DATA_DIR / "latency_benchmark.md"


class BenchmarkRow(TypedDict):
    concurrency: int
    requests: int
    throughput_rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    path: str


def environment_summary() -> str:
    """Describe the machine so the reported numbers can be interpreted."""
    load = ", ".join(f"{value:.2f}" for value in os.getloadavg())
    return (
        f"Python {platform.python_version()} on {platform.system()} "
        f"| {os.cpu_count()} cores | load average {load}"
    )


def load_sample() -> list[float]:
    """Return one malignant feature vector to feed every scenario."""
    data = load_breast_cancer()
    index = next(
        i for i, value in enumerate(data.target) if value == config.MALIGNANT_LABEL
    )
    return [float(value) for value in data.data[index]]


def measure_model_load_ms() -> float:
    """Time a cold start of the model service (file read + pipeline build)."""
    clear_service_cache()
    start = time.perf_counter()
    get_service()
    return (time.perf_counter() - start) * 1000.0


def _summarise(
    latencies: npt.NDArray[np.float64], wall: float, concurrency: int, path: str
) -> BenchmarkRow:
    return BenchmarkRow(
        concurrency=concurrency,
        requests=int(latencies.size),
        throughput_rps=float(latencies.size / wall),
        p50_ms=float(np.percentile(latencies, 50)),
        p95_ms=float(np.percentile(latencies, 95)),
        p99_ms=float(np.percentile(latencies, 99)),
        path=path,
    )


def median_row(runs: Sequence[BenchmarkRow]) -> BenchmarkRow:
    """Median statistics across repeated runs (damps scheduler noise)."""
    first = runs[0]
    return BenchmarkRow(
        concurrency=first["concurrency"],
        requests=int(np.median([run["requests"] for run in runs])),
        throughput_rps=float(np.median([run["throughput_rps"] for run in runs])),
        p50_ms=float(np.median([run["p50_ms"] for run in runs])),
        p95_ms=float(np.median([run["p95_ms"] for run in runs])),
        p99_ms=float(np.median([run["p99_ms"] for run in runs])),
        path=first["path"],
    )


def benchmark_direct(
    features: list[float], concurrency: int, requests_per_worker: int
) -> BenchmarkRow:
    """Measure in-process inference latency with a thread pool."""
    service = get_service()
    for _ in range(WARMUP_CALLS):
        service.predict_one(features)

    def worker(_: int) -> list[float]:
        latencies: list[float] = []
        for _ in range(requests_per_worker):
            start = time.perf_counter()
            service.predict_one(features)
            latencies.append((time.perf_counter() - start) * 1000.0)
        return latencies

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        batches = list(pool.map(worker, range(concurrency)))
    wall = time.perf_counter() - start

    latencies = np.concatenate(
        [np.asarray(batch, dtype=np.float64) for batch in batches]
    )
    return _summarise(latencies, wall, concurrency, "model (direct)")


async def benchmark_api(
    features: list[float], concurrency: int, requests_per_worker: int
) -> BenchmarkRow:
    """Measure the full HTTP round trip through the FastAPI application."""
    payload = {"features": features}
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport, base_url="http://benchmark"
    ) as client:
        await client.get("/health")

        async def worker() -> list[float]:
            latencies: list[float] = []
            for _ in range(requests_per_worker):
                start = time.perf_counter()
                response = await client.post("/predict", json=payload)
                response.raise_for_status()
                latencies.append((time.perf_counter() - start) * 1000.0)
            return latencies

        start = time.perf_counter()
        batches = await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - start

    latencies = np.concatenate(
        [np.asarray(batch, dtype=np.float64) for batch in batches]
    )
    return _summarise(latencies, wall, concurrency, "api (/predict)")


def plot(benchmark: pd.DataFrame) -> None:
    """Plot p95 latency and throughput against concurrency."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for path, group in benchmark.groupby("path"):
        ordered = group.sort_values("concurrency")
        axes[0].plot(ordered["concurrency"], ordered["p95_ms"], marker="o", label=path)
        axes[1].plot(
            ordered["concurrency"], ordered["throughput_rps"], marker="o", label=path
        )

    axes[0].axhline(
        config.LATENCY_BUDGET_MS, color="black", linestyle="--", label="100 ms budget"
    )
    axes[0].set_xlabel("Concurrent workers")
    axes[0].set_ylabel("p95 latency (ms)")
    axes[0].set_title("Latency under concurrency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Concurrent workers")
    axes[1].set_ylabel("Throughput (requests/s)")
    axes[1].set_title("Throughput under concurrency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Inference performance: direct model vs HTTP API", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def to_markdown(benchmark: pd.DataFrame) -> str:
    """Render one Markdown table per measured path."""
    sections: list[str] = []
    for path, group in benchmark.groupby("path", sort=False):
        lines = [
            f"### {path}",
            "",
            "| Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) |",
            "|-------------|----------|----------|----------|--------------------|",
        ]
        for _, row in group.sort_values("concurrency").iterrows():
            lines.append(
                f"| {int(row['concurrency'])} "
                f"| {float(row['p50_ms']):.2f} "
                f"| {float(row['p95_ms']):.2f} "
                f"| {float(row['p99_ms']):.2f} "
                f"| ~{float(row['throughput_rps']):.0f} |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="Concurrency levels to benchmark (default: 1 2 4 8 16).",
    )
    parser.add_argument(
        "--direct-requests",
        type=int,
        default=DIRECT_REQUESTS_PER_WORKER,
        help="Requests per worker on the direct model path.",
    )
    parser.add_argument(
        "--api-requests",
        type=int,
        default=API_REQUESTS_PER_WORKER,
        help="Requests per worker on the HTTP API path.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT,
        help="Runs per scenario; the median statistic is reported (default: 3).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG generation.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    features = load_sample()
    load_ms = measure_model_load_ms()
    environment = environment_summary()
    print(environment)
    print(f"Cold model load: {load_ms:.1f} ms ({config.MODEL_PATH.name})")

    rows: list[BenchmarkRow] = []
    all_runs: list[BenchmarkRow] = []
    for level in args.levels:
        runs = [
            benchmark_direct(features, level, args.direct_requests)
            for _ in range(args.repeat)
        ]
        all_runs.extend(runs)
        row = median_row(runs)
        rows.append(row)
        print(
            f"model (direct) @ concurrency {level}: "
            f"p95 {row['p95_ms']:.2f} ms (median of {args.repeat} runs)"
        )

    for level in args.levels:
        runs = [
            asyncio.run(benchmark_api(features, level, args.api_requests))
            for _ in range(args.repeat)
        ]
        all_runs.extend(runs)
        row = median_row(runs)
        rows.append(row)
        print(
            f"api (/predict) @ concurrency {level}: "
            f"p95 {row['p95_ms']:.2f} ms (median of {args.repeat} runs)"
        )

    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(CSV_PATH, index=False)

    if not args.no_plot:
        plot(benchmark)

    markdown_lines = [
        "# Latência e vazão da inferência",
        "",
        f"- Ambiente: {environment}",
        f"- Carga do modelo a frio: {load_ms:.1f} ms ({config.MODEL_PATH.name})",
        f"- Agregação: mediana de {args.repeat} execução(ões) por cenário",
        f"- Orçamento de latência: {config.LATENCY_BUDGET_MS:.0f} ms (p95)",
        "",
        to_markdown(benchmark),
        "",
    ]
    MARKDOWN_PATH.write_text("\n".join(markdown_lines), encoding="utf-8")

    print()
    print(to_markdown(benchmark))
    print()
    written = [CSV_PATH.relative_to(LAB_ROOT), MARKDOWN_PATH.relative_to(LAB_ROOT)]
    if not args.no_plot:
        written.append(PLOT_PATH.relative_to(LAB_ROOT))
    print("Results written to " + ", ".join(str(path) for path in written))

    worst_run = max(all_runs, key=lambda run: run["p95_ms"])
    print(
        "Single-run extremes: highest p95 "
        f"{worst_run['p95_ms']:.2f} ms on {worst_run['path']} "
        f"at concurrency {worst_run['concurrency']} "
        f"(reported values are medians of {args.repeat} runs)."
    )

    worst = max(rows, key=lambda row: row["p95_ms"])
    worst_p95 = worst["p95_ms"]
    if worst_p95 >= config.LATENCY_BUDGET_MS:
        print(
            f"FAIL: worst p95 {worst_p95:.2f} ms on {worst['path']} "
            f"at concurrency {worst['concurrency']} "
            f"exceeds the {config.LATENCY_BUDGET_MS:.0f} ms budget."
        )
        return 1

    print(
        f"OK: worst p95 {worst_p95:.2f} ms on {worst['path']} "
        f"is within the {config.LATENCY_BUDGET_MS:.0f} ms budget."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
