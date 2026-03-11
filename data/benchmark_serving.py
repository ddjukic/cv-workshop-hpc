#!/usr/bin/env python3
"""Load-test a YOLO serving endpoint.

Sends concurrent requests to a YOLO serving endpoint and measures
latency percentiles and throughput at various concurrency levels.

Usage:
    uv run python data/benchmark_serving.py \
        --url http://localhost:8001/predict \
        --image data/milk_images/milk-video-3-00000.png \
        --requests 30 \
        --concurrency 1,2,4,8 \
        --output results.json
"""

import argparse
import asyncio
import json
import statistics
import sys
import time

import httpx


async def send_request(
    client: httpx.AsyncClient, url: str, image_bytes: bytes, filename: str
) -> float:
    """Send one prediction request and return latency in milliseconds."""
    t0 = time.perf_counter()
    response = await client.post(
        url,
        files={"file": (filename, image_bytes, "image/png")},
        timeout=60.0,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    response.raise_for_status()
    return latency_ms


async def run_concurrency_level(
    url: str,
    image_bytes: bytes,
    filename: str,
    concurrency: int,
    num_requests: int,
) -> dict:
    """Run num_requests at a given concurrency level and collect metrics."""
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0

    async def bounded_request(client: httpx.AsyncClient):
        nonlocal errors
        async with semaphore:
            try:
                lat = await send_request(client, url, image_bytes, filename)
                latencies.append(lat)
            except Exception as e:
                errors += 1
                print(f"  Error: {e}", file=sys.stderr)

    async with httpx.AsyncClient() as client:
        wall_start = time.perf_counter()
        tasks = [bounded_request(client) for _ in range(num_requests)]
        await asyncio.gather(*tasks)
        wall_elapsed = time.perf_counter() - wall_start

    if not latencies:
        return {
            "concurrency": concurrency,
            "requests": num_requests,
            "errors": errors,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "throughput_rps": 0,
        }

    latencies.sort()
    n = len(latencies)

    def percentile(pct: float) -> float:
        idx = int(pct / 100 * (n - 1))
        return latencies[idx]

    return {
        "concurrency": concurrency,
        "requests": num_requests,
        "successful": n,
        "errors": errors,
        "p50_ms": round(percentile(50), 1),
        "p95_ms": round(percentile(95), 1),
        "p99_ms": round(percentile(99), 1),
        "mean_ms": round(statistics.mean(latencies), 1),
        "throughput_rps": round(n / wall_elapsed, 2),
    }


def print_table(all_results: list[dict], server_name: str):
    """Print a clean ASCII table of benchmark results."""
    header = f"{'Server':<12} | {'Conc':>4} | {'p50 (ms)':>9} | {'p95 (ms)':>9} | {'p99 (ms)':>9} | {'Mean (ms)':>10} | {'Throughput':>12}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in all_results:
        p50 = f"{r['p50_ms']:.1f}" if r["p50_ms"] is not None else "N/A"
        p95 = f"{r['p95_ms']:.1f}" if r["p95_ms"] is not None else "N/A"
        p99 = f"{r['p99_ms']:.1f}" if r["p99_ms"] is not None else "N/A"
        mean = f"{r['mean_ms']:.1f}" if r.get("mean_ms") is not None else "N/A"
        tput = f"{r['throughput_rps']:.2f} req/s"
        print(
            f"{server_name:<12} | {r['concurrency']:>4} | {p50:>9} | {p95:>9} | {p99:>9} | {mean:>10} | {tput:>12}"
        )
    print(sep)
    print()


async def main_async(args):
    with open(args.image, "rb") as f:
        image_bytes = f.read()
    filename = args.image.split("/")[-1]

    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",")]

    print(f"Benchmarking {args.url}")
    print(f"Image: {args.image} ({len(image_bytes)//1024} KB)")
    print(f"Requests per level: {args.requests}")
    print(f"Concurrency levels: {concurrency_levels}")

    # Wait for server to be ready
    print("Waiting for server …", end="", flush=True)
    async with httpx.AsyncClient() as client:
        for _ in range(30):
            try:
                # Try health endpoint first, then just a quick predict
                base = args.url.rsplit("/", 1)[0]
                resp = await client.get(f"{base}/health", timeout=2.0)
                if resp.status_code == 200:
                    print(" ready!")
                    break
            except Exception:
                pass
            print(".", end="", flush=True)
            await asyncio.sleep(1)
        else:
            print("\nServer not responding after 30s, aborting.")
            sys.exit(1)

    # Warm-up: send a few requests before measuring
    print("Warming up (3 requests) …")
    async with httpx.AsyncClient() as client:
        for _ in range(3):
            await send_request(client, args.url, image_bytes, filename)

    results = []
    for conc in concurrency_levels:
        print(f"Running concurrency={conc}, requests={args.requests} …")
        r = await run_concurrency_level(
            args.url, image_bytes, filename, conc, args.requests
        )
        results.append(r)
        print(
            f"  p50={r['p50_ms']}ms  p95={r['p95_ms']}ms  throughput={r['throughput_rps']} req/s"
        )

    server_name = args.name or args.url.split(":")[2].split("/")[0]
    print_table(results, server_name)

    if args.output:
        output_data = {"server": server_name, "url": args.url, "results": results}
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO serving endpoint")
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Prediction endpoint URL (e.g. http://localhost:8001/predict)",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument(
        "--requests",
        type=int,
        default=30,
        help="Number of requests per concurrency level (default: 30)",
    )
    parser.add_argument(
        "--concurrency",
        type=str,
        default="1,2,4,8",
        help="Comma-separated concurrency levels (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file for results"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Server name for output table"
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
