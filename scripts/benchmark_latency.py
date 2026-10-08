"""
scripts/benchmark_latency.py
============================
Benchmarks real-time inference latency across 500 consecutive transactions.
Measures p50, p90, p95, and p99 latency against banking SLAs (<10ms).
"""

import time
import numpy as np
import os
import sys

# Ensure src is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from multi_agent_gateway import MultiAgentSystem

def run_latency_benchmark(num_samples: int = 500):
    print(f"🚀 Initializing MultiAgentSystem for {num_samples} transaction latency test...")
    mas = MultiAgentSystem()

    latencies = []
    sample_txn = {
        "TransactionID": "TXN_BENCHMARK_8801",
        "TransactionAmt": 150.0,
        "card1": 13500.0,
        "card2": 250.0,
        "card4": "visa",
        "card6": "debit",
        "addr1": 315.0,
        "P_emaildomain": "gmail.com",
        "C1": 1.0,
        "C2": 1.0,
        "DeviceInfo": "iOS Device"
    }

    # Warmup
    for _ in range(20):
        mas.evaluate_transaction(sample_txn)

    # Actual timing
    for i in range(num_samples):
        start = time.perf_counter()
        mas.evaluate_transaction(sample_txn)
        duration_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(duration_ms)

    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print("\n" + "=" * 50)
    print(" 📊 LATENCY BENCHMARK RESULTS (500 Transactions)")
    print("=" * 50)
    print(f"  • Mean Latency:           {mean_lat:.3f} ms")
    print(f"  • Median (p50):           {p50:.3f} ms")
    print(f"  • 90th Percentile (p90):  {p90:.3f} ms")
    print(f"  • 95th Percentile (p95):  {p95:.3f} ms")
    print(f"  • 99th Percentile (p99):  {p99:.3f} ms")
    print(f"  • Sub-10ms SLA Passed:    {p95 < 10.0} (Target: <10ms)")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_latency_benchmark()
