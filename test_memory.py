#!/usr/bin/env python3
"""
DenseDict v2 — Extended Memory & Performance Benchmark
======================================================

Tests:
  1. Memory efficiency vs CPython dict (string keys)
  2. Integrity verification (sanity check correctness at 98% load)
  3. Probe statistics (average levels visited per lookup)
  4. Integer-key stress test (hash-collision–hostile workload)
  5. Insert-time degradation per 100K batch
  6. Cold-cache vs warm-cache lookup
"""

import sys
import time
import tracemalloc
import gc
import random
import densedict


# ── Helpers ────────────────────────────────────────────────────────

def fmt(nbytes):
    """Format bytes as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024.0:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2f} TB"


def section(title, char="═", width=70):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def subsection(title, char="─", width=70):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


# ── 1. Insert Progression ─────────────────────────────────────────

def benchmark_insert_progression(target, keys, name):
    """Insert keys in 100K batches, reporting rate per batch."""
    subsection(f"INSERT PROGRESSION — {name}")

    batch = 100_000
    times = []
    for b, start in enumerate(range(0, len(keys), batch)):
        end = min(start + batch, len(keys))
        t0 = time.perf_counter()
        for i in range(start, end):
            target[keys[i]] = i
        dt = time.perf_counter() - t0
        times.append(dt)
        rate = (end - start) / dt
        print(f"  Batch {b}: {start:>7,}–{end:>7,}  "
              f"{dt:.3f}s  ({rate:,.0f} items/s)")

    if len(times) > 1:
        pct = (times[-1] / times[0] - 1) * 100
        print(f"\n  Degradation: {pct:+.1f}% "
              f"(first {times[0]:.3f}s → last {times[-1]:.3f}s)")
    return target


# ── 2. Integrity Verification ─────────────────────────────────────

def verify_integrity(dd, keys, name="DenseDict"):
    """Read back every key and assert correctness."""
    subsection(f"INTEGRITY CHECK — {name} ({len(keys):,} keys)")

    errors = 0
    t0 = time.perf_counter()
    for i, k in enumerate(keys):
        val = dd[k]
        if val != i:
            errors += 1
            if errors <= 5:
                print(f"  ✗ dd[{k!r}] = {val!r}, expected {i}")
    dt = time.perf_counter() - t0

    if errors == 0:
        print(f"  ✓ All {len(keys):,} values correct  ({dt:.3f}s)")
    else:
        print(f"  ✗ {errors:,} / {len(keys):,} WRONG!")

    # Spot-check with random sample
    sample = random.sample(range(len(keys)), min(1000, len(keys)))
    for i in sample:
        assert dd[keys[i]] == i, f"random spot-check failed: key={keys[i]}"
    print(f"  ✓ 1,000 random spot-checks passed")

    return errors


# ── 3. Probe Statistics ───────────────────────────────────────────

def measure_probe_stats(dd, keys, label):
    """Run lookups with probe-stat tracking and report averages."""
    dd.reset_probe_stats()

    t0 = time.perf_counter()
    for k in keys:
        _ = dd[k]
    dt = time.perf_counter() - t0

    stats = dd.average_probes()
    rate = len(keys) / dt
    print(f"\n  📊 Probe Stats — {label} ({len(keys):,} lookups)")
    print(f"     Avg levels visited : {stats['avg_levels']:.3f}")
    print(f"     Avg probes tried   : {stats['avg_probes']:.3f}")
    print(f"     Lookup rate        : {rate:,.0f} /s  ({dt:.3f}s)")
    return stats


# ── 4. Cold / Warm Cache ──────────────────────────────────────────

def cold_warm_test(target, keys, n, label):
    """Three runs: first is "cold", best of rest is "warm"."""
    gc.collect()
    time.sleep(0.05)

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for i in range(n):
            _ = target[keys[i % len(keys)]]
        times.append(time.perf_counter() - t0)

    cold = times[0]
    warm = min(times[1:])
    print(f"\n  🧊 Cache Test — {label}")
    print(f"     Cold: {n / cold:,.0f} /s  ({cold:.3f}s)")
    print(f"     Warm: {n / warm:,.0f} /s  ({warm:.3f}s)")
    print(f"     Speedup: {cold / warm:.2f}×")
    return cold, warm


# ══════════════════════════════════════════════════════════════════
# BENCHMARKS
# ══════════════════════════════════════════════════════════════════

def benchmark_densedict_strings():
    """DenseDict with uniform string keys."""
    section("DENSEDICT — STRING KEYS (900K @ 90% load)")

    N = 900_000
    keys = [f"key_{i:08d}" for i in range(N)]

    tracemalloc.start()
    mem0 = tracemalloc.get_traced_memory()[0]

    dd = densedict.DenseDict(capacity=1_000_000, probe_limit=8)
    print(f"  Initial: {dd.allocated_levels()} levels, "
          f"{fmt(dd.memory_usage())}")

    dd = benchmark_insert_progression(dd, keys, "DenseDict str")

    mem1, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n  ✓ {len(dd):,} items | load {dd.load_factor():.1%} | "
          f"{dd.allocated_levels()} levels")
    print(f"    Internal mem : {fmt(dd.memory_usage())}")
    print(f"    Tracked mem  : {fmt(mem1 - mem0)}")

    for s in dd.level_stats()[:5]:
        print(f"    L{s['level']}: {s['used']:,}/{s['capacity']:,} "
              f"({s['load']:.1%})")

    # Integrity
    verify_integrity(dd, keys)

    # Probe stats
    stats = measure_probe_stats(dd, keys, "all 900K string keys")

    # Cache
    cold_warm_test(dd, keys, 1_000_000, "DenseDict strings")

    return {
        "internal_mem": dd.memory_usage(),
        "tracked": mem1 - mem0,
        "peak": peak,
        "load": dd.load_factor(),
        "avg_levels": stats["avg_levels"],
        "avg_probes": stats["avg_probes"],
    }


def benchmark_cpython_strings():
    """CPython dict with the same string keys."""
    section("CPYTHON DICT — STRING KEYS (900K)")

    N = 900_000
    keys = [f"key_{i:08d}" for i in range(N)]

    tracemalloc.start()
    mem0 = tracemalloc.get_traced_memory()[0]

    d = {}
    d = benchmark_insert_progression(d, keys, "CPython str")

    mem1, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    sizeof = sys.getsizeof(d)
    print(f"\n  ✓ {len(d):,} items | sizeof {fmt(sizeof)}")
    print(f"    Tracked mem : {fmt(mem1 - mem0)}")

    # Integrity
    subsection("INTEGRITY CHECK — CPython dict")
    errors = 0
    for i, k in enumerate(keys):
        if d[k] != i:
            errors += 1
    print(f"  ✓ All {N:,} values correct" if errors == 0
          else f"  ✗ {errors} wrong")

    # Cache
    cold_warm_test(d, keys, 1_000_000, "CPython strings")

    return {
        "sizeof": sizeof,
        "tracked": mem1 - mem0,
        "peak": peak,
    }


def benchmark_densedict_integers():
    """DenseDict with integer keys — stress-test for trivial hashes."""
    section("DENSEDICT — INTEGER KEYS (900K)")

    N = 900_000
    keys = list(range(N))

    dd = densedict.DenseDict(capacity=1_000_000, probe_limit=8)
    dd = benchmark_insert_progression(dd, keys, "DenseDict int")

    print(f"\n  ✓ {len(dd):,} items | load {dd.load_factor():.1%} | "
          f"{dd.allocated_levels()} levels")
    print(f"    Internal mem : {fmt(dd.memory_usage())}")

    # Integrity
    verify_integrity(dd, keys, "DenseDict int")

    # Probe stats
    stats = measure_probe_stats(dd, keys, "all 900K integer keys")

    # Cache
    cold_warm_test(dd, keys, 1_000_000, "DenseDict integers")

    return {
        "internal_mem": dd.memory_usage(),
        "load": dd.load_factor(),
        "avg_levels": stats["avg_levels"],
        "avg_probes": stats["avg_probes"],
    }


def benchmark_cpython_integers():
    """CPython dict with integer keys."""
    section("CPYTHON DICT — INTEGER KEYS (900K)")

    N = 900_000
    keys = list(range(N))

    d = {}
    d = benchmark_insert_progression(d, keys, "CPython int")

    sizeof = sys.getsizeof(d)
    print(f"\n  ✓ {len(d):,} items | sizeof {fmt(sizeof)}")

    # Integrity
    subsection("INTEGRITY CHECK — CPython int")
    errors = 0
    for i in keys:
        if d[i] != i:
            errors += 1
    print(f"  ✓ All {N:,} values correct" if errors == 0
          else f"  ✗ {errors} wrong")

    cold_warm_test(d, keys, 1_000_000, "CPython integers")

    return {"sizeof": sizeof}


def benchmark_exhaustion():
    """Fill to 98% and verify correctness + probe behaviour."""
    section("EXHAUSTION TEST — DenseDict @ 98% Load")

    cap = 200_000
    target = int(cap * 0.98)
    keys = [f"key_{i:08d}" for i in range(target)]

    dd = densedict.DenseDict(capacity=cap, probe_limit=8)

    t0 = time.perf_counter()
    for i, k in enumerate(keys):
        dd[k] = i
        if (i + 1) % 20_000 == 0:
            print(f"  {i+1:>7,} items  load={dd.load_factor():.1%}  "
                  f"levels={dd.allocated_levels()}")
    dt = time.perf_counter() - t0

    print(f"\n  ✓ Inserted {target:,} in {dt:.3f}s")
    print(f"    load={dd.load_factor():.1%}  levels={dd.allocated_levels()}  "
          f"mem={fmt(dd.memory_usage())}")

    for s in dd.level_stats():
        print(f"    L{s['level']}: {s['used']:,}/{s['capacity']:,} "
              f"({s['load']:.1%}, ε={s['epsilon']:.3f})")

    # Integrity at 98%
    verify_integrity(dd, keys, "98% load")

    # Probe stats at 98%
    measure_probe_stats(dd, keys, "98% load string keys")


# ── Comparison ─────────────────────────────────────────────────────

def compare(dd_str, cpython_str, dd_int, cpython_int):
    section("FINAL COMPARISON", "█")

    dd_mem = dd_str["internal_mem"]
    cp_mem = cpython_str["sizeof"]

    print(f"\n  ┌──────────────────────────┬──────────────┬──────────────┐")
    print(f"  │ Metric                   │ DenseDict    │ CPython dict │")
    print(f"  ├──────────────────────────┼──────────────┼──────────────┤")
    print(f"  │ String-key memory        │ {fmt(dd_mem):>12s} │ {fmt(cp_mem):>12s} │")

    if dd_mem < cp_mem:
        saved = cp_mem - dd_mem
        pct = saved / cp_mem * 100
        print(f"  │ Memory saved             │       —      │ {pct:.1f}%        │")
    else:
        extra = dd_mem - cp_mem
        pct = extra / cp_mem * 100
        print(f"  │ Memory overhead           │ +{pct:.1f}%       │       —      │")

    print(f"  │ Load factor (strings)    │ {dd_str['load']:.1%}        │ ~70%         │")
    print(f"  │ Avg levels / lookup (str)│ {dd_str['avg_levels']:.3f}        │      —       │")
    print(f"  │ Avg probes / lookup (str)│ {dd_str['avg_probes']:.3f}        │      —       │")

    dd_imem = dd_int["internal_mem"]
    cp_imem = cpython_int["sizeof"]
    print(f"  │                          │              │              │")
    print(f"  │ Integer-key memory       │ {fmt(dd_imem):>12s} │ {fmt(cp_imem):>12s} │")
    print(f"  │ Avg levels / lookup (int)│ {dd_int['avg_levels']:.3f}        │      —       │")
    print(f"  │ Avg probes / lookup (int)│ {dd_int['avg_probes']:.3f}        │      —       │")
    print(f"  └──────────────────────────┴──────────────┴──────────────┘")

    # Verdict
    print()
    if dd_mem < cp_mem:
        pct = (cp_mem - dd_mem) / cp_mem * 100
        print(f"  ✅ DenseDict uses {pct:.1f}% less memory (strings)")
    else:
        print(f"  ⚠️  DenseDict uses MORE memory (strings)")

    print(f"  📐 Average {dd_str['avg_levels']:.2f} levels / lookup "
          f"(paper claims ~1.x)")
    print(f"  🔢 Integer keys: avg {dd_int['avg_levels']:.2f} levels "
          f"(hash-collision stress)")


# ── Main ───────────────────────────────────────────────────────────

def main():
    print("╔" + "═" * 68 + "╗")
    print("║   DenseDict v2 — Extended Benchmark (Probes + Integrity + Int)   ║")
    print("╚" + "═" * 68 + "╝")

    dd_str  = benchmark_densedict_strings()
    cp_str  = benchmark_cpython_strings()
    dd_int  = benchmark_densedict_integers()
    cp_int  = benchmark_cpython_integers()

    compare(dd_str, cp_str, dd_int, cp_int)

    benchmark_exhaustion()

    section("BENCHMARK COMPLETE", "█")


if __name__ == "__main__":
    main()
