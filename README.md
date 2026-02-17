<p align="center">
  <h1 align="center">DenseDict: High-Density Elastic Hashing for Python</h1>
  <p align="center">
    A memory-efficient <code>dict</code> replacement using the Elastic Hashing algorithm
    <br />
    <a href="https://arxiv.org/abs/2501.02305"><strong>Read the paper »</strong></a>
    ·
    <a href="#benchmarks"><strong>See benchmarks »</strong></a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white" alt="Python 3.7+" />
  <img src="https://img.shields.io/badge/C11-Extension-green?logo=c&logoColor=white" alt="C11" />
  <img src="https://img.shields.io/badge/arXiv-2501.02305-b31b1b?logo=arxiv&logoColor=white" alt="arXiv" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" />
  <img src="https://img.shields.io/badge/Memory%20Saved-20--42%25-brightgreen" alt="Memory" />
</p>

---

## The Challenge

CPython's built-in `dict` uses open addressing with **power-of-two resizing**. When the table reaches ~66% occupancy, Python **doubles** the entire table — allocating 2× memory and rehashing every key:

```
Before resize: 666K items in 1M slots  → 66% load  ✓
After resize:  666K items in 2M slots  → 33% load  ✗ (67% wasted)
```

This creates two problems:

1. **Memory waste** — the average load factor hovers around 33–50%, meaning up to **half the allocated memory is empty slots**
2. **Latency spikes** — each resize is O(n), causing unpredictable pauses in latency-sensitive applications

For applications holding millions of entries (caches, feature stores, embeddings), this overhead is substantial.

## The Solution

**DenseDict** implements the [Elastic Hashing](https://arxiv.org/abs/2501.02305) algorithm (Farach-Colton, Krapivin, Kuszmaul, 2025), which achieves **near-optimal memory usage without ever resizing or reordering elements**.

Instead of one monolithic table, DenseDict uses a **funnel of geometrically-shrinking levels**:

```
Level 0:  ████████████████████████████████  524,288 slots (50%)
Level 1:  ████████████████                  262,144 slots (25%)
Level 2:  ████████                          131,072 slots (12.5%)
Level 3:  ████                               65,536 slots (6.25%)
Level 4:  ██                                 32,768 slots (allocated lazily)
  ...
Level N:  █                                  (minimum 64 slots)
```

**Key properties:**

| Feature | DenseDict | CPython dict |
|---|---|---|
| Resize | **Never** | Doubles at ~66% |
| Move keys | **Never** | Every resize |
| Load factor | **88–95%** | ~33–66% |
| Level allocation | **Lazy** (on demand) | Upfront |
| Probe budget | **Dynamic** ⌈log₂(1/ε)⌉ | Fixed |

## Benchmarks

All benchmarks run on Apple M-series, Python 3.11, 900K items:

### Memory Efficiency

| Key Type | DenseDict | CPython dict | Savings |
|---|:---:|:---:|:---:|
| **String keys** (900K) | 23.25 MB | 29.33 MB | **20.7%** |
| **Integer keys** (900K) | 23.25 MB | 40.00 MB | **42.2%** |

### Performance

| Metric | DenseDict | CPython dict |
|---|:---:|:---:|
| Load factor | **88.6%** | ~70% |
| Avg levels/lookup | **1.65** | — |
| Insert degradation | **+5.6%** | +20.1% |
| Lookup speed (strings) | 3.8M/s | 4.3M/s |
| Lookup speed (integers) | 6.2M/s | 12.4M/s |

> **Key insight:** DenseDict trades ~1% lookup speed (for string keys) for **20%+ memory savings**. Integer keys are faster in CPython due to decades of specialised optimisation, but DenseDict saves **42% memory**.

### Probe Statistics

The paper claims avg levels visited per lookup should be ~1.x. Our implementation achieves:

- **String keys:** 1.65 levels / lookup
- **Integer keys:** 1.65 levels / lookup
- **At 98% load:** 1.41 levels / lookup, 12M lookups/sec

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    DenseDict Object                  │
│  seed: uint64  |  base_probe_limit: int             │
│  total_capacity: size_t  |  total_used: size_t      │
├─────────────────────────────────────────────────────┤
│  Level 0  [████████████████████████] 95% full       │
│  Level 1  [████████████████████████] 95% full       │
│  Level 2  [██████████              ] 43% full       │
│  Level 3  [                        ] (not allocated) │
│  ...                                                 │
│  Level 19 [                        ] (not allocated) │
├─────────────────────────────────────────────────────┤
│  Each entry: | key* (8B) | value* (8B) | hash (8B) │
│              24 bytes, cache-line aligned            │
└─────────────────────────────────────────────────────┘
```

### Design Decisions

- **Lazy level allocation** — Level 0 is created at init; subsequent levels only when overflow occurs. For 900K items, only 5/20 levels are allocated.
- **Compact 24-byte entries** — No `occupied` flag; `NULL` = empty, `(PyObject*)1` = tombstone.
- **Cache-line alignment** — All level arrays are 64-byte aligned via `posix_memalign` / `_aligned_malloc`.
- **Dynamic probe budget** — `f(ε) = ⌈log₂(1/ε)⌉` adapts automatically as levels fill up.
- **Independent hash mixing** — Each (level, probe) pair produces an independent hash index, preventing collision chain repetition.

## Installation

### From source

```bash
git clone https://github.com/bueno1222/python-hashing.git
cd python-hashing
pip install -e .
```

### Manual build

```bash
python setup.py build_ext --inplace
```

### Requirements

- Python ≥ 3.7
- C11 compiler (GCC, Clang, MSVC)
- No external dependencies

## Usage

```python
import densedict

# Create with capacity and probe limit
dd = densedict.DenseDict(capacity=1_000_000, probe_limit=8)

# Standard dict operations
dd["key"] = "value"
print(dd["key"])       # → "value"
del dd["key"]
print(len(dd))         # → 0

# Introspection
dd.memory_usage()      # → bytes consumed by hash table overhead
dd.load_factor()       # → used / allocated_capacity
dd.allocated_levels()  # → number of lazily-allocated levels
dd.level_stats()       # → per-level diagnostics

# Probe statistics (for benchmarking)
dd.reset_probe_stats()
_ = dd["key"]          # trigger lookups
dd.average_probes()    # → {"avg_levels": 1.2, "avg_probes": 3.5, ...}
```

## Interactive Notebook

Try the full benchmark in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16rb4VKajCH4unfycPYRC2RdZoDBN9FmW)

The notebook `DenseDict_Elastic_Hashing_PoC.ipynb` includes:
- Memory vs items graphs (DenseDict vs CPython)
- Insert-time spike comparison
- Integrity verification
- Probe statistics validation
- Integer-key stress test

## Future Roadmap

This is a **proof-of-concept** targeting a potential CPython enhancement (PEP):

- [ ] **Iterator protocol** — `for k in dd`, `dd.keys()`, `dd.values()`, `dd.items()`
- [ ] **`__contains__`** — `"key" in dd`
- [ ] **Compact mode** — separate keys/values array à la CPython 3.6+ compact dict
- [ ] **Tombstone bitmap** — replace inline tombstones with a separate bitfield
- [ ] **Thread safety** — per-level locks for concurrent access
- [ ] **PEP proposal** — formal proposal for CPython integration
- [ ] **Reach 40% savings** — more aggressive lazy allocation (98% threshold)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Farach-Colton, M., Krapivin, M., & Kuszmaul, W. (2025). *Optimal Bounds for Open Addressing Without Reordering*. [arXiv:2501.02305](https://arxiv.org/abs/2501.02305)
2. CPython [dictobject.c](https://github.com/python/cpython/blob/main/Objects/dictobject.c) — reference implementation
3. Knuth, D. E. (1998). *The Art of Computer Programming, Volume 3: Sorting and Searching*. §6.4 Hashing.

---

<p align="center">
  <sub>Built with 🧪 science and ☕ coffee</sub>
</p>
