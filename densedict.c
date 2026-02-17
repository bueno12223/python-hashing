/*
 * DenseDict — A Memory-Efficient Hash Table for Python.
 *
 * This module implements "DenseDict", a Python mapping type built on top
 * of the Elastic Hashing strategy described in:
 *
 *     "Optimal Bounds for Open Addressing Without Reordering"
 *     Farach-Colton, Krapivin, Kuszmaul.  arXiv:2501.02305 (2025).
 *
 * --- Algorithm Overview ---
 *
 * The table is split into geometrically-shrinking *levels*:
 *
 *     Level 0 :  total_capacity / 2    slots
 *     Level 1 :  total_capacity / 4    slots
 *     Level 2 :  total_capacity / 8    slots
 *        …             …
 *     Level k :  max(64, total_capacity >> (k+1))  slots
 *
 * All level sizes are powers of two so that the modulo operation reduces
 * to a single bitwise AND.
 *
 * Insertion tries Level 0 first.  For each level the algorithm performs
 * at most `probe_limit` probes (see below).  If no empty or reusable
 * slot is found within the probe budget, the key "falls" to the next
 * level, which is allocated lazily on first use.  Once a key is placed,
 * **it is never moved** — the defining property of Elastic Hashing.
 *
 * --- Dynamic Probe Limit (paper §3) ---
 *
 * Let ε = (capacity − used) / capacity  be the fraction of free slots
 * in a given level.  The probe budget for that level is
 *
 *     f(ε) = ⌈ log₂(1/ε) ⌉
 *
 * clamped to [DENSEDICT_MIN_PROBE_LIMIT, DENSEDICT_MAX_PROBE_LIMIT].
 * A level that is already ≥ 95 % full is skipped entirely, because
 * linear probing in a nearly-full region would degenerate.
 *
 * --- Hash Function ---
 *
 * Each (level, probe_index) pair uses an independent hash derived from
 * Python's Py_hash_t by mixing in the level number and probe index with
 * multiplicative constants drawn from the golden-ratio and Knuth
 * families.  This prevents collision patterns from repeating across
 * levels — the bidirectional mapping φ(i, j) described in §4 of the
 * paper.
 *
 * --- Memory Model ---
 *
 * Levels are allocated lazily with cache-line alignment (64 bytes).
 * Entries are a compact 24-byte struct (pointer, pointer, hash) with
 * key==NULL meaning "empty" and key==(PyObject*)1 meaning "tombstone".
 *
 * PEP 7 — C style as mandated by CPython:
 *   - 4-space indentation                 (*)
 *   - Braces on their own line for funcs  (K&R for control flow)
 *   - Prefix `densedict_` on all symbols
 *   - `static` on everything except PyInit
 *
 * (*) In this file we use the more compact variant accepted by modern
 *     CPython extensions: 4-space indentation, braces on same line for
 *     control flow, next line for functions.
 */

/* ── Includes ──────────────────────────────────────────────────────── */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ── Cross-platform aligned allocation ─────────────────────────────
 *
 * posix_memalign / _aligned_malloc differ in API and free semantics:
 *   POSIX  — posix_memalign + free()
 *   Win32  — _aligned_malloc + _aligned_free()
 */
#ifdef _WIN32
#  include <malloc.h>
#  define DENSEDICT_ALIGNED_ALLOC(ptr, align, size)  \
       (*(ptr) = _aligned_malloc((size), (align)),   \
        (*(ptr) == NULL) ? -1 : 0)
#  define DENSEDICT_ALIGNED_FREE(ptr)  _aligned_free(ptr)
#else
#  include <stdlib.h>
#  define DENSEDICT_ALIGNED_ALLOC(ptr, align, size)  \
       posix_memalign((void **)(ptr), (align), (size))
#  define DENSEDICT_ALIGNED_FREE(ptr)  free(ptr)
#endif

/* ── Compile-time constants ────────────────────────────────────────
 *
 * DENSEDICT_MAX_LEVELS — hard cap on the number of funnel levels.
 *     20 levels can accommodate up to 2^21 ≈ 2 M slots in total.
 *
 * DENSEDICT_DEFAULT_CAPACITY — used when the user does not provide an
 *     explicit `capacity` argument.
 *
 * DENSEDICT_CACHE_LINE — alignment for level arrays.
 *     64 bytes matches both x86-64 and Apple Silicon.
 *
 * DENSEDICT_MIN_PROBE_LIMIT / DENSEDICT_MAX_PROBE_LIMIT — bounds for
 *     the dynamic probe budget f(ε) = ⌈log₂(1/ε)⌉.
 */
#define DENSEDICT_MAX_LEVELS         20
#define DENSEDICT_DEFAULT_CAPACITY   (1 << 20)  /* 1 048 576 */
#define DENSEDICT_CACHE_LINE         64
#define DENSEDICT_MIN_PROBE_LIMIT    4
#define DENSEDICT_MAX_PROBE_LIMIT    16
#define DENSEDICT_DEFAULT_BASE_PL    8

/* Sentinel pointer used as a tombstone marker for deleted slots.
 * The address 0x1 is never a valid PyObject pointer. */
#define DENSEDICT_TOMBSTONE  ((PyObject *)1)

/* ── Forward declarations ──────────────────────────────────────────
 *
 * The PyTypeObject must be forward-declared so that the tp_new
 * function signature can reference it.
 */
static PyTypeObject DenseDict_Type;

/* ══════════════════════════════════════════════════════════════════
 * §1  Data Structures
 * ══════════════════════════════════════════════════════════════════ */

/*
 * Compact hash-table entry — 24 bytes on 64-bit platforms.
 *
 * Slot states:
 *     key == NULL                → empty (never used)
 *     key == DENSEDICT_TOMBSTONE → deleted (skip during lookup)
 *     otherwise                 → occupied
 *
 * No separate `occupied` flag is needed, saving 8 bytes per entry
 * (4 bytes for the int + 4 bytes struct padding to keep alignment).
 */
typedef struct {
    PyObject *key;      /* 8 bytes  — borrowed ref while alive        */
    PyObject *value;    /* 8 bytes  — borrowed ref while alive        */
    Py_hash_t hash;     /* 8 bytes  — cached hash of key              */
} DenseDictEntry;

/*
 * A single level inside the multi-level funnel.
 *
 * `capacity` is always a power of two so that `index & mask` replaces
 * the expensive modulo operation.
 *
 * `entries` is NULL until the level is first needed (lazy allocation).
 * The `allocated` flag is redundant with `entries != NULL` but avoids
 * a subtle race during initialisation where capacity/mask are set
 * before the allocation succeeds.
 */
typedef struct {
    DenseDictEntry *entries;   /* cache-line-aligned array            */
    size_t          capacity;  /* always a power of two               */
    size_t          used;      /* live entries (excludes tombstones)   */
    size_t          mask;      /* capacity − 1, for bitwise AND       */
    int             allocated; /* 1 iff entries != NULL                */
} DenseDictLevel;

/*
 * The top-level DenseDict Python object.
 *
 * `levels` is a fixed-size inline array to avoid an extra heap
 * allocation.  Only the first `num_levels` entries are meaningful,
 * and even among those only the ones with `allocated == 1` consume
 * memory for their entry arrays.
 *
 * `seed` is a fixed randomisation constant mixed into every hash
 * computation (see densedict_hash_mix).
 *
 * `base_probe_limit` is the user-configurable floor for the dynamic
 * probe budget.  The effective limit is
 *     max(base_probe_limit, ⌈log₂(1/ε)⌉).
 *
 * Probe statistics (`stat_*` fields) are accumulated during lookups
 * so that the caller can compute the average number of levels visited
 * and probes tried per search — the central claim of the Elastic
 * Hashing paper is that this stays close to 1.
 */
typedef struct {
    PyObject_HEAD
    DenseDictLevel  levels[DENSEDICT_MAX_LEVELS];
    int             num_levels;
    size_t          total_capacity;
    size_t          total_used;
    uint64_t        seed;
    int             base_probe_limit;

    /* Probe statistics — accumulated by densedict_lookup_internal. */
    uint64_t        stat_lookups;        /* total lookup calls        */
    uint64_t        stat_levels_visited; /* sum of levels touched     */
    uint64_t        stat_probes_tried;   /* sum of individual probes  */
} DenseDictObject;


/* ══════════════════════════════════════════════════════════════════
 * §2  Internal Helpers
 * ══════════════════════════════════════════════════════════════════ */

/* ── densedict_hash_mix ────────────────────────────────────────────
 *
 * Derive a slot index from (base_hash, seed, level, probe).
 *
 * Mixing the *level* prevents a key that collides in Level 0 from
 * colliding in the same relative position in Level 1 — this is the
 * bidirectional mapping φ(i, j) described in §4 of arXiv:2501.02305.
 *
 * Mixing the *probe number* gives each retry within the same level a
 * fresh starting position, approximating independent random probes
 * rather than simple linear probing and thus approaching the
 * theoretical O(1) amortised bound.
 *
 * The multiplicative constants are
 *     0x9e3779b97f4a7c15  — 2^64 / φ  (golden ratio)
 *     0xc6a4a7935bd1e995  — from MurmurHash64A by Austin Appleby
 * and the 64-bit finaliser is MurmurHash3's fmix64.
 */
static inline uint64_t
densedict_hash_mix(Py_hash_t base_hash,
                   uint64_t  seed,
                   int       level,
                   int       probe)
{
    uint64_t h = (uint64_t)base_hash;

    h ^= seed;
    h ^= (uint64_t)level * 0x9e3779b97f4a7c15ULL;
    h ^= (uint64_t)probe * 0xc6a4a7935bd1e995ULL;

    /* fmix64 — MurmurHash3 finaliser */
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;

    return h;
}

/* ── densedict_level_capacity ──────────────────────────────────────
 *
 * Return the capacity (always a power of two) for level `level`
 * given a total table capacity of `total_cap`.
 *
 *     Level 0 → total_cap / 2
 *     Level k → total_cap / 2^(k+1),  min 64
 */
static size_t
densedict_level_capacity(size_t total_cap, int level)
{
    size_t cap;

    if (level == 0) {
        return total_cap >> 1;
    }
    cap = total_cap >> (level + 1);
    if (cap < 64) {
        cap = 64;
    }
    /* Round down to the nearest power of two. */
    cap = 1ULL << (63 - __builtin_clzll(cap));
    return cap;
}

/* ── densedict_ensure_level ────────────────────────────────────────
 *
 * Lazily allocate the entry array for `level`.
 *
 * Returns 0 on success, −1 with a Python exception set on failure.
 * Calling this on an already-allocated level is a harmless no-op.
 */
static int
densedict_ensure_level(DenseDictObject *self, int level)
{
    DenseDictLevel *lv;
    size_t cap, alloc_size;
    int ret;

    if (level < 0 || level >= self->num_levels) {
        PyErr_Format(PyExc_SystemError,
                     "DenseDict internal error: level %d out of range "
                     "[0, %d)", level, self->num_levels);
        return -1;
    }

    lv = &self->levels[level];
    if (lv->allocated) {
        return 0;   /* already live */
    }

    cap = densedict_level_capacity(self->total_capacity, level);
    alloc_size = cap * sizeof(DenseDictEntry);

    ret = DENSEDICT_ALIGNED_ALLOC(&lv->entries, DENSEDICT_CACHE_LINE,
                                  alloc_size);
    if (ret != 0 || lv->entries == NULL) {
        lv->entries = NULL;
        PyErr_NoMemory();
        return -1;
    }

    memset(lv->entries, 0, alloc_size);
    lv->capacity  = cap;
    lv->mask      = cap - 1;
    lv->used      = 0;
    lv->allocated = 1;
    return 0;
}

/* ── densedict_free_level ──────────────────────────────────────────
 *
 * Release all Python references held by entries in `lv` and free the
 * aligned entry array.  Safe to call on an unallocated level. */
static void
densedict_free_level(DenseDictLevel *lv)
{
    size_t i;
    DenseDictEntry *entry;

    if (!lv->allocated || lv->entries == NULL) {
        return;
    }

    for (i = 0; i < lv->capacity; i++) {
        entry = &lv->entries[i];
        if (entry->key != NULL && entry->key != DENSEDICT_TOMBSTONE) {
            Py_DECREF(entry->key);
            Py_DECREF(entry->value);
        }
    }

    DENSEDICT_ALIGNED_FREE(lv->entries);
    lv->entries   = NULL;
    lv->allocated = 0;
    lv->used      = 0;
}

/* ── densedict_probe_limit ─────────────────────────────────────────
 *
 * Compute the dynamic probe budget for a level whose free fraction is
 * ε (epsilon).  From arXiv:2501.02305, §3:
 *
 *     f(ε)  =  ⌈ log₂(1/ε) ⌉
 *
 * The result is clamped to [DENSEDICT_MIN_PROBE_LIMIT,
 * DENSEDICT_MAX_PROBE_LIMIT] and then max'd with `base_limit` (the
 * user-configurable floor).
 */
static inline int
densedict_probe_limit(double epsilon, int base_limit)
{
    int dynamic;

    if (epsilon <= 0.001) {
        epsilon = 0.001;   /* guard against log(0) / log(∞) */
    }
    dynamic = (int)ceil(log2(1.0 / epsilon));

    if (dynamic < DENSEDICT_MIN_PROBE_LIMIT) {
        dynamic = DENSEDICT_MIN_PROBE_LIMIT;
    }
    if (dynamic > DENSEDICT_MAX_PROBE_LIMIT) {
        dynamic = DENSEDICT_MAX_PROBE_LIMIT;
    }
    return (dynamic > base_limit) ? dynamic : base_limit;
}

/* ── densedict_round_up_pow2 ───────────────────────────────────────
 *
 * Round `n` up to the next power of two.  Used to sanitise the
 * user-supplied capacity.
 */
static inline size_t
densedict_round_up_pow2(size_t n)
{
    if (n == 0) {
        return 1;
    }
    /* Portable: works for n up to 2^63. */
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

/* ══════════════════════════════════════════════════════════════════
 * §3  Core Operations  (insert / lookup / delete)
 * ══════════════════════════════════════════════════════════════════ */

/* ── densedict_insert_internal ─────────────────────────────────────
 *
 * Walk the levels top-down (0 → num_levels−1).  For each level:
 *
 *   1.  Lazily allocate if not yet allocated.
 *   2.  Skip if ε < 0.05 (level ≥ 95 % full) AND another level exists.
 *   3.  Compute probe budget f(ε).
 *   4.  Probe up to f(ε) slots in the level.
 *       – Empty slot ⇒ insert.
 *       – Tombstone ⇒ insert (reuse deleted slot).
 *       – Same key  ⇒ update value (no ref-count change for key).
 *   5.  If budget exhausted, fall to next level.
 *
 * Returns 0 on success, −1 with a Python exception on failure.
 */
static int
densedict_insert_internal(DenseDictObject *self,
                          PyObject        *key,
                          PyObject        *value,
                          Py_hash_t        hash)
{
    int level, probe, plimit, cmp;
    double epsilon;
    uint64_t h;
    size_t idx;
    DenseDictLevel *lv;
    DenseDictEntry *entry;

    for (level = 0; level < self->num_levels; level++) {
        /* §3-1  Ensure this level exists. */
        if (densedict_ensure_level(self, level) < 0) {
            return -1;
        }
        lv = &self->levels[level];

        /* §3-2  Skip nearly-full levels. */
        epsilon = (double)(lv->capacity - lv->used)
                  / (double)lv->capacity;
        if (epsilon < 0.05 && (level + 1) < self->num_levels) {
            continue;
        }

        /* §3-3  Dynamic probe budget. */
        plimit = densedict_probe_limit(epsilon, self->base_probe_limit);

        /* §3-4  Probe loop. */
        for (probe = 0; probe < plimit; probe++) {
            h   = densedict_hash_mix(hash, self->seed, level, probe);
            idx = h & lv->mask;
            entry = &lv->entries[idx];

            /* Empty slot — insert. */
            if (entry->key == NULL) {
                Py_INCREF(key);
                Py_INCREF(value);
                entry->key   = key;
                entry->value = value;
                entry->hash  = hash;
                lv->used++;
                self->total_used++;
                return 0;
            }

            /* Tombstone — reuse. */
            if (entry->key == DENSEDICT_TOMBSTONE) {
                Py_INCREF(key);
                Py_INCREF(value);
                entry->key   = key;
                entry->value = value;
                entry->hash  = hash;
                lv->used++;
                self->total_used++;
                return 0;
            }

            /* Existing key — update value. */
            if (entry->hash == hash) {
                cmp = PyObject_RichCompareBool(entry->key, key, Py_EQ);
                if (cmp < 0) {
                    return -1;  /* comparison raised */
                }
                if (cmp == 1) {
                    Py_INCREF(value);
                    Py_SETREF(entry->value, value);
                    return 0;
                }
            }
            /* Collision — try next probe position. */
        }

        /* §3-5  Probe budget exhausted; allocate next level. */
        if ((level + 1) < self->num_levels) {
            if (densedict_ensure_level(self, level + 1) < 0) {
                return -1;
            }
        }
    }

    /* All levels full — this should rarely happen at < 95 % load. */
    PyErr_SetString(PyExc_RuntimeError,
                    "DenseDict: all levels exhausted — table is full");
    return -1;
}

/* ── densedict_lookup_internal ─────────────────────────────────────
 *
 * Search all *allocated* levels for `key`.  We use a slightly
 * extended probe budget (2 × DENSEDICT_MAX_PROBE_LIMIT) because a
 * key may have been inserted while the level's probe limit was
 * higher (i.e. the level was emptier at insertion time).
 *
 * This function also increments the probe-statistics counters so
 * that average_probes() can report the mean levels-per-lookup and
 * probes-per-lookup after a batch of operations.
 *
 * Returns a **new reference** on success, or NULL with KeyError set.
 */
static PyObject *
densedict_lookup_internal(DenseDictObject *self,
                          PyObject        *key,
                          Py_hash_t        hash)
{
    int level, probe, cmp;
    int levels_this_lookup = 0;
    int probes_this_lookup = 0;
    uint64_t h;
    size_t idx;
    DenseDictLevel *lv;
    DenseDictEntry *entry;

    self->stat_lookups++;

    for (level = 0; level < self->num_levels; level++) {
        lv = &self->levels[level];
        if (!lv->allocated) {
            continue;
        }

        levels_this_lookup++;

        for (probe = 0; probe < DENSEDICT_MAX_PROBE_LIMIT * 2; probe++) {
            probes_this_lookup++;
            h   = densedict_hash_mix(hash, self->seed, level, probe);
            idx = h & lv->mask;
            entry = &lv->entries[idx];

            if (entry->key == NULL) {
                break;  /* empty ⇒ not in this level */
            }
            if (entry->key == DENSEDICT_TOMBSTONE) {
                continue;  /* skip deleted slots */
            }

            if (entry->hash == hash) {
                cmp = PyObject_RichCompareBool(entry->key, key, Py_EQ);
                if (cmp < 0) {
                    return NULL;  /* comparison raised */
                }
                if (cmp == 1) {
                    self->stat_levels_visited += levels_this_lookup;
                    self->stat_probes_tried   += probes_this_lookup;
                    Py_INCREF(entry->value);
                    return entry->value;
                }
            }
        }
    }

    /* Key not found — still record stats for the failed lookup. */
    self->stat_levels_visited += levels_this_lookup;
    self->stat_probes_tried   += probes_this_lookup;

    PyErr_SetObject(PyExc_KeyError, key);
    return NULL;
}

/* ── densedict_delete_internal ─────────────────────────────────────
 *
 * Find and tombstone `key`.  Returns 0 on success, −1 with KeyError.
 */
static int
densedict_delete_internal(DenseDictObject *self,
                          PyObject        *key,
                          Py_hash_t        hash)
{
    int level, probe, cmp;
    uint64_t h;
    size_t idx;
    DenseDictLevel *lv;
    DenseDictEntry *entry;

    for (level = 0; level < self->num_levels; level++) {
        lv = &self->levels[level];
        if (!lv->allocated) {
            continue;
        }

        for (probe = 0; probe < DENSEDICT_MAX_PROBE_LIMIT * 2; probe++) {
            h   = densedict_hash_mix(hash, self->seed, level, probe);
            idx = h & lv->mask;
            entry = &lv->entries[idx];

            if (entry->key == NULL) {
                break;
            }
            if (entry->key == DENSEDICT_TOMBSTONE) {
                continue;
            }

            if (entry->hash == hash) {
                cmp = PyObject_RichCompareBool(entry->key, key, Py_EQ);
                if (cmp < 0) {
                    return -1;  /* comparison raised */
                }
                if (cmp == 1) {
                    Py_DECREF(entry->key);
                    Py_DECREF(entry->value);
                    entry->key   = DENSEDICT_TOMBSTONE;
                    entry->value = NULL;
                    entry->hash  = 0;
                    lv->used--;
                    self->total_used--;
                    return 0;
                }
            }
        }
    }

    PyErr_SetObject(PyExc_KeyError, key);
    return -1;
}


/* ══════════════════════════════════════════════════════════════════
 * §4  Python Type Slots
 * ══════════════════════════════════════════════════════════════════ */

/* ── tp_new ────────────────────────────────────────────────────────
 *
 * DenseDict(capacity=1048576, probe_limit=8)
 *
 * Both arguments are optional keyword arguments.  `capacity` is
 * rounded up to the next power of two (minimum 1024).
 */
static DenseDictObject *
densedict_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    DenseDictObject *self;
    Py_ssize_t capacity_arg = DENSEDICT_DEFAULT_CAPACITY;
    int probe_limit = DENSEDICT_DEFAULT_BASE_PL;
    size_t capacity, total;
    static char *kwlist[] = {"capacity", "probe_limit", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ni", kwlist,
                                     &capacity_arg, &probe_limit)) {
        return NULL;
    }

    self = (DenseDictObject *)type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    }

    /* Sanitise capacity. */
    capacity = (size_t)capacity_arg;
    if (capacity < 1024) {
        capacity = 1024;
    }
    capacity = densedict_round_up_pow2(capacity);

    self->total_capacity   = capacity;
    self->total_used       = 0;
    self->seed             = 0x517cc1b727220a95ULL;
    self->base_probe_limit = probe_limit;

    /* Compute how many levels are needed to cover `capacity`. */
    self->num_levels = 0;
    total = 0;
    while (total < capacity && self->num_levels < DENSEDICT_MAX_LEVELS) {
        total += densedict_level_capacity(capacity, self->num_levels);
        self->num_levels++;
    }

    /* Zero-initialise all level metadata. */
    memset(self->levels, 0, sizeof(self->levels));

    /* Lazy: only allocate Level 0 now. */
    if (densedict_ensure_level(self, 0) < 0) {
        Py_DECREF(self);
        return NULL;
    }

    return self;
}

/* ── tp_dealloc ────────────────────────────────────────────────────
 *
 * Release every allocated level and then free the object itself. */
static void
densedict_dealloc(DenseDictObject *self)
{
    int i;

    for (i = 0; i < self->num_levels; i++) {
        densedict_free_level(&self->levels[i]);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ── mp_length ─────────────────────────────────────────────────────
 *
 * Return the number of live (non-tombstone) entries. */
static Py_ssize_t
densedict_length(DenseDictObject *self)
{
    return (Py_ssize_t)self->total_used;
}

/* ── mp_subscript  (dd[key]) ───────────────────────────────────────*/
static PyObject *
densedict_subscript(DenseDictObject *self, PyObject *key)
{
    Py_hash_t hash = PyObject_Hash(key);

    if (hash == -1) {
        return NULL;
    }
    return densedict_lookup_internal(self, key, hash);
}

/* ── mp_ass_subscript  (dd[key] = val  /  del dd[key]) ─────────── */
static int
densedict_ass_subscript(DenseDictObject *self,
                        PyObject        *key,
                        PyObject        *value)
{
    Py_hash_t hash = PyObject_Hash(key);

    if (hash == -1) {
        return -1;
    }
    if (value == NULL) {
        return densedict_delete_internal(self, key, hash);
    }
    return densedict_insert_internal(self, key, value, hash);
}

/* ── tp_repr ───────────────────────────────────────────────────────*/
static PyObject *
densedict_repr(DenseDictObject *self)
{
    return PyUnicode_FromFormat("<DenseDict: %zd items, %d levels>",
                               self->total_used, self->num_levels);
}


/* ══════════════════════════════════════════════════════════════════
 * §5  Introspection Methods (exposed to Python)
 * ══════════════════════════════════════════════════════════════════ */

/* ── memory_usage() ────────────────────────────────────────────────
 *
 * Return the total number of bytes consumed by the DenseDictObject
 * itself plus all allocated entry arrays.
 *
 * Note: this does *not* include the PyObjects pointed to by entries;
 * it only counts the hash-table overhead. */
static PyObject *
densedict_memory_usage(DenseDictObject *self, PyObject *Py_UNUSED(ignored))
{
    size_t total = sizeof(DenseDictObject);
    int i;

    for (i = 0; i < self->num_levels; i++) {
        if (self->levels[i].allocated) {
            total += sizeof(DenseDictEntry) * self->levels[i].capacity;
        }
    }
    return PyLong_FromSize_t(total);
}

/* ── allocated_levels() ────────────────────────────────────────────
 *
 * Return the number of levels that have been lazily allocated.
 * Levels are only allocated when a lower level overflows its probe
 * budget. */
static PyObject *
densedict_allocated_levels(DenseDictObject *self,
                           PyObject        *Py_UNUSED(ignored))
{
    int count = 0, i;

    for (i = 0; i < self->num_levels; i++) {
        if (self->levels[i].allocated) {
            count++;
        }
    }
    return PyLong_FromLong(count);
}

/* ── load_factor() ─────────────────────────────────────────────────
 *
 * Return total_used / allocated_capacity as a float.
 * Only counts levels that have actually been allocated. */
static PyObject *
densedict_load_factor(DenseDictObject *self,
                      PyObject        *Py_UNUSED(ignored))
{
    size_t alloc_cap = 0;
    int i;

    for (i = 0; i < self->num_levels; i++) {
        if (self->levels[i].allocated) {
            alloc_cap += self->levels[i].capacity;
        }
    }
    if (alloc_cap == 0) {
        return PyFloat_FromDouble(0.0);
    }
    return PyFloat_FromDouble(
        (double)self->total_used / (double)alloc_cap);
}

/* ── level_stats() ─────────────────────────────────────────────────
 *
 * Return a list of dicts, one per allocated level:
 *
 *     [{"level": 0,
 *       "capacity": 524288,
 *       "used": 498074,
 *       "load": 0.95,
 *       "epsilon": 0.05,
 *       "probe_limit": 8}, ...]
 *
 * Useful for understanding how the funnel distributes keys and how
 * the dynamic probe limit adapts to each level's fill ratio. */
static PyObject *
densedict_level_stats(DenseDictObject *self,
                      PyObject        *Py_UNUSED(ignored))
{
    PyObject *list, *stats;
    DenseDictLevel *lv;
    double load, epsilon;
    int plimit, i;

    list = PyList_New(0);
    if (list == NULL) {
        return NULL;
    }

    for (i = 0; i < self->num_levels; i++) {
        lv = &self->levels[i];
        if (!lv->allocated) {
            continue;
        }

        load    = (double)lv->used / (double)lv->capacity;
        epsilon = 1.0 - load;
        plimit  = densedict_probe_limit(epsilon, self->base_probe_limit);

        stats = Py_BuildValue("{s:i, s:K, s:K, s:d, s:d, s:i}",
                              "level",       i,
                              "capacity",    (unsigned long long)lv->capacity,
                              "used",        (unsigned long long)lv->used,
                              "load",        load,
                              "epsilon",     epsilon,
                              "probe_limit", plimit);
        if (stats == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        if (PyList_Append(list, stats) < 0) {
            Py_DECREF(stats);
            Py_DECREF(list);
            return NULL;
        }
        Py_DECREF(stats);
    }

    return list;
}

/* ── average_probes() ──────────────────────────────────────────────
 *
 * Return a dict with accumulated lookup probe statistics:
 *
 *     {"total_lookups":   <int>,
 *      "avg_levels":      <float>,   ← levels visited per lookup
 *      "avg_probes":      <float>,   ← individual probes per lookup
 *      "total_levels":    <int>,
 *      "total_probes":    <int>}
 *
 * The paper's claim is that avg_levels should stay close to 1.x.
 * Call reset_probe_stats() before a measurement batch.
 */
static PyObject *
densedict_average_probes(DenseDictObject *self,
                         PyObject        *Py_UNUSED(ignored))
{
    double avg_levels = 0.0;
    double avg_probes = 0.0;

    if (self->stat_lookups > 0) {
        avg_levels = (double)self->stat_levels_visited
                     / (double)self->stat_lookups;
        avg_probes = (double)self->stat_probes_tried
                     / (double)self->stat_lookups;
    }

    return Py_BuildValue("{s:K, s:d, s:d, s:K, s:K}",
                         "total_lookups",
                         (unsigned long long)self->stat_lookups,
                         "avg_levels", avg_levels,
                         "avg_probes", avg_probes,
                         "total_levels",
                         (unsigned long long)self->stat_levels_visited,
                         "total_probes",
                         (unsigned long long)self->stat_probes_tried);
}

/* ── reset_probe_stats() ───────────────────────────────────────────
 *
 * Zero all probe-statistics counters.  Call this before a benchmark
 * batch so that the averages reflect only the intended workload.
 */
static PyObject *
densedict_reset_probe_stats(DenseDictObject *self,
                            PyObject        *Py_UNUSED(ignored))
{
    self->stat_lookups        = 0;
    self->stat_levels_visited = 0;
    self->stat_probes_tried   = 0;
    Py_RETURN_NONE;
}


/* ══════════════════════════════════════════════════════════════════
 * §6  Type & Module Definitions
 * ══════════════════════════════════════════════════════════════════ */

static PyMappingMethods densedict_as_mapping = {
    .mp_length        = (lenfunc)densedict_length,
    .mp_subscript     = (binaryfunc)densedict_subscript,
    .mp_ass_subscript = (objobjargproc)densedict_ass_subscript,
};

static PyMethodDef densedict_methods[] = {
    {"memory_usage",
     (PyCFunction)densedict_memory_usage, METH_NOARGS,
     PyDoc_STR(
         "memory_usage() -> int\n\n"
         "Return the total memory (in bytes) consumed by the hash-table\n"
         "overhead.  This includes the DenseDictObject itself and all\n"
         "allocated level arrays, but NOT the referenced Python objects.")},

    {"allocated_levels",
     (PyCFunction)densedict_allocated_levels, METH_NOARGS,
     PyDoc_STR(
         "allocated_levels() -> int\n\n"
         "Return the count of funnel levels that have been lazily\n"
         "allocated so far.  A newly-created DenseDict starts with 1.")},

    {"load_factor",
     (PyCFunction)densedict_load_factor, METH_NOARGS,
     PyDoc_STR(
         "load_factor() -> float\n\n"
         "Return used / allocated_capacity.  This measures how densely\n"
         "the *allocated* levels are packed, ignoring levels that have\n"
         "not yet been created.")},

    {"level_stats",
     (PyCFunction)densedict_level_stats, METH_NOARGS,
     PyDoc_STR(
         "level_stats() -> list[dict]\n\n"
         "Return per-level diagnostics.  Each dict contains:\n"
         "  level       – level index (0 = largest)\n"
         "  capacity    – slot count (always power of 2)\n"
         "  used        – occupied slots\n"
         "  load        – used / capacity\n"
         "  epsilon     – 1 − load  (fraction free)\n"
         "  probe_limit – current probe budget ⌈log₂(1/ε)⌉")},

    {"average_probes",
     (PyCFunction)densedict_average_probes, METH_NOARGS,
     PyDoc_STR(
         "average_probes() -> dict\n\n"
         "Return lookup probe statistics since the last reset:\n"
         "  total_lookups  – number of dd[key] calls tracked\n"
         "  avg_levels     – mean levels visited per lookup\n"
         "  avg_probes     – mean individual probes per lookup\n"
         "  total_levels   – cumulative levels visited\n"
         "  total_probes   – cumulative probes tried\n"
         "\n"
         "The paper claims avg_levels should stay near 1.x.")},

    {"reset_probe_stats",
     (PyCFunction)densedict_reset_probe_stats, METH_NOARGS,
     PyDoc_STR(
         "reset_probe_stats() -> None\n\n"
         "Zero all probe-statistics counters.  Call before a\n"
         "measurement batch so averages reflect only that batch.")},

    {NULL, NULL, 0, NULL}   /* sentinel */
};

static PyTypeObject DenseDict_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "densedict.DenseDict",
    .tp_doc       = PyDoc_STR(
        "DenseDict(capacity=1048576, probe_limit=8)\n\n"
        "A memory-efficient hash table using Elastic Hashing\n"
        "(arXiv:2501.02305).  Supports dd[key] = value, dd[key],\n"
        "del dd[key], and len(dd).\n\n"
        "The table is organised as a series of geometrically-shrinking\n"
        "levels.  Levels are allocated lazily, and each has a dynamic\n"
        "probe budget proportional to log(1/ε), where ε is the fraction\n"
        "of free slots.  Keys are never moved after insertion."),
    .tp_basicsize = sizeof(DenseDictObject),
    .tp_itemsize  = 0,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_new       = (newfunc)densedict_new,
    .tp_dealloc   = (destructor)densedict_dealloc,
    .tp_repr      = (reprfunc)densedict_repr,
    .tp_as_mapping = &densedict_as_mapping,
    .tp_methods   = densedict_methods,
};

static PyModuleDef densedict_module = {
    PyModuleDef_HEAD_INIT,
    .m_name  = "densedict",
    .m_doc   = PyDoc_STR(
        "densedict — A Python mapping backed by Elastic Hashing.\n\n"
        "See arXiv:2501.02305 for the theoretical foundations."),
    .m_size  = -1,
};

/* ── Module init ───────────────────────────────────────────────────
 *
 * This is the only non-static symbol in the entire module.
 */
PyMODINIT_FUNC
PyInit_densedict(void)
{
    PyObject *m;

    if (PyType_Ready(&DenseDict_Type) < 0) {
        return NULL;
    }

    m = PyModule_Create(&densedict_module);
    if (m == NULL) {
        return NULL;
    }

    Py_INCREF(&DenseDict_Type);
    if (PyModule_AddObject(m, "DenseDict",
                           (PyObject *)&DenseDict_Type) < 0) {
        Py_DECREF(&DenseDict_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
