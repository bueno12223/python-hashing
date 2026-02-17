#ifndef PYTHON_DICT_H
#define PYTHON_DICT_H

#include "hash_common.h"

// ============================================================================
// CPython Dict Simulation - Compact Indices + Dense Entries
// ============================================================================

#define PYDICT_INITIAL_SIZE 8
#define PYDICT_EMPTY_INDEX -1
#define PYDICT_DUMMY_INDEX -2

typedef struct {
    uint64_t hash;
    Key key;
    Value value;
} PyDictEntry;

typedef struct {
    int32_t *indices;        // Sparse index array
    PyDictEntry *entries;    // Dense entries array
    size_t capacity;         // Size of indices array (power of 2)
    size_t num_entries;      // Number of filled entries
    size_t num_used;         // Entries used (including deleted)
    PerfMetrics metrics;
} PythonDict;

// ============================================================================
// Function Declarations
// ============================================================================

PythonDict* pydict_create(size_t initial_capacity);
void pydict_destroy(PythonDict *dict);
bool pydict_insert(PythonDict *dict, const Key *key, const Value *value);
bool pydict_lookup(PythonDict *dict, const Key *key, Value *value_out, uint64_t *probe_count);
double pydict_load_factor(const PythonDict *dict);
size_t pydict_memory_usage(const PythonDict *dict);
void pydict_print_stats(const PythonDict *dict);

#endif // PYTHON_DICT_H
