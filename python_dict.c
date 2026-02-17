#include "python_dict.h"
#include <assert.h>

// ============================================================================
// Helper Functions
// ============================================================================

static size_t next_power_of_2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

static void init_indices(int32_t *indices, size_t capacity) {
    for (size_t i = 0; i < capacity; i++) {
        indices[i] = PYDICT_EMPTY_INDEX;
    }
}

// CPython's perturbation-based probing
static size_t perturbed_probe(size_t hash, size_t index, size_t *perturb, size_t mask) {
    // Python uses: j = (5*j) + 1 + perturb
    index = ((5 * index) + 1 + *perturb) & mask;
    *perturb >>= 5;
    return index;
}

static bool needs_resize(const PythonDict *dict) {
    // Resize at 2/3 load factor (CPython strategy)
    return dict->num_entries >= (dict->capacity * 2) / 3;
}

static bool pydict_resize_internal(PythonDict *dict, size_t new_capacity);

// ============================================================================
// Core API Implementation
// ============================================================================

PythonDict* pydict_create(size_t initial_capacity) {
    PythonDict *dict = (PythonDict *)malloc(sizeof(PythonDict));
    assert(dict != NULL);
    
    if (initial_capacity < PYDICT_INITIAL_SIZE) {
        initial_capacity = PYDICT_INITIAL_SIZE;
    }
    
    dict->capacity = next_power_of_2(initial_capacity);
    dict->num_entries = 0;
    dict->num_used = 0;
    init_metrics(&dict->metrics);
    
    // Allocate indices array (sparse, mostly empty)
    dict->indices = (int32_t *)malloc(sizeof(int32_t) * dict->capacity);
    assert(dict->indices != NULL);
    init_indices(dict->indices, dict->capacity);
    
    // Allocate entries array (dense, compact)
    dict->entries = (PyDictEntry *)malloc(sizeof(PyDictEntry) * dict->capacity);
    assert(dict->entries != NULL);
    
    return dict;
}

void pydict_destroy(PythonDict *dict) {
    if (!dict) return;
    
    // Free all keys
    for (size_t i = 0; i < dict->num_used; i++) {
        free_key(&dict->entries[i].key);
    }
    
    free(dict->indices);
    free(dict->entries);
    free(dict);
}

static bool pydict_insert_no_resize(PythonDict *dict, const Key *key, const Value *value, uint64_t hash) {
    size_t mask = dict->capacity - 1;
    size_t index = hash & mask;
    size_t perturb = hash;
    uint64_t probe_count = 0;
    
    while (true) {
        probe_count++;
        int32_t entry_idx = dict->indices[index];
        
        if (entry_idx == PYDICT_EMPTY_INDEX) {
            // Found empty slot - insert new entry
            size_t new_entry_idx = dict->num_used;
            
            dict->entries[new_entry_idx].hash = hash;
            dict->entries[new_entry_idx].key = copy_key(key);
            dict->entries[new_entry_idx].value = *value;
            
            dict->indices[index] = (int32_t)new_entry_idx;
            dict->num_entries++;
            dict->num_used++;
            
            record_probe(&dict->metrics, probe_count);
            return true;
        }
        
        // Check if key already exists
        PyDictEntry *entry = &dict->entries[entry_idx];
        if (entry->hash == hash && keys_equal(&entry->key, key)) {
            // Update existing value
            entry->value = *value;
            return true;
        }
        
        // Continue probing with perturbation
        index = perturbed_probe(hash, index, &perturb, mask);
        
        // Safety check to avoid infinite loop
        if (probe_count > dict->capacity) {
            fprintf(stderr, "ERROR: Probing limit exceeded in pydict_insert\n");
            return false;
        }
    }
}

bool pydict_insert(PythonDict *dict, const Key *key, const Value *value) {
    double start_time = get_time_ms();
    
    // Check if resize needed
    if (needs_resize(dict)) {
        size_t new_capacity = dict->capacity * 2;
        if (!pydict_resize_internal(dict, new_capacity)) {
            dict->metrics.total_insert_time_ms += get_time_ms() - start_time;
            return false;
        }
    }
    
    uint64_t hash = hash_key(key, 0);
    bool result = pydict_insert_no_resize(dict, key, value, hash);
    
    dict->metrics.total_insert_time_ms += get_time_ms() - start_time;
    return result;
}

bool pydict_lookup(PythonDict *dict, const Key *key, Value *value_out, uint64_t *probe_count) {
    double start_time = get_time_ms();
    
    uint64_t hash = hash_key(key, 0);
    size_t mask = dict->capacity - 1;
    size_t index = hash & mask;
    size_t perturb = hash;
    uint64_t probes = 0;
    
    while (true) {
        probes++;
        int32_t entry_idx = dict->indices[index];
        
        if (entry_idx == PYDICT_EMPTY_INDEX) {
            // Key not found
            if (probe_count) {
                *probe_count = probes;
            }
            record_lookup(&dict->metrics, false, probes);
            dict->metrics.total_lookup_time_ms += get_time_ms() - start_time;
            return false;
        }
        
        // Check if this is our key
        PyDictEntry *entry = &dict->entries[entry_idx];
        if (entry->hash == hash && keys_equal(&entry->key, key)) {
            if (value_out) {
                *value_out = entry->value;
            }
            if (probe_count) {
                *probe_count = probes;
            }
            record_lookup(&dict->metrics, true, probes);
            dict->metrics.total_lookup_time_ms += get_time_ms() - start_time;
            return true;
        }
        
        // Continue probing with perturbation
        index = perturbed_probe(hash, index, &perturb, mask);
        
        // Safety check
        if (probes > dict->capacity) {
            if (probe_count) {
                *probe_count = probes;
            }
            record_lookup(&dict->metrics, false, probes);
            dict->metrics.total_lookup_time_ms += get_time_ms() - start_time;
            return false;
        }
    }
}

static bool pydict_resize_internal(PythonDict *dict, size_t new_capacity) {
    // Save old data
    int32_t *old_indices = dict->indices;
    PyDictEntry *old_entries = dict->entries;
    size_t old_capacity = dict->capacity;
    size_t old_num_used = dict->num_used;
    
    // Allocate new arrays
    dict->capacity = new_capacity;
    dict->indices = (int32_t *)malloc(sizeof(int32_t) * new_capacity);
    dict->entries = (PyDictEntry *)malloc(sizeof(PyDictEntry) * new_capacity);
    
    if (!dict->indices || !dict->entries) {
        // Restore old state on failure
        free(dict->indices);
        free(dict->entries);
        dict->indices = old_indices;
        dict->entries = old_entries;
        dict->capacity = old_capacity;
        return false;
    }
    
    init_indices(dict->indices, new_capacity);
    dict->num_entries = 0;
    dict->num_used = 0;
    
    // Reinsert all old entries
    for (size_t i = 0; i < old_num_used; i++) {
        PyDictEntry *old_entry = &old_entries[i];
        // Note: We reuse the key without copying (transfer ownership)
        if (!pydict_insert_no_resize(dict, &old_entry->key, &old_entry->value, old_entry->hash)) {
            // This shouldn't happen in a well-sized table
            fprintf(stderr, "ERROR: Failed to reinsert during resize\n");
        }
    }
    
    // Free old arrays (but not the keys - they were transferred)
    free(old_indices);
    free(old_entries);
    
    return true;
}

double pydict_load_factor(const PythonDict *dict) {
    return (double)dict->num_entries / (double)dict->capacity;
}

size_t pydict_memory_usage(const PythonDict *dict) {
    size_t total = sizeof(PythonDict);
    total += sizeof(int32_t) * dict->capacity;  // indices
    total += sizeof(PyDictEntry) * dict->capacity;  // entries
    return total;
}

void pydict_print_stats(const PythonDict *dict) {
    printf("=== Python Dict Statistics ===\n");
    printf("Capacity: %zu\n", dict->capacity);
    printf("Entries: %zu\n", dict->num_entries);
    printf("Load factor: %.2f%%\n", pydict_load_factor(dict) * 100.0);
    printf("Memory usage: %zu bytes (%.2f MB)\n", 
           pydict_memory_usage(dict),
           pydict_memory_usage(dict) / (1024.0 * 1024.0));
    
    printf("\nPerformance metrics:\n");
    printf("  Average PSL: %.2f\n", get_avg_psl(&dict->metrics));
    printf("  Max PSL: %lu\n", dict->metrics.max_probe_length);
    printf("  Total lookups: %lu (Success: %lu, Failed: %lu)\n",
           dict->metrics.total_lookups,
           dict->metrics.successful_lookups,
           dict->metrics.failed_lookups);
    printf("  Total insert time: %.2f ms\n", dict->metrics.total_insert_time_ms);
    printf("  Total lookup time: %.2f ms\n", dict->metrics.total_lookup_time_ms);
}
