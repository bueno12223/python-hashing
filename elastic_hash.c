#include "elastic_hash.h"
#include <assert.h>
#include <math.h>

// ============================================================================
// Helper Functions
// ============================================================================

static uint64_t hash_for_level(const Key *key, int level, uint64_t base_seed) {
    // Each level uses a different seed to create independent hash functions
    uint64_t level_seed = base_seed + (uint64_t)(level * 0x9e3779b97f4a7c15ULL);
    return hash_key(key, level_seed);
}

static size_t level_capacity(size_t total_capacity, int level) {
    // Level i has capacity proportional to total_capacity / 2^i
    // Level 0 gets half the total, level 1 gets quarter, etc.
    if (level == 0) {
        return total_capacity / 2;
    }
    size_t cap = total_capacity / (1ULL << (level + 1));
    return cap > 0 ? cap : 1; // At least 1 slot per level
}

static void init_level(ElasticLevel *level, size_t capacity) {
    level->capacity = capacity;
    level->count = 0;
    level->slots = (ElasticEntry *)calloc(capacity, sizeof(ElasticEntry));
    assert(level->slots != NULL);
}

static void free_level(ElasticLevel *level) {
    if (level->slots) {
        for (size_t i = 0; i < level->capacity; i++) {
            if (level->slots[i].occupied) {
                free_key(&level->slots[i].key);
            }
        }
        free(level->slots);
        level->slots = NULL;
    }
}

// ============================================================================
// Core API Implementation
// ============================================================================

ElasticHashTable* elastic_create(size_t initial_capacity) {
    ElasticHashTable *table = (ElasticHashTable *)malloc(sizeof(ElasticHashTable));
    assert(table != NULL);
    
    if (initial_capacity < INITIAL_CAPACITY) {
        initial_capacity = INITIAL_CAPACITY;
    }
    
    // Calculate number of levels needed
    // We want enough levels so that sum of level capacities >= initial_capacity
    size_t num_levels = 0;
    size_t total_cap = 0;
    while (total_cap < initial_capacity && num_levels < MAX_LEVELS) {
        total_cap += level_capacity(initial_capacity, num_levels);
        num_levels++;
    }
    
    table->num_levels = num_levels;
    table->total_capacity = initial_capacity;
    table->total_count = 0;
    table->base_seed = 0x517cc1b727220a95ULL; // Random seed
    init_metrics(&table->metrics);
    
    // Allocate and initialize levels
    table->levels = (ElasticLevel *)malloc(sizeof(ElasticLevel) * num_levels);
    assert(table->levels != NULL);
    
    for (size_t i = 0; i < num_levels; i++) {
        size_t cap = level_capacity(initial_capacity, i);
        init_level(&table->levels[i], cap);
    }
    
    return table;
}

void elastic_destroy(ElasticHashTable *table) {
    if (!table) return;
    
    for (size_t i = 0; i < table->num_levels; i++) {
        free_level(&table->levels[i]);
    }
    free(table->levels);
    free(table);
}

bool elastic_insert(ElasticHashTable *table, const Key *key, const Value *value) {
    double start_time = get_time_ms();
    
    // Try to insert into each level in sequence
    for (int level = 0; level < (int)table->num_levels; level++) {
        ElasticLevel *lv = &table->levels[level];
        
        // Check if this level is too full (>95% for elastic strategy)
        double level_load = (double)lv->count / (double)lv->capacity;
        if (level_load > 0.95 && level + 1 < (int)table->num_levels) {
            // Skip to next level if this one is too full
            continue;
        }
        
        uint64_t hash = hash_for_level(key, level, table->base_seed);
        size_t idx = hash % lv->capacity;
        size_t start_idx = idx;
        size_t probe_count = 0;
        
        // Linear probing within this level
        do {
            probe_count++;
            
            // Found empty slot or matching key
            if (!lv->slots[idx].occupied) {
                // Insert here
                lv->slots[idx].key = copy_key(key);
                lv->slots[idx].value = *value;
                lv->slots[idx].hash = hash;
                lv->slots[idx].occupied = true;
                lv->count++;
                table->total_count++;
                
                record_probe(&table->metrics, probe_count);
                table->metrics.total_insert_time_ms += get_time_ms() - start_time;
                return true;
            }
            
            // Check if key already exists
            if (lv->slots[idx].hash == hash && keys_equal(&lv->slots[idx].key, key)) {
                // Update existing value
                lv->slots[idx].value = *value;
                table->metrics.total_insert_time_ms += get_time_ms() - start_time;
                return true;
            }
            
            // Continue probing
            idx = (idx + 1) % lv->capacity;
            
            // Limit probing to avoid infinite loops
            if (probe_count > lv->capacity) {
                break; // This level is full, try next level
            }
            
        } while (idx != start_idx);
    }
    
    // Failed to insert - table is full
    table->metrics.total_insert_time_ms += get_time_ms() - start_time;
    fprintf(stderr, "ERROR: Elastic hash table is full!\n");
    return false;
}

bool elastic_lookup(ElasticHashTable *table, const Key *key, Value *value_out, uint64_t *probe_count) {
    double start_time = get_time_ms();
    uint64_t total_probes = 0;
    
    // Search through levels in order
    for (int level = 0; level < (int)table->num_levels; level++) {
        ElasticLevel *lv = &table->levels[level];
        uint64_t hash = hash_for_level(key, level, table->base_seed);
        size_t idx = hash % lv->capacity;
        size_t start_idx = idx;
        
        do {
            total_probes++;
            
            if (!lv->slots[idx].occupied) {
                // Empty slot means key might be in a deeper level
                break;
            }
            
            // Check if this is our key
            if (lv->slots[idx].hash == hash && keys_equal(&lv->slots[idx].key, key)) {
                if (value_out) {
                    *value_out = lv->slots[idx].value;
                }
                if (probe_count) {
                    *probe_count = total_probes;
                }
                record_lookup(&table->metrics, true, total_probes);
                table->metrics.total_lookup_time_ms += get_time_ms() - start_time;
                return true;
            }
            
            // Continue probing within this level
            idx = (idx + 1) % lv->capacity;
            
        } while (idx != start_idx);
    }
    
    // Key not found
    if (probe_count) {
        *probe_count = total_probes;
    }
    record_lookup(&table->metrics, false, total_probes);
    table->metrics.total_lookup_time_ms += get_time_ms() - start_time;
    return false;
}

double elastic_load_factor(const ElasticHashTable *table) {
    return (double)table->total_count / (double)table->total_capacity;
}

size_t elastic_memory_usage(const ElasticHashTable *table) {
    size_t total = sizeof(ElasticHashTable);
    total += sizeof(ElasticLevel) * table->num_levels;
    
    for (size_t i = 0; i < table->num_levels; i++) {
        total += sizeof(ElasticEntry) * table->levels[i].capacity;
    }
    
    return total;
}

void elastic_print_stats(const ElasticHashTable *table) {
    printf("=== Elastic Hash Table Statistics ===\n");
    printf("Total capacity: %zu\n", table->total_capacity);
    printf("Total count: %zu\n", table->total_count);
    printf("Load factor: %.2f%%\n", elastic_load_factor(table) * 100.0);
    printf("Number of levels: %zu\n", table->num_levels);
    printf("Memory usage: %zu bytes (%.2f MB)\n", 
           elastic_memory_usage(table),
           elastic_memory_usage(table) / (1024.0 * 1024.0));
    
    printf("\nLevel distribution:\n");
    for (size_t i = 0; i < table->num_levels; i++) {
        ElasticLevel *lv = &table->levels[i];
        printf("  Level %zu: %zu/%zu (%.1f%% full)\n", 
               i, lv->count, lv->capacity,
               (double)lv->count / (double)lv->capacity * 100.0);
    }
    
    printf("\nPerformance metrics:\n");
    printf("  Average PSL: %.2f\n", get_avg_psl(&table->metrics));
    printf("  Max PSL: %lu\n", table->metrics.max_probe_length);
    printf("  Total lookups: %lu (Success: %lu, Failed: %lu)\n",
           table->metrics.total_lookups,
           table->metrics.successful_lookups,
           table->metrics.failed_lookups);
    printf("  Total insert time: %.2f ms\n", table->metrics.total_insert_time_ms);
    printf("  Total lookup time: %.2f ms\n", table->metrics.total_lookup_time_ms);
}
