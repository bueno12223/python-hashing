#ifndef ELASTIC_HASH_H
#define ELASTIC_HASH_H

#include "hash_common.h"

// ============================================================================
// Elastic Hash Table - Funnel/Level Strategy
// ============================================================================

#define MAX_LEVELS 32
#define INITIAL_CAPACITY 16

typedef struct {
    Key key;
    Value value;
    uint64_t hash;
    bool occupied;
} ElasticEntry;

typedef struct {
    ElasticEntry *slots;
    size_t capacity;
    size_t count;
} ElasticLevel;

typedef struct {
    ElasticLevel *levels;
    size_t num_levels;
    size_t total_capacity;
    size_t total_count;
    uint64_t base_seed;
    PerfMetrics metrics;
} ElasticHashTable;

// ============================================================================
// Function Declarations
// ============================================================================

ElasticHashTable* elastic_create(size_t initial_capacity);
void elastic_destroy(ElasticHashTable *table);
bool elastic_insert(ElasticHashTable *table, const Key *key, const Value *value);
bool elastic_lookup(ElasticHashTable *table, const Key *key, Value *value_out, uint64_t *probe_count);
double elastic_load_factor(const ElasticHashTable *table);
size_t elastic_memory_usage(const ElasticHashTable *table);
void elastic_print_stats(const ElasticHashTable *table);

#endif // ELASTIC_HASH_H
