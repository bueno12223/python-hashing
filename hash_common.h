#ifndef HASH_COMMON_H
#define HASH_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

// ============================================================================
// Common Data Types
// ============================================================================

typedef enum {
    KEY_TYPE_INT,
    KEY_TYPE_STRING
} KeyType;

typedef struct {
    KeyType type;
    union {
        int64_t int_val;
        char *str_val;
    } data;
} Key;

typedef struct {
    int64_t value;
} Value;

// ============================================================================
// Hash Functions
// ============================================================================

// Simple but effective hash function for integers
static inline uint64_t hash_int(int64_t key, uint64_t seed) {
    uint64_t x = (uint64_t)key;
    x ^= seed;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// MurmurHash3-inspired string hash function
static inline uint64_t hash_string(const char *str, uint64_t seed) {
    uint64_t hash = seed;
    size_t len = strlen(str);
    
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)(unsigned char)str[i];
        hash *= 0x100000001b3ULL; // FNV prime
    }
    
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= hash >> 33;
    hash *= 0xc4ceb9fe1a85ec53ULL;
    hash ^= hash >> 33;
    
    return hash;
}

// Generic hash function that dispatches based on key type
static inline uint64_t hash_key(const Key *key, uint64_t seed) {
    if (key->type == KEY_TYPE_INT) {
        return hash_int(key->data.int_val, seed);
    } else {
        return hash_string(key->data.str_val, seed);
    }
}

// ============================================================================
// Key Comparison
// ============================================================================

static inline bool keys_equal(const Key *a, const Key *b) {
    if (a->type != b->type) {
        return false;
    }
    
    if (a->type == KEY_TYPE_INT) {
        return a->data.int_val == b->data.int_val;
    } else {
        return strcmp(a->data.str_val, b->data.str_val) == 0;
    }
}

// ============================================================================
// Key Utilities
// ============================================================================

static inline Key make_int_key(int64_t val) {
    Key k;
    k.type = KEY_TYPE_INT;
    k.data.int_val = val;
    return k;
}

static inline Key make_string_key(const char *str) {
    Key k;
    k.type = KEY_TYPE_STRING;
    k.data.str_val = strdup(str);
    return k;
}

static inline void free_key(Key *k) {
    if (k->type == KEY_TYPE_STRING && k->data.str_val != NULL) {
        free(k->data.str_val);
        k->data.str_val = NULL;
    }
}

static inline Key copy_key(const Key *k) {
    if (k->type == KEY_TYPE_INT) {
        return make_int_key(k->data.int_val);
    } else {
        return make_string_key(k->data.str_val);
    }
}

// ============================================================================
// Performance Metrics
// ============================================================================

typedef struct {
    uint64_t total_probes;
    uint64_t total_lookups;
    uint64_t max_probe_length;
    uint64_t successful_lookups;
    uint64_t failed_lookups;
    double total_insert_time_ms;
    double total_lookup_time_ms;
} PerfMetrics;

static inline void init_metrics(PerfMetrics *m) {
    memset(m, 0, sizeof(PerfMetrics));
}

static inline void record_probe(PerfMetrics *m, uint64_t probe_count) {
    m->total_probes += probe_count;
    if (probe_count > m->max_probe_length) {
        m->max_probe_length = probe_count;
    }
}

static inline void record_lookup(PerfMetrics *m, bool found, uint64_t probe_count) {
    m->total_lookups++;
    m->total_probes += probe_count;
    if (probe_count > m->max_probe_length) {
        m->max_probe_length = probe_count;
    }
    if (found) {
        m->successful_lookups++;
    } else {
        m->failed_lookups++;
    }
}

static inline double get_avg_psl(const PerfMetrics *m) {
    if (m->total_lookups == 0) {
        return 0.0;
    }
    return (double)m->total_probes / (double)m->total_lookups;
}

// ============================================================================
// Timing Utilities
// ============================================================================

static inline double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

#endif // HASH_COMMON_H
