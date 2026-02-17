#include "elastic_hash.h"
#include "python_dict.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <math.h>

// ============================================================================
// Test Data Generation
// ============================================================================

#define NUM_TEST_KEYS 1000000
#define STRING_KEY_LENGTH 16

typedef struct {
    Key *keys;
    Value *values;
    size_t count;
} TestData;

static char random_char(void) {
    static const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    return charset[rand() % (sizeof(charset) - 1)];
}

static void generate_random_string(char *str, size_t length) {
    for (size_t i = 0; i < length - 1; i++) {
        str[i] = random_char();
    }
    str[length - 1] = '\0';
}

TestData* generate_string_keys(size_t count) {
    TestData *data = (TestData *)malloc(sizeof(TestData));
    data->count = count;
    data->keys = (Key *)malloc(sizeof(Key) * count);
    data->values = (Value *)malloc(sizeof(Value) * count);
    
    for (size_t i = 0; i < count; i++) {
        char str[STRING_KEY_LENGTH + 1];
        generate_random_string(str, STRING_KEY_LENGTH + 1);
        data->keys[i] = make_string_key(str);
        data->values[i].value = (int64_t)i;
    }
    
    return data;
}

TestData* generate_int_keys(size_t count) {
    TestData *data = (TestData *)malloc(sizeof(TestData));
    data->count = count;
    data->keys = (Key *)malloc(sizeof(Key) * count);
    data->values = (Value *)malloc(sizeof(Value) * count);
    
    for (size_t i = 0; i < count; i++) {
        // Generate random integers with good distribution
        int64_t key_val = (int64_t)rand() * (int64_t)rand();
        data->keys[i] = make_int_key(key_val);
        data->values[i].value = (int64_t)i;
    }
    
    return data;
}

TestData* generate_mixed_keys(size_t count) {
    TestData *data = (TestData *)malloc(sizeof(TestData));
    data->count = count;
    data->keys = (Key *)malloc(sizeof(Key) * count);
    data->values = (Value *)malloc(sizeof(Value) * count);
    
    for (size_t i = 0; i < count; i++) {
        if (rand() % 2 == 0) {
            // String key
            char str[STRING_KEY_LENGTH + 1];
            generate_random_string(str, STRING_KEY_LENGTH + 1);
            data->keys[i] = make_string_key(str);
        } else {
            // Integer key
            int64_t key_val = (int64_t)rand() * (int64_t)rand();
            data->keys[i] = make_int_key(key_val);
        }
        data->values[i].value = (int64_t)i;
    }
    
    return data;
}

void free_test_data(TestData *data) {
    if (!data) return;
    
    for (size_t i = 0; i < data->count; i++) {
        free_key(&data->keys[i]);
    }
    
    free(data->keys);
    free(data->values);
    free(data);
}

// ============================================================================
// Benchmarking Functions
// ============================================================================

void benchmark_elastic_at_load_factor(TestData *data, double target_load) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("ELASTIC HASH - Load Factor: %.0f%%\n", target_load * 100.0);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    size_t num_inserts = (size_t)(data->count * target_load);
    ElasticHashTable *table = elastic_create(data->count);
    
    // Insert keys
    double insert_start = get_time_ms();
    for (size_t i = 0; i < num_inserts; i++) {
        if (!elastic_insert(table, &data->keys[i], &data->values[i])) {
            fprintf(stderr, "Failed to insert key %zu\n", i);
            break;
        }
    }
    double insert_time = get_time_ms() - insert_start;
    
    printf("Inserted %zu keys in %.2f ms (%.2f keys/ms)\n", 
           num_inserts, insert_time, num_inserts / insert_time);
    
    // Perform lookups
    size_t num_lookups = num_inserts;
    double lookup_start = get_time_ms();
    size_t found_count = 0;
    
    for (size_t i = 0; i < num_lookups; i++) {
        Value val;
        uint64_t probe_count;
        if (elastic_lookup(table, &data->keys[i % num_inserts], &val, &probe_count)) {
            found_count++;
        }
    }
    double lookup_time = get_time_ms() - lookup_start;
    
    printf("Performed %zu lookups in %.2f ms (%.2f lookups/ms)\n",
           num_lookups, lookup_time, num_lookups / lookup_time);
    printf("Found: %zu / %zu\n", found_count, num_lookups);
    
    elastic_print_stats(table);
    elastic_destroy(table);
}

void benchmark_pydict_at_load_factor(TestData *data, double target_load) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("PYTHON DICT - Load Factor: %.0f%%\n", target_load * 100.0);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    size_t num_inserts = (size_t)(data->count * target_load);
    PythonDict *dict = pydict_create(data->count);
    
    // Insert keys
    double insert_start = get_time_ms();
    for (size_t i = 0; i < num_inserts; i++) {
        if (!pydict_insert(dict, &data->keys[i], &data->values[i])) {
            fprintf(stderr, "Failed to insert key %zu\n", i);
            break;
        }
    }
    double insert_time = get_time_ms() - insert_start;
    
    printf("Inserted %zu keys in %.2f ms (%.2f keys/ms)\n", 
           num_inserts, insert_time, num_inserts / insert_time);
    
    // Perform lookups
    size_t num_lookups = num_inserts;
    double lookup_start = get_time_ms();
    size_t found_count = 0;
    
    for (size_t i = 0; i < num_lookups; i++) {
        Value val;
        uint64_t probe_count;
        if (pydict_lookup(dict, &data->keys[i % num_inserts], &val, &probe_count)) {
            found_count++;
        }
    }
    double lookup_time = get_time_ms() - lookup_start;
    
    printf("Performed %zu lookups in %.2f ms (%.2f lookups/ms)\n",
           num_lookups, lookup_time, num_lookups / lookup_time);
    printf("Found: %zu / %zu\n", found_count, num_lookups);
    
    pydict_print_stats(dict);
    pydict_destroy(dict);
}

void benchmark_10m_lookups(TestData *data) {
    printf("\n════════════════════════════════════════════════════\n");
    printf("10 MILLION LOOKUP BENCHMARK (90%% Load Factor)\n");
    printf("════════════════════════════════════════════════════\n");
    
    size_t num_inserts = (size_t)(data->count * 0.90);
    size_t num_lookups = 10000000;
    
    // Elastic Hash
    printf("\n--- Elastic Hash ---\n");
    ElasticHashTable *etable = elastic_create(data->count);
    for (size_t i = 0; i < num_inserts; i++) {
        elastic_insert(etable, &data->keys[i], &data->values[i]);
    }
    
    double elastic_start = get_time_ms();
    for (size_t i = 0; i < num_lookups; i++) {
        Value val;
        elastic_lookup(etable, &data->keys[i % num_inserts], &val, NULL);
    }
    double elastic_time = get_time_ms() - elastic_start;
    
    printf("10M lookups: %.2f ms (%.2f M lookups/sec)\n", 
           elastic_time, (num_lookups / elastic_time) / 1000.0);
    printf("Average PSL: %.2f\n", get_avg_psl(&etable->metrics));
    printf("Max PSL: %llu\n", (unsigned long long)etable->metrics.max_probe_length);
    
    // Python Dict
    printf("\n--- Python Dict ---\n");
    PythonDict *pdict = pydict_create(data->count);
    for (size_t i = 0; i < num_inserts; i++) {
        pydict_insert(pdict, &data->keys[i], &data->values[i]);
    }
    
    double pydict_start = get_time_ms();
    for (size_t i = 0; i < num_lookups; i++) {
        Value val;
        pydict_lookup(pdict, &data->keys[i % num_inserts], &val, NULL);
    }
    double pydict_time = get_time_ms() - pydict_start;
    
    printf("10M lookups: %.2f ms (%.2f M lookups/sec)\n", 
           pydict_time, (num_lookups / pydict_time) / 1000.0);
    printf("Average PSL: %.2f\n", get_avg_psl(&pdict->metrics));
    printf("Max PSL: %llu\n", (unsigned long long)pdict->metrics.max_probe_length);
    
    // Comparison
    printf("\n--- Comparison ---\n");
    double speedup = pydict_time / elastic_time;
    printf("Elastic vs Python: %.2fx %s\n", 
           fabs(speedup),
           speedup > 1.0 ? "faster" : "slower");
    
    elastic_destroy(etable);
    pydict_destroy(pdict);
}

void print_comparative_table(TestData *data) {
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║           COMPARATIVE PERFORMANCE SUMMARY                     ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    double load_factors[] = {0.60, 0.80, 0.90, 0.95};
    
    printf("┌─────────────────┬──────────────┬──────────────┬──────────────┐\n");
    printf("│ Metric          │ Elastic Hash │ Python Dict  │ Winner       │\n");
    printf("├─────────────────┼──────────────┼──────────────┼──────────────┤\n");
    
    for (int i = 0; i < 4; i++) {
        double lf = load_factors[i];
        size_t num_inserts = (size_t)(data->count * lf);
        
        // Benchmark Elastic
        ElasticHashTable *etable = elastic_create(data->count);
        for (size_t j = 0; j < num_inserts; j++) {
            elastic_insert(etable, &data->keys[j], &data->values[j]);
        }
        double elastic_psl = get_avg_psl(&etable->metrics);
        size_t elastic_mem = elastic_memory_usage(etable);
        
        // Benchmark Python
        PythonDict *pdict = pydict_create(data->count);
        for (size_t j = 0; j < num_inserts; j++) {
            pydict_insert(pdict, &data->keys[j], &data->values[j]);
        }
        double pydict_psl = get_avg_psl(&pdict->metrics);
        size_t pydict_mem = pydict_memory_usage(pdict);
        
        printf("│ Avg PSL @%2.0f%%   │ %12.2f │ %12.2f │ %-12s │\n",
               lf * 100.0, elastic_psl, pydict_psl,
               elastic_psl < pydict_psl ? "Elastic" : "Python");
        
        elastic_destroy(etable);
        pydict_destroy(pdict);
    }
    
    printf("└─────────────────┴──────────────┴──────────────┴──────────────┘\n");
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║     ELASTIC HASHING vs CPYTHON DICT - PROOF OF CONCEPT        ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    printf("\nGenerating %d test keys...\n", NUM_TEST_KEYS);
    TestData *data = generate_mixed_keys(NUM_TEST_KEYS);
    printf("Test data generated successfully.\n");
    
    // Benchmark at different load factors
    double load_factors[] = {0.60, 0.80, 0.90, 0.95};
    
    for (int i = 0; i < 4; i++) {
        benchmark_elastic_at_load_factor(data, load_factors[i]);
        benchmark_pydict_at_load_factor(data, load_factors[i]);
    }
    
    // 10M lookup stress test
    benchmark_10m_lookups(data);
    
    // Summary table
    print_comparative_table(data);
    
    // Cleanup
    free_test_data(data);
    
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                    BENCHMARK COMPLETE                          ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
