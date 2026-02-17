# Elastic Hashing vs CPython Dict Benchmark

CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c11 -march=native
LDFLAGS = 

SOURCES = benchmark.c elastic_hash.c python_dict.c
HEADERS = hash_common.h elastic_hash.h python_dict.h
OBJECTS = $(SOURCES:.c=.o)
TARGET = benchmark

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

# Convenience targets for testing
test-elastic: $(TARGET)
	./$(TARGET) --elastic-only

test-pydict: $(TARGET)
	./$(TARGET) --pydict-only

# Debug build
debug: CFLAGS = -O0 -g -Wall -Wextra -std=c11
debug: clean $(TARGET)
