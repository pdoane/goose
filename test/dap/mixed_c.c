/* The C half of mixed.goose: a call back into Goose, and a signal. */
#include <stdint.h>

int32_t goose_twice(int32_t x);

int32_t walk(int32_t n) {
    int32_t twice = goose_twice(n);
    return twice + 1;
}

int32_t crash(void) {
    volatile int32_t *nowhere = 0;
    return *nowhere;
}
