#ifndef SIMPLE_PARALLEL_H
#define SIMPLE_PARALLEL_H

#include <string>

void edit_distance_parallel(const char *body,
    const unsigned int body_len,
    const char *input,
    const unsigned int input_len,
    unsigned int& min_idx,
    unsigned int& min_distance);

#endif /* SIMPLE_PARALLEL_H */