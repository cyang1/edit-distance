#ifndef CUDA_PARALLEL_H
#define CUDA_PARALLEL_H

void edit_distance_cuda(const char *body,
    const unsigned int body_len,
    const char *input,
    const unsigned int input_len,
    unsigned int& min_idx,
    unsigned int& min_distance);
void printCudaInfo();

#endif /* CUDA_PARALLEL_H */