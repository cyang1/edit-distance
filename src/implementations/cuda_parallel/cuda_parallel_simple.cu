#include <stdio.h>
#include <stdint.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define WARP_SIZE 32
// Assumed to be power of 2 for optimizations.
#define QUAD_SIZE 64
#define MAX_INPUT_LENGTH (QUAD_SIZE * QUAD_SIZE)
#define MAX_THREADS (MAX_INPUT_LENGTH / QUAD_SIZE)

#define MAX_ULL 0xFFFFFFFFFFFFFFFFULL
#define ULL_MSB (MAX_ULL ^ (MAX_ULL >> 1))

__device__ unsigned int device_min_idx, device_min_distance;

// Follows the method from http://www.gersteinlab.org/courses/452/09-spring/pdf/Myers.pdf
__global__ void
edit_distance_warp_kernel(const unsigned char* body,
    const unsigned int body_len,
    const unsigned char* input,
    const unsigned int input_len)
{
    unsigned int index = threadIdx.x;

    // Throw out threads that we don't need.
    int num_threads_necessary = (input_len + QUAD_SIZE - 1) / QUAD_SIZE;

    if (index >= num_threads_necessary)
        return;

    bool is_last_thread = index == (num_threads_necessary - 1);

    // Precompute each thread's portion of Peq[\sigma]
    uint64_t char_eq[UCHAR_MAX];
    for (int i = 0; i < UCHAR_MAX; i++) {
        char_eq[i] = 0;
    }
    __syncthreads();
    for (int i = index * QUAD_SIZE; i < min(input_len, (index + 1) * QUAD_SIZE); i++) {
        // Equivalent to (1 << (i % QUAD_SIZE))
        char_eq[input[i]] |= (1ull << (i & (QUAD_SIZE - 1)));
    }

    // Only the horizontal deltas need to be shared.
    uint64_t Pv = MAX_ULL,                                  // Pv = 1^m
             Mv = 0;                                        // Mv = 0
    __shared__ uint64_t Ph[MAX_THREADS], Mh[MAX_THREADS];
    __shared__ unsigned long long carry, carry_through;

    // Only matters for the last thread in the block.
    uint32_t score = input_len;                             // Score = m
    uint32_t min_score = input_len;
    uint32_t min_idx = 0;

    for (int j = 0; j < body_len; j++) {
        uint64_t Eq = char_eq[body[j]];
        uint64_t Xv = Eq | Mv;

        uint64_t Xh;
        if (num_threads_necessary == 1) {
            Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq;
        } else {
            // Computing Xh is hard because of carrying in the addition.
            if (threadIdx.x == 0) {
                carry = 0;
                carry_through = 0;
            }
            uint64_t Eq_Pv = Eq & Pv;
            uint64_t add_result = Eq_Pv + Pv;
            // Should carry
            if (add_result < Eq_Pv) atomicAdd(&carry, 2ull << index);
            if (add_result == MAX_ULL) atomicAdd(&carry_through, 1ull << index);
            if (threadIdx.x == 0) {
                // This is a problem...recursive?
                carry |= (((carry & carry_through) + carry_through) ^ carry_through);
            }
            if (carry & (1ull << index)) add_result++;
            Xh = (add_result ^ Pv) | Eq;
        }

        Ph[index] = Mv | ~(Xh | Pv);
        Mh[index] = Pv & Xh;

        if (is_last_thread) {
            // Equivalent to (1 << ((input_len - 1) % QUAD_SIZE))
            const uint64_t last_bit = 1ull << ((input_len - 1) & (QUAD_SIZE - 1));
            if (Ph[index] & last_bit)
                score++;
            else if (Mh[index] & last_bit)
                score--;

            if (score < min_score) {
                min_score = score;
                min_idx = j;
            }
        }

        // Have to ensure memory storage to Ph and Mh is done before
        // this next step.
        __syncthreads();

        uint8_t last_bit_p = 0, last_bit_m = 0;
        if (index != 0) {
            last_bit_p = (Ph[index - 1] & ULL_MSB) != 0;
            last_bit_m = (Mh[index - 1] & ULL_MSB) != 0;
        }

        Ph[index] = (Ph[index] << 1) | last_bit_p;
        Mh[index] = (Mh[index] << 1) | last_bit_m;

        Pv = Mh[index] | ~(Xv | Ph[index]);
        Mv = Ph[index] & Xv;
    }

    if (is_last_thread) {
        device_min_idx = min_idx;
        device_min_distance = min_score;
    }
}

void
edit_distance_cuda(const char *body,
    const unsigned int body_len,
    const char *input,
    const unsigned int input_len,
    unsigned int& min_idx,
    unsigned int& min_distance)
{
    // compute number of blocks and threads per block
    const int blocks = 1, threadsPerBlock = (input_len + QUAD_SIZE - 1) / QUAD_SIZE;

    if (input_len > MAX_INPUT_LENGTH) {
        printf("This simple CUDA implementation can only support inputs "
               "of length up to 2048.\n");
        return;
    }

    unsigned char *device_body, *device_input;

    cudaMalloc(&device_body, sizeof(unsigned char) * body_len);
    cudaMalloc(&device_input, sizeof(unsigned char) * input_len);

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    // copy the input data to the graphics card
    cudaMemcpy(device_body, body, sizeof(unsigned char) * body_len, cudaMemcpyHostToDevice);
    cudaMemcpy(device_input, input, sizeof(unsigned char) * input_len, cudaMemcpyHostToDevice);

    // second timer that only checks computation time
    double startComputeTime = CycleTimer::currentSeconds();

    // run kernel
    edit_distance_warp_kernel<<<blocks, threadsPerBlock>>>(
        device_body, body_len, device_input, input_len);
    cudaThreadSynchronize();

    // get the duration of the computation
    double computeDuration = CycleTimer::currentSeconds() - startComputeTime;

    // copy the results back to the host
    cudaMemcpyFromSymbol(&min_idx, device_min_idx, sizeof(min_idx), 0);
    cudaMemcpyFromSymbol(&min_distance, device_min_distance, sizeof(min_distance), 0);

    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("Location: %d\n", min_idx);
    printf("Minimum Distance: %d\n", min_distance);
    printf("Overall: %.3f ms\n", 1000.f * overallDuration);
    printf("Actual Computation Time: %.3f ms\n", 1000.f * computeDuration);

    // free memory buffers on the GPU
    cudaFree(device_body);
    cudaFree(device_input);
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
