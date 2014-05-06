#include <cstdio>
#include <limits>
#include <string>
#include <iostream>
#include <fstream>
#include <getopt.h>

#include "CycleTimer.h"
#include "implementations/cuda_parallel/cuda_parallel.h"
#include "implementations/simple_parallel/simple_parallel.h"

#define NUM_METHODS 2

std::string gen(unsigned int n) {
    const std::string letters = "ATCG";
    std::string ret = "";
    for(unsigned int i = 0; i < n; i++)
        ret += letters[rand() % letters.length()];
    return ret;
}

void usage(const char* progname) {
    printf("Usage: %s [options] scenename\n", progname);
    printf("Valid scenenames are: rgb, rgby, rand10k, rand100k, biglittle, littlebig, pattern, snow, snowsingle\n");
    printf("Program Options:\n");
    printf("  -b  --body <INT>           Use a body text length of <INT>\n");
    printf("  -c  --check                Check correctness of output\n");
    printf("  -i  --input <INT>          Use an input length of <INT>\n");
    printf("  -n  --num <INT>            Number of iterations\n");
    printf("  -h  --help                 This message\n");
}

int main(int argc, char** argv) {
    unsigned int input_length = 25;
    unsigned int body_length = 100000;
    unsigned int num_iterations = 3;

    bool check_correctness = false;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"help",     0, 0,  'h'},
        {"check",    0, 0,  'c'},
        {"body",     1, 0,  'b'},
        {"input",    1, 0,  'i'},
        {"num",      1, 0,  'n'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "b:i:n:ch", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'b':
            body_length = atoi(optarg);
            break;
        case 'c':
            check_correctness = true;
            break;
        case 'i':
            input_length = atoi(optarg);
            break;
        case 'n':
            num_iterations = atoi(optarg);
            break;
        case 'h':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    printCudaInfo();

    double best_times[NUM_METHODS];
    for (int i = 0; i < NUM_METHODS; i++)
        best_times[i] = std::numeric_limits<double>::infinity();
    double startTime;

    for (unsigned int i = 0; i < num_iterations; i++) {
        std::string body = gen(body_length);
        std::string input = gen(input_length);

        unsigned int min_idx[NUM_METHODS], min_distance[NUM_METHODS];

        startTime = CycleTimer::currentSeconds();
        edit_distance_parallel(body.c_str(), body.length(),
            input.c_str(), input.length(), min_idx[0], min_distance[0]);
        best_times[0] = std::min(best_times[0], CycleTimer::currentSeconds() - startTime);

        startTime = CycleTimer::currentSeconds();
        edit_distance_cuda(body.c_str(), body.length(),
            input.c_str(), input.length(), min_idx[1], min_distance[1]);
        best_times[1] = std::min(best_times[1], CycleTimer::currentSeconds() - startTime);

        if (check_correctness) {
            for (int j = 0; j < NUM_METHODS; j++) {
                for (int k = j + 1; k < NUM_METHODS; k++) {
                    if (min_idx[j] != min_idx[k] || min_distance[j] != min_distance[k]) {
                        printf("ERROR: Results do not agree.\n");
                        printf("Minimum Indicies: %d: %d, %d: %d\n", j, min_idx[j], k, min_idx[k]);
                        printf("Minimum Distances: %d: %d, %d: %d\n", j, min_distance[j], k, min_distance[k]);
                    }
                }
            }
        }
    }

    printf("Simple Parallel Method: %lf ms\n", best_times[0] * 1000);
    printf("CUDA: %lf ms\n", best_times[1] * 1000);

    return 0;
}