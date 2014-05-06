#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <fstream>

#include "cuda_parallel.h"

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 2) {
        printf("Usage:\n");
        printf("./%s input1 input2\n", argv[0]);
        printf("./%s input_file\n", argv[0]);
        exit(0);
    }

    std::string input, body;

    printCudaInfo();

    // Reading inputs from command line
    if (argc == 3) {
        input = std::string(argv[1]);
        body = std::string(argv[2]);
    }
    else {
        char* path = argv[1];
        std::ifstream file; file.open(path);

        if (!file.is_open()) {
            printf("Invalid file path: %s\n", path);
            exit(0);
        }

        int n1, n2;

        file >> n1 >> input >> n2 >> body;

        file.close();
    }

    // Input must be shorter than body.
    if (input.length() > body.length()) {
        std::string tmp = input;
        input = body;
        body = tmp;
    }

    unsigned int min_idx, min_distance;
    edit_distance_cuda(body.c_str(), body.length(),
        input.c_str(), input.length(), min_idx, min_distance);
}
