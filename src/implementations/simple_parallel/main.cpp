#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>

#include "CycleTimer.h"
#include "simple_parallel.h"

int main(int argc, char** args) {

    if (argc != 3 && argc != 2) {
        printf("Usage:\n");
        printf("./%s input1 input2\n", args[0]);
        printf("./%s input_file\n", args[0]);
        exit(0);
    }

    std::string input, body;

    // Reading inputs from command line
    if (argc == 3) {
        input = std::string(args[1]);
        body = std::string(args[2]);
    } else {
        char* path = args[1];
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
    double startTime = CycleTimer::currentSeconds();
    edit_distance_parallel(body.c_str(), body.length(),
        input.c_str(), input.length(), min_idx, min_distance);
    double endTime = CycleTimer::currentSeconds();

    printf("Location: %d\n", min_idx);
    printf("Minimum Distance: %d\n", min_distance);
    printf("Time spent: %f\n", (endTime-startTime));
}