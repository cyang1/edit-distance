#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<iostream>
#include<fstream>
#include "CycleTimer.h"

int edit_distance_serial(char* s1, int n1, char* s2, int n2);

int main(int argc, char** args) {

	if (argc != 3 && argc != 2) {
		printf("Usage:\n");
		printf("./edit_distance_serial input1 input2\n");
		printf("./edit_distance_serial input_file\n");
		exit(0);
	}

	// Reading inputs from command line
	if (argc == 3) {
		int n1 = strlen(args[1]), n2 = strlen(args[2]);

		double startTime = CycleTimer::currentSeconds();
		printf("Distance: %d\n", edit_distance_serial(args[1], n1, args[2], n2));
		double endTime = CycleTimer::currentSeconds();

		printf("Time spent: %f\n", (endTime-startTime));
	}
	else {

		char* path = args[1];
		std::ifstream file; file.open(path);

		if (!file.is_open()) {
			printf("Invalid file path: %s\n", path);
			exit(0);
		}

		char n1str[12];
		char n2str[12];

		// read n1
		file >> n1str;
		int n1 = std::stoi(n1str);

		// read input1
		char* input1 = (char*)malloc(n1+1);
		file >> input1;

		// read n2;
		file >> n2str;
		int n2 = std::stoi(n2str);

		// read input2
		char* input2 = (char*)malloc(n2+1);
		file >> input2;

		// close
		file.close();

		// Compute edit distance with timer
		double startTime = CycleTimer::currentSeconds();
		printf("Distance: %d\n", edit_distance_serial(input1, n1, input2, n2));
		double endTime = CycleTimer::currentSeconds();

		printf("Time spent: %f\n", (endTime-startTime));
	}
}


int edit_distance_serial(char* s1, int n1, char* s2, int n2) {

	// Ensure second input is shorter
	if (n2 > n1) {
		char* tempS = s1; int tempN = n1;
		s1 = s2; n1 = n2;
		s2 = tempS; n2 = tempN;
	}

	// DP table initialization
	int* prevLine = (int*)malloc((n2+1) * sizeof(int));
	int* currentLine = (int*)malloc((n2+1) * sizeof(int));

	// Base-case values
	for (int i = 0; i < n2+1; i++)
		prevLine[i] = i;

	// Evaluate DP entries
	for (int r = 1; r < n1+1; r++) {
		currentLine[0] = r;
		for (int c = 1; c < n2+1; c++) {
			if (s1[r-1] == s2[c-1]) {
				currentLine[c] = prevLine[c-1];
			}
			else {
				int up = prevLine[c], left = currentLine[c-1], upLeft = prevLine[c-1];
				int min = up < left ? up : left;
				min = min < upLeft ? min : upLeft;
				currentLine[c] = min + 1;
			}
		}
		int* temp = prevLine;
		prevLine = currentLine;
		currentLine = temp;
	}

	// Extract answer, free memory
	int answer = currentLine[n2-1];
	free(prevLine); free(currentLine);
	return answer;
}
