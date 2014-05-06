#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <ctime>

int seed = 0;

char* random_string(int size) {
	char* output = (char*)malloc((size+1)*sizeof(char));
	std::srand(seed == 0 ? std::time(0) : seed); // current time is seed
	for (int i = 0; i < size; i++) {
		int random_int = std::rand();
		char c = (char)('a' + (random_int%26));
		output[i] = c;
	}
	output[size] = '\0';
	seed = std::rand()%(1<<30)+1;
	return output;
}