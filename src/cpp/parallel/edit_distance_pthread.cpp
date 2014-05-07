#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include "CycleTimer.h"
#include "Tools.h"

void* hello(void* stuff);
char* random_string(int size);
inline int min(int a, int b, int c);
int edit_distance_serial(char* s1, int n1, char* s2, int n2);
int edit_distance_diag(char* s1, int n1, char* s2, int n2);
int edit_distance_diag_parallel(char* s1, int n1, char* s2, int n2, int num_threads);
void* worker_function(void* ptr);


// worker data for edit distance diag parallel
struct worker_data {
	int serial_rounds;
	char* s1;
	char* s2;
	int n1;
	int n2;
	int* prevDiagA;
	int* prevDiagB;
	int* curDiag;
	int tid;
	int num_threads;
	pthread_barrier_t* barr_ptr;
};

int num_threads_global = 6;

int main(int argc, char** argv) {
	
	int size_left = 50000, size_right = 100000;
	char* left = random_string(size_left);
	char* right = random_string(size_right);
	printf("Left Size: %d Right Size: %d\n", size_left, size_right);
	//printf("S1: %s\n", left);
	//printf("S2: %s\n", right);
	
	double startTime = CycleTimer::currentSeconds();
	printf("Distance: %d\n", edit_distance_serial(left, size_left, right, size_right));
	double endTime = CycleTimer::currentSeconds();
	printf("Edit Distance Row: %f seconds\n", (endTime-startTime));
	
	startTime = CycleTimer::currentSeconds();
	printf("Distance: %d\n", edit_distance_diag(left, size_left, right, size_right));
	endTime = CycleTimer::currentSeconds();
	printf("Edit Distance Diag: %f seconds\n", (endTime-startTime));
	
	startTime = CycleTimer::currentSeconds();
	printf("Distance: %d\n", edit_distance_diag_parallel(left, size_left, right, size_right, num_threads_global));
	endTime = CycleTimer::currentSeconds();
	printf("Edit Distance Diag Parallel: %f seconds\n", (endTime-startTime));

	
	return 0;
}

inline int min(int a, int b, int c) {
	int x = a < b ? a : b;
	return x < c ? x : c;
}


int edit_distance_diag_parallel(char* s1, int n1, char* s2, int n2, int num_threads) {
	// Make s2 longer
	if (n1 > n2) {
		char* tempS = s1; int tempN = n1;
		s1 = s2; n1 = n2;
		s2 = tempS; n2 = tempN;
	}
	
	int* prevDiagA = (int*)malloc(sizeof(int)*(n1+1));
	int* prevDiagB = (int*)malloc(sizeof(int)*(n1+1));
	int* curDiag = (int*)malloc(sizeof(int)*(n1+1));
	
	int serial_rounds = 1000 < n1 ? 1000 : n1;
	
	pthread_barrier_t barr;
	pthread_barrier_init(&barr, NULL, num_threads);
	
	worker_data* datas = (worker_data*)malloc(sizeof(worker_data)*num_threads);
	for (int i = 0; i < num_threads; i++) {
		datas[i].serial_rounds = serial_rounds;
		datas[i].s1 = s1; 
		datas[i].s2 = s2;
		datas[i].n1 = n1; 
		datas[i].n2 = n2;
		datas[i].prevDiagA = prevDiagA; 
		datas[i].prevDiagB = prevDiagB;
		datas[i].curDiag = curDiag;
		datas[i].tid = i;
		datas[i].barr_ptr = &barr;
		datas[i].num_threads = num_threads;
	}
	
	pthread_t tids[num_threads];
	for (int i = 0; i < num_threads; i++) {
		pthread_create(&tids[i], NULL, worker_function, (void*)&datas[i]);
	}
	for (int i = 0; i < num_threads; i++) {
		pthread_join(tids[i], NULL);
	}
	
	int ans;
	switch ((n1 + n2 + 1) % 3) {
		case 0:
			ans = curDiag[0];
		case 1:
			ans = prevDiagA[0];
		case 2:
			ans = prevDiagB[0];
		}
	free(prevDiagA); free(prevDiagB); free(curDiag);
	return ans;;
}

/*
struct worker_data {
	int serial_rounds;
	char* s1, s2;
	int n1, n2;
	int* prevDiagA, prevDiagB, curDiag;
	int tid;
	pthread_barrier_t* barr_ptr;
};
*/

void* worker_function(void* ptr) {
	worker_data* data = (worker_data*)ptr;
	int serial_rounds = data->serial_rounds;
	char* s1 = data->s1; char* s2 = data->s2;
	int n1 = data->n1; int n2 = data->n2;
	int* prevDiagA = data->prevDiagA; 
	int* prevDiagB = data->prevDiagB;
	int* curDiag = data->curDiag;
	int tid = data->tid;
	pthread_barrier_t* barr_ptr = data->barr_ptr;
	int num_threads = data->num_threads;
	
	if (tid == 0) {
		prevDiagA[0] = 0; // illustration
		prevDiagB[0] = 1;
		prevDiagB[1] = 1;

		for (int a = 2; a < serial_rounds; a++) {
			curDiag[0] = a;
			curDiag[a] = a;
			for (int i = 1; i < a; i++) {
				int idx1 = a - i - 1, idx2 = i - 1;
				if (s1[idx1] == s2[idx2])
					curDiag[i] = prevDiagA[i - 1];
				else {
					int up = prevDiagB[i], left = prevDiagB[i - 1];
					int upLeft = prevDiagA[i - 1];
					curDiag[i] = 1 + min(up, left, upLeft);
				}
			}
			int* tempA = prevDiagA;
			prevDiagA = prevDiagB;
			prevDiagB = curDiag;
			curDiag = tempA;
		}
		pthread_barrier_wait(barr_ptr);
	} else
		pthread_barrier_wait(barr_ptr);

	if (tid != 0) {
		switch (serial_rounds % 3) {
		case 0: {
			int* temp = curDiag;
			curDiag = prevDiagA;
			prevDiagA = prevDiagB;
			prevDiagB = temp;
		}; break;
		case 1: {
			int* temp = curDiag;
			curDiag = prevDiagB;
			prevDiagB = prevDiagA;
			prevDiagA = temp;
		}; break;
		case 2: {
			// do nothing
		}; break;
		}
	}

	for (int a = serial_rounds; a <= n1; a++) {
		if (tid == 0) {
			curDiag[0] = a;
			curDiag[a] = a;
		}
		int work = a - 1;
		int perWork = work / num_threads;

		for (int i = 1 + perWork * tid; i < ((tid == num_threads - 1) ? a
				: 1 + perWork * (tid + 1)); i++) {
			int idx1 = a - i - 1, idx2 = i - 1;
			if (s1[idx1] == s2[idx2])
				curDiag[i] = prevDiagA[i - 1];
			else {
				int up = prevDiagB[i], left = prevDiagB[i - 1];
				int upLeft = prevDiagA[i - 1];
				curDiag[i] = 1 + min(up, left, upLeft);
			}
		}
		int* tempA = prevDiagA;
		prevDiagA = prevDiagB;
		prevDiagB = curDiag;
		curDiag = tempA;

		pthread_barrier_wait(barr_ptr);
	}

	for (int b = 0; b < (n2 - n1); b++) {
		if (tid == 0)
			curDiag[n1] = n1 + b + 1;
		int work = n1;
		int perWork = work/num_threads;

		for (int i = perWork * tid; i < ((tid == num_threads - 1) ? n1 : perWork * (tid+1)); i++) {
			int idx1 = n1 - i - 1, idx2 = b + i;
			if (s1[idx1] == s2[idx2])
				curDiag[i] = (b == 0 ? prevDiagA[i]
						: prevDiagA[i + 1]);
			else {
				int up = prevDiagB[i + 1], left = prevDiagB[i];
				int upLeft = (b == 0 ? prevDiagA[i]
						: prevDiagA[i + 1]);
				curDiag[i] = 1 + min(up, left, upLeft);
			}
		}
		int* tempA = prevDiagA;
		prevDiagA = prevDiagB;
		prevDiagB = curDiag;
		curDiag = tempA;

		pthread_barrier_wait(barr_ptr);
	}

	for (int c = 0; c < n1 - serial_rounds; c++) {
		int diagSize = n1 - c;
		int perWork = diagSize/num_threads;
		
		for (int i = perWork * tid; i < ((tid == num_threads - 1) ? diagSize : perWork * (tid+1)); i++) {
			int idx1 = n1 - i - 1, idx2 = (n2 - n1) + c + i;
			if (s1[idx1] == s2[idx2])
				curDiag[i] = ((n2 - n1 == 0) && (c == 0) ? prevDiagA[i]
						: prevDiagA[i + 1]);
			else {
				int up = prevDiagB[i + 1], left = prevDiagB[i];
				int upLeft = ((n2 - n1 == 0) && (c == 0) ? prevDiagA[i]
						: prevDiagA[i + 1]);
				curDiag[i] = 1 + min(up, left, upLeft);
			}
		}
		int* tempA = prevDiagA;
		prevDiagA = prevDiagB;
		prevDiagB = curDiag;
		curDiag = tempA;
		
		pthread_barrier_wait(barr_ptr);
	}

	if (tid == 0) {
		for (int c = n1 - serial_rounds; c < n1; c++) {
			int diagSize = n1 - c;
			
			for (int i = 0; i < diagSize; i++) {
				int idx1 = n1 - i - 1, idx2 = (n2 - n1) + c + i;
				if (s1[idx1] == s2[idx2])
					curDiag[i] = ((n2 - n1 == 0) && (c == 0) ? prevDiagA[i]
							: prevDiagA[i + 1]);
				else {
					int up = prevDiagB[i + 1], left = prevDiagB[i];
					int upLeft = ((n2 - n1 == 0) && (c == 0) ? prevDiagA[i]
							: prevDiagA[i + 1]);
					curDiag[i] = 1 + min(up, left, upLeft);
				}
			}
			int* tempA = prevDiagA;
			prevDiagA = prevDiagB;
			prevDiagB = curDiag;
			curDiag = tempA;
		}
	}
}

int edit_distance_diag(char* s1, int n1, char* s2, int n2) {
	// Make s2 longer
	if (n1 > n2) {
		char* tempS = s1; int tempN = n1;
		s1 = s2; n1 = n2;
		s2 = tempS; n2 = tempN;
	}

	int* prevDiagA = (int*)malloc(sizeof(int)*(n1+1));
	int* prevDiagB = (int*)malloc(sizeof(int)*(n1+1));
	int* curDiag = (int*)malloc(sizeof(int)*(n1+1));

	prevDiagA[0] = 0; // illustration
	prevDiagB[0] = 1;
	prevDiagB[1] = 1;

	for (int a = 2; a <= n1; a++) {
		curDiag[0] = a;
		curDiag[a] = a;
		for (int i = 1; i < a; i++) {
			int idx1 = a - i - 1, idx2 = i - 1;
			if (s1[idx1] == s2[idx2])
				curDiag[i] = prevDiagA[i - 1];
			else {
				int up = prevDiagB[i], left = prevDiagB[i - 1];
				int upLeft = prevDiagA[i - 1];
				curDiag[i] = 1 + min(up, left, upLeft);
			}
		}
		int* tempA = prevDiagA;
		prevDiagA = prevDiagB;
		prevDiagB = curDiag;
		curDiag = tempA;
	}

	for (int b = 0; b < (n2 - n1); b++) {
		curDiag[n1] = n1 + b + 1;
		for (int i = 0; i < n1; i++) {
			int idx1 = n1 - i - 1, idx2 = b + i;
			if (s1[idx1] == s2[idx2])
				curDiag[i] = (b == 0 ? prevDiagA[i] : prevDiagA[i + 1]);
			else {
				int up = prevDiagB[i + 1], left = prevDiagB[i];
				int upLeft = (b == 0 ? prevDiagA[i] : prevDiagA[i + 1]);
				curDiag[i] = 1 + min(up, left, upLeft);
			}
		}
		int* tempA = prevDiagA;
		prevDiagA = prevDiagB;
		prevDiagB = curDiag;
		curDiag = tempA;
	}

	for (int c = 0; c < n1; c++) {
		int diagSize = n1 - c;
		for (int i = 0; i < diagSize; i++) {
			int idx1 = n1 - i - 1, idx2 = (n2 - n1) + c + i;
			if (s1[idx1] == s2[idx2])
				curDiag[i] = ((n2 - n1 == 0) && (c == 0) ? prevDiagA[i] : prevDiagA[i + 1]);
			else {
				int up = prevDiagB[i + 1], left = prevDiagB[i];
				int upLeft = ((n2 - n1 == 0) && (c == 0) ? prevDiagA[i]
						: prevDiagA[i + 1]);
				curDiag[i] = 1 + min(up, left, upLeft);
			}
		}
		int* tempA = prevDiagA;
		prevDiagA = prevDiagB;
		prevDiagB = curDiag;
		curDiag = tempA;
	}
	
	int ans = prevDiagB[0];
	free(prevDiagA); free(prevDiagB); free(curDiag);
	return ans;	
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
				currentLine[c] = min(up, left, upLeft) + 1;
			}
		}
		int* temp = prevLine;
		prevLine = currentLine;
		currentLine = temp;
	}
	
	// Extract answer, free memory
	int answer = prevLine[n2];
	free(prevLine); free(currentLine);
	return answer;
}
