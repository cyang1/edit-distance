CC = g++
PARALLEL = edit_distance_pthread

all:
	$(CC) -std=c++11 -lpthread $(PARALLEL).cpp -g -O3 -o $(PARALLEL)
run: all
	./edit_distance_pthread
clean:
	rm $(PARALLEL)
