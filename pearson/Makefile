CXX2=g++
CXXFLAGS2=-std=c++17 -g -O3 -flto -march=native -Wall -Wunused -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Woverloaded-virtual -Wpedantic -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wuseless-cast
 
CXX=g++-13
CXXFLAGS=-std=c++17 -g -Wunused -Wall -Wunused

all: pearson_par pearson verify

pearson_par: vectorOpt datasetOpt analysisOpt pearson.cpp 
	$(CXX2) $(CXXFLAGS2) pearson.cpp vector.opt.o dataset.opt.o analysis.opt.o -o pearson_par

pearson: vector dataset analysis pearson.cpp 
	$(CXX) $(CXXFLAGS) pearson.cpp vector.o dataset.o analysis.o -o pearson

analysis: vector analysis.hpp analysis.cpp
	$(CXX) $(CXXFLAGS) -c analysis.cpp -o analysis.o

dataset: vector dataset.hpp dataset.cpp
	$(CXX) $(CXXFLAGS) -c dataset.cpp -o dataset.o

vector: vector.hpp vector.cpp
	$(CXX) $(CXXFLAGS) -c vector.cpp -o vector.o

analysisOpt: vector analysis.hpp analysis.cpp
	$(CXX2) $(CXXFLAGS2) -c analysis.cpp -o analysis.opt.o

datasetOpt: vector dataset.hpp dataset.cpp
	$(CXX2) $(CXXFLAGS2) -c dataset.cpp -o dataset.opt.o

vectorOpt: vector.hpp vector.cpp
	$(CXX2) $(CXXFLAGS2) -c vector.cpp -o vector.opt.o

verify: verify.c
	$(CC) verify.c -o verify

clean:
	rm -rf verify pearson pearson_par *.o *.dSYM 2> /dev/null
