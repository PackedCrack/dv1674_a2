CXX2=g++-13
CXXFLAGS2=-std=c++20 -g -O0 -flto -march=native -pthread -save-temps=obj -fverbose-asm -masm=intel -Wall -Wunused -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Woverloaded-virtual -Wpedantic -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wuseless-cast
 
CXX=g++-13
CXXFLAGS=-std=c++17 -g -Wunused -Wall -Wunused

all: pearson_par pearson verify

pearson_par: threadpool simd_analysis optimization4/main.cpp 
	$(CXX2) $(CXXFLAGS2) optimization4/main.cpp optimization4/ThreadPool.o optimization4/simd_analysis.o -o pearson_par -pthread

pearson: vector dataset analysis pearson.cpp 
	$(CXX) $(CXXFLAGS) pearson.cpp vector.o dataset.o analysis.o -o pearson

analysis: vector analysis.hpp analysis.cpp
	$(CXX) $(CXXFLAGS) -c analysis.cpp -o analysis.o

dataset: vector dataset.hpp dataset.cpp
	$(CXX) $(CXXFLAGS) -c dataset.cpp -o dataset.o

vector: vector.hpp vector.cpp
	$(CXX) $(CXXFLAGS) -c vector.cpp -o vector.o

simd_analysis: optimization4/simd_analysis.hpp optimization4/simd_analysis.cpp
	$(CXX2) $(CXXFLAGS2) -c optimization4/simd_analysis.cpp -o optimization4/simd_analysis.o

threadpool: optimization4/ThreadPool.hpp optimization4/ThreadPool.cpp
	$(CXX2) $(CXXFLAGS2) -c optimization4/ThreadPool.cpp -o optimization4/ThreadPool.o

verify: verify.c
	$(CC) verify.c -o verify

clean:
	rm -rf verify pearson pearson_par optimization3/*.ii optimization3/*.o optimization3/*.s optimization4/*.ii optimization4/*.o optimization4/*.s *.res *.args.0 *.ltrans_args *.ltrans.out *.ii *.s *.o *.dSYM 2> /dev/null
