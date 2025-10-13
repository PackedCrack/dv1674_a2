CXX2=g++-13
CXXFLAGS2=-std=c++20 -g -DNDEBUG -O0 -flto -march=native -pthread -save-temps=obj -fverbose-asm -masm=intel -Wall -Wunused -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Woverloaded-virtual -Wpedantic -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wuseless-cast

CXX=g++
CXXFLAGS=-std=c++17 -g -Wunused -Wall -Wunused

all: blur_par blur

blur_par: matrixopt ppmopt filtersopt optimization1/main.cpp 
	$(CXX2) $(CXXFLAGS2) optimization1/main.cpp optimization1/filtersopt.o optimization1/matrixopt.o optimization1/ppmopt.o -o blur_par -pthread

blur: matrix ppm filters blur.cpp
	$(CXX) $(CXXFLAGS) blur.cpp matrix.o ppm.o filters.o -o blur

filters: matrix filters.hpp filters.cpp
	$(CXX) $(CXXFLAGS) -c filters.cpp -o filters.o

matrix: matrix.hpp matrix.cpp
	$(CXX) $(CXXFLAGS) -c matrix.cpp -o matrix.o

ppm: ppm.hpp ppm.cpp
	$(CXX) $(CXXFLAGS) -c ppm.cpp -o ppm.o

filtersopt: matrixopt optimization1/filters.hpp optimization1/filters.cpp
	$(CXX2) $(CXXFLAGS2) -c optimization1/filters.cpp -o optimization1/filtersopt.o

matrixopt: optimization1/matrix.hpp optimization1/matrix.cpp
	$(CXX2) $(CXXFLAGS2) -c optimization1/matrix.cpp -o optimization1/matrixopt.o

ppmopt: optimization1/ppm.hpp optimization1/ppm.cpp
	$(CXX2) $(CXXFLAGS2) -c optimization1/ppm.cpp -o optimization1/ppmopt.o

clean:
	rm -rf verify pearson pearson_par optimization3/*.ii optimization3/*.o optimization3/*.s optimization4/*.ii optimization4/*.o optimization4/*.s optimization5/*.ii optimization5/*.o optimization5/*.s *.res *.args.0 *.ltrans_args *.ltrans.out *.ii *.s *.o *.dSYM 2> /dev/null
