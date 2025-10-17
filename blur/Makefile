CXX2=g++-13
CXXFLAGS2=-std=c++20 -g -O3 -flto -march=native -pthread -save-temps=obj -fverbose-asm -masm=intel -Wall -Wunused -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Woverloaded-virtual -Wpedantic -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wuseless-cast

CXX=g++
CXXFLAGS=-std=c++17 -g -Wunused -Wall -Wunused

all: blur_par blur

blur_par: gaussian threadpool optimization2/main.cpp 
	$(CXX2) $(CXXFLAGS2) optimization2/main.cpp optimization2/ThreadPool.o optimization2/gaussian.o -o blur_par -pthread

blur: matrix ppm filters blur.cpp
	$(CXX) $(CXXFLAGS) blur.cpp matrix.o ppm.o filters.o -o blur

filters: matrix filters.hpp filters.cpp
	$(CXX) $(CXXFLAGS) -c filters.cpp -o filters.o

matrix: matrix.hpp matrix.cpp
	$(CXX) $(CXXFLAGS) -c matrix.cpp -o matrix.o

ppm: ppm.hpp ppm.cpp
	$(CXX) $(CXXFLAGS) -c ppm.cpp -o ppm.o

threadpool: optimization2/ThreadPool.hpp optimization2/ThreadPool.cpp
	$(CXX2) $(CXXFLAGS2) -c optimization2/ThreadPool.cpp -o optimization2/ThreadPool.o

gaussian: optimization2/gaussian.hpp optimization2/gaussian.cpp
	$(CXX2) $(CXXFLAGS2) -c optimization2/gaussian.cpp -o optimization2/gaussian.o

clean:
	rm -rf verify pearson pearson_par optimization1/*.ii optimization1/*.o optimization1/*.s optimization2/*.ii optimization2/*.o optimization2/*.s optimization3/*.ii optimization3/*.o optimization3/*.s optimization4/*.ii optimization4/*.o optimization4/*.s optimization5/*.ii optimization5/*.o optimization5/*.s *.res *.args.0 *.ltrans_args *.ltrans.out *.ii *.s *.o *.dSYM 2> /dev/null

