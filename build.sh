#!/usr/bin/env bash


CXX="g++"                         # o clang++
CXXFLAGS="-std=c++20 -O3 -Wall -Wextra -pedantic \
          -fsanitize=undefined,address"

echo "Building training executable..."
$CXX $CXXFLAGS temp/train.cpp -o pong_ai_train

echo "Building tests..."
mkdir -p build/tests
for test_file in $(find tests -name 'test_*.cpp'); do
    base_name=$(basename ${test_file} .cpp)
    echo "Compiling ${test_file} to build/tests/${base_name}_run"
    $CXX $CXXFLAGS ${test_file} -o build/tests/${base_name}_run
done


