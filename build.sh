#!/usr/bin/env bash


CXX="g++"                         # o clang++
CXXFLAGS="-std=c++20 -O3 -Wall -Wextra -pedantic \
          -fsanitize=undefined,address"

$CXX $CXXFLAGS \
     $(find src   -name '*.cpp') \
     $(find tests -name '*_test.cpp') \
     -o pong_ai


