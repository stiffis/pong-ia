#include "../include/utec/algebra/Tensor.h"
#include "../include/utec/algebra/matmul.h"
#include <array>
#include <chrono>
#include <iostream>
using namespace utec::algebra;

void benchmark_matmul() {
    std::array<std::size_t, 2> shapeA = {64, 64};
    std::array<std::size_t, 2> shapeB = {64, 64};

    // Crear las matrices A y B
    Tensor<int, 2> A(shapeA);
    Tensor<int, 2> B(shapeB);

    // Inicializar las matrices con algunos valores
    A.fill(1);
    B.fill(1);

    // Realizar el benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        auto result = matmul(A, B); // Ejecutar matmul 100 veces
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Benchmark (100 executions) took: " << duration.count()
              << " seconds" << std::endl;
}

int main() {
    benchmark_matmul();
    return 0;
}
