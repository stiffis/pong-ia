#include "../include/utec/algebra/Tensor.h"
#include "../include/utec/algebra/matmul.h"
#include <cstring>
#include <iostream>

void test_matmul() {
    // Crear dos tensores 2x3 y 3x2
    utec::algebra::Tensor<int, 2> A(2, 3);
    utec::algebra::Tensor<int, 2> B(3, 2);

    // Rellenar los tensores con algunos valores
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;
    B(0, 0) = 7;
    B(0, 1) = 8;
    B(1, 0) = 9;
    B(1, 1) = 10;
    B(2, 0) = 11;
    B(2, 1) = 12;

    // Realizar la multiplicación
    auto result = utec::algebra::matmul(A, B);

    // Comprobar los valores de la multiplicación
    bool res = (result(0, 0) == 58 && result(0, 1) == 64 &&
                result(1, 0) == 139 && result(1, 1) == 154);

    // Mostrar el resultado
    std::cout << "test_matmul: " << (res ? "Passed" : "Failed") << std::endl;
}

int main() {
    test_matmul();
    return 0;
}
