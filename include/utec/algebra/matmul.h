#pragma once
#include "Tensor.h"
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace utec::algebra {

template <typename T>
Tensor<T, 2> matmul(const Tensor<T, 2> &A, const Tensor<T, 2> &B) {
    if (A.shape()[1] != B.shape()[0]) {
        throw std::invalid_argument(
            "Matrices cannot be multiplied: incompatible shapes");
    }
    std::array<std::size_t, 2> result_shape = {A.shape()[0], B.shape()[1]};
    Tensor<T, 2> result(result_shape);

    for (std::size_t i = 0; i < A.shape()[0]; ++i) {
        for (std::size_t j = 0; j < B.shape()[1]; ++j) {
            result(i, j) = 0;
            for (std::size_t k = 0; k < A.shape()[1]; ++k) {
                result(i, j) += A(i, k) * B(k, j);
            }
        }
    }

    return result;
}

} // namespace utec::algebra
