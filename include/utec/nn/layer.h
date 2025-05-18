#pragma once
#include "../algebra/Tensor.h"
#include "../algebra/matmul.h"

namespace utec::neural_network {

template <typename T> class ILayer {
  public:
    virtual ~ILayer() = default;

    // Forward: recibe batch x features, devuelve batch x units
    virtual algebra::Tensor<T, 2> forward(const algebra::Tensor<T, 2> &x) = 0;

    // Backward: recibe gradiente de salida, devuelve gradiente de entrada
    virtual algebra::Tensor<T, 2>
    backward(const algebra::Tensor<T, 2> &grad) = 0;
};

} // namespace utec::neural_network
