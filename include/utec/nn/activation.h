#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../algebra/Tensor.h"
#include "../algebra/matmul.h"
#include "layer.h"

namespace utec::neural_network {

template <typename T> class ReLU : public ILayer<T> {
    algebra::Tensor<T, 2> mask;

  public:
    algebra::Tensor<T, 2> forward(const algebra::Tensor<T, 2> &x) override {
        mask = x;
        for (size_t i = 0; i < mask.shape()[0] * mask.shape()[1]; ++i) {
            mask[i] = mask[i] > 0 ? 1 : 0;
        }
        algebra::Tensor<T, 2> out = x;
        for (size_t i = 0; i < out.shape()[0] * out.shape()[1]; ++i) {
            if (mask[i] == 0)
                out[i] = 0;
        }
        return out;
    }

    algebra::Tensor<T, 2> backward(const algebra::Tensor<T, 2> &grad) override {
        algebra::Tensor<T, 2> grad_input = grad;
        for (size_t i = 0; i < grad_input.shape()[0] * grad_input.shape()[1];
             ++i) {
            grad_input[i] *= mask[i];
        }
        return grad_input;
    }
};

} // namespace utec::neural_network
#endif // !ACTIVATION_H
