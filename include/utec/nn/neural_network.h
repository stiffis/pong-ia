#pragma once
#include "../algebra/Tensor.h"
#include "../algebra/matmul.h"
#include "activation.h"
#include "dense.h"
#include "layer.h"
#include "loss.h"
#include <memory>
#include <vector>

namespace utec::neural_network {

template <typename T> class NeuralNetwork {
    std::vector<std::unique_ptr<ILayer<T>>> layers;
    MSELoss<T> criterion;

  public:
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers.push_back(std::move(layer));
    }

    algebra::Tensor<T, 2> forward(const algebra::Tensor<T, 2> &x) {
        algebra::Tensor<T, 2> out = x;
        for (auto &layer : layers) {
            out = layer->forward(out);
        }
        return out;
    }

    void backward(const algebra::Tensor<T, 2> &grad) {
        algebra::Tensor<T, 2> grad_current = grad;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad_current = (*it)->backward(grad_current);
        }
    }

    void optimize(T lr) {
        for (auto &layer : layers) {
            // Solo actualizamos capas Dense (tendrás que castear o usar una
            // interfaz)
            if (auto dense = dynamic_cast<Dense<T> *>(layer.get())) {
                dense->update_params(lr);
            }
        }
    }

    // Entrena la red con X, Y durante epochs
    T train(const algebra::Tensor<T, 2> &X, const algebra::Tensor<T, 2> &Y,
            size_t epochs, T lr) {
        T loss_val = 0;
        for (size_t e = 0; e < epochs; ++e) {
            auto pred = forward(X);
            loss_val = criterion.forward(pred, Y);
            auto grad = criterion.backward();
            backward(grad);
            optimize(lr);
        }
        return loss_val;
    }
};

} // namespace utec::neural_network
