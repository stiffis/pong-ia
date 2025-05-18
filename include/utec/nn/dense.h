#pragma once
#include "../algebra/Tensor.h"
#include "../algebra/matmul.h"
#include "layer.h"
#include <random>
namespace utec::neural_network {

template <typename T> class Dense : public ILayer<T> {
    algebra::Tensor<T, 2> W, dW;  // pesos [input, output] y su gradiente
    algebra::Tensor<T, 1> b, db;  // bias [output] y su gradiente
    algebra::Tensor<T, 2> last_x; // cache para backward

  public:
    Dense(size_t in_feats, size_t out_feats)
        : W(in_feats, out_feats), dW(in_feats, out_feats), b(out_feats),
          db(out_feats) {
        // Inicialización simple: pesos pequeños aleatorios
        std::default_random_engine gen;
        std::uniform_real_distribution<T> dist(-0.1, 0.1);
        for (size_t i = 0; i < W.shape()[0] * W.shape()[1]; ++i)
            W[i] = dist(gen);
        for (size_t i = 0; i < b.shape()[0]; ++i)
            b[i] = 0;
    }

    algebra::Tensor<T, 2> forward(const algebra::Tensor<T, 2> &x) override {
        last_x = x;
        // output = x * W + b (broadcast b)
        algebra::Tensor<T, 2> out = matmul(x, W);
        // broadcast sum b per row
        for (size_t i = 0; i < out.shape()[0]; ++i)
            for (size_t j = 0; j < out.shape()[1]; ++j)
                out(i, j) += b[j];
        return out;
    }

    algebra::Tensor<T, 2> backward(const algebra::Tensor<T, 2> &grad) override {
        // grad: dL/dout
        // calcular gradientes para pesos y bias
        // dW = x^T * grad
        dW = matmul(last_x.transpose_2d(), grad);

        // db = sum rows grad
        db.fill(0);
        for (size_t j = 0; j < grad.shape()[1]; ++j) {
            T sum = 0;
            for (size_t i = 0; i < grad.shape()[0]; ++i)
                sum += grad(i, j);
            db[j] = sum;
        }

        // grad entrada = grad * W^T
        return matmul(grad, W.transpose_2d());
    }

    void update_params(T lr) {
        for (size_t i = 0; i < W.shape()[0] * W.shape()[1]; ++i)
            W[i] -= lr * dW[i];
        for (size_t i = 0; i < b.shape()[0]; ++i)
            b[i] -= lr * db[i];
    }
};

} // namespace utec::neural_network
