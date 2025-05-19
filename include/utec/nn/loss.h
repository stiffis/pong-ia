#ifndef LOSS_H
#define LOSS_H

#include "../algebra/Tensor.h"
#include "../algebra/matmul.h"

namespace utec::neural_network {

template <typename T> class MSELoss {
    algebra::Tensor<T, 2> last_pred, last_target;

  public:
    // Forward calcula la pérdida media
    T forward(const algebra::Tensor<T, 2> &pred,
              const algebra::Tensor<T, 2> &target) {
        last_pred = pred;
        last_target = target;
        T sum = 0;
        size_t n = pred.shape()[0] * pred.shape()[1];
        for (size_t i = 0; i < n; ++i) {
            T diff = pred[i] - target[i];
            sum += diff * diff;
        }
        return sum / n;
    }

    // Backward calcula gradiente dL/dpred
    algebra::Tensor<T, 2> backward() {
        algebra::Tensor<T, 2> grad(last_pred.shape());
        size_t n = last_pred.shape()[0] * last_pred.shape()[1];
        for (size_t i = 0; i < n; ++i) {
            grad[i] = 2 * (last_pred[i] - last_target[i]) / n;
        }
        return grad;
    }
};

} // namespace utec::neural_network
#endif // !LOSS_H
