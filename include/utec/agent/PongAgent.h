#ifndef PONG_AGENT_H
#define PONG_AGENT_H

#include "../algebra/Tensor.h"
#include "../algebra/matmul.h"
#include "../nn/activation.h"
#include "../nn/dense.h"
#include "../nn/layer.h"
#include "../nn/loss.h"
#include "../nn/neural_network.h"
#include "state.h"
#include <memory>

namespace utec::neural_network {

template <typename T> class PongAgent {
  public:
    // Construye con modelo entrenado (ILayer<T> con forward)
    explicit PongAgent(std::unique_ptr<ILayer<T>> model);

    // Decide acción: -1 (arriba), 0 (quieto), +1 (abajo)
    int act(const State &state);

  private:
    std::unique_ptr<ILayer<T>> model_;
};

} // namespace utec::neural_network
#endif // !PONG_AGENT_H
