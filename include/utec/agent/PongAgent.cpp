#ifndef PONG_AGENT_CPP
#define PONG_AGENT_CPP

// TODO: DELETE SMART POINTERS
#include "PongAgent.h"
#include <algorithm>

namespace utec::neural_network {

template <typename T>
PongAgent<T>::PongAgent(std::unique_ptr<ILayer<T>> model)
    : model_(std::move(model)) {}

template <typename T> int PongAgent<T>::act(const State &state) {
    // Convertir State a tensor 1x3: [ball_x, ball_y, paddle_y]
    algebra::Tensor<T, 2> input(1, 3);
    input(0, 0) = static_cast<T>(state.ball_x);
    input(0, 1) = static_cast<T>(state.ball_y);
    input(0, 2) = static_cast<T>(state.paddle_y);

    auto output = model_->forward(input);

    // Salida es vector 1x3 con valores para acciones -1,0,1 respectivamente
    // Escoger índice máximo
    size_t max_idx = 0;
    T max_val = output(0, 0);
    for (size_t i = 1; i < 3; ++i) {
        if (output(0, i) > max_val) {
            max_val = output(0, i);
            max_idx = i;
        }
    }

    // Mapear índice a acción
    // 0 -> -1 (arriba), 1 -> 0 (quieto), 2 -> +1 (abajo)
    static constexpr int actions[] = {-1, 0, 1};
    return actions[max_idx];
}

// Necesario para compilar plantilla en archivo separado
template class PongAgent<float>;
template class PongAgent<double>;

} // namespace utec::neural_network
#endif // !PONG_AGENT_CPP
