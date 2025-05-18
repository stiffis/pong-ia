#include "../include/utec/agent/EnvGym.cpp"
#include "../include/utec/agent/EnvGym.h"
#include "../include/utec/agent/PongAgent.cpp"
#include "../include/utec/agent/PongAgent.h"
#include "../include/utec/agent/state.h"

#include "../include/utec/algebra/Tensor.h"
#include "../include/utec/algebra/matmul.h"

#include "../include/utec/nn/activation.h"
#include "../include/utec/nn/dense.h"
#include "../include/utec/nn/layer.h"
#include "../include/utec/nn/loss.h"
#include "../include/utec/nn/neural_network.h"
#include <iostream>

template <typename T>
void train_on_policy(utec::neural_network::NeuralNetwork<T> &net,
                     utec::neural_network::PongAgent<T> &agent,
                     utec::neural_network::EnvGym &env, size_t episodes,
                     T gamma, T lr) {
    std::default_random_engine rng(std::random_device{}());
    std::uniform_real_distribution<T> dist(0, 1);

    for (size_t ep = 0; ep < episodes; ++ep) {
        utec::neural_network::State s = env.reset();
        bool done = false;

        while (!done) {
            // 1. Obtener vector Q para s
            utec::algebra::Tensor<T, 2> Qs =
                net.forward(utec::algebra::Tensor<T, 2>({1, 3}) = {
                                s.ball_x, s.ball_y, s.paddle_y});

            // 2. Elegir acción (epsilon-greedy para explorar)
            T epsilon = 0.1; // probabilidad de explorar
            int action;
            if (dist(rng) < epsilon) {
                int random_idx = rng() % 3;
                static constexpr int acts[] = {-1, 0, 1};
                action = acts[random_idx];
            } else {
                // Explota acción mejor Q
                size_t max_i = 0;
                for (size_t i = 1; i < 3; ++i)
                    if (Qs(0, i) > Qs(0, max_i))
                        max_i = i;
                static constexpr int acts[] = {-1, 0, 1};
                action = acts[max_i];
            }

            // 3. Ejecutar acción
            float reward;
            utec::neural_network::State s_next = env.step(action, reward, done);

            // 4. Obtener Q(s')
            utec::algebra::Tensor<T, 2> Qs_next =
                net.forward(utec::algebra::Tensor<T, 2>({1, 3}) = {
                                s_next.ball_x, s_next.ball_y, s_next.paddle_y});

            // 5. Calcular target vector igual a Q(s)
            utec::algebra::Tensor<T, 2> target = Qs;

            // Mapear acción a índice
            static constexpr int acts[] = {-1, 0, 1};
            size_t a_idx = 0;
            for (; a_idx < 3; ++a_idx)
                if (acts[a_idx] == action)
                    break;

            // 6. target[a] = r + gamma * max Q(s')
            T max_q_next = Qs_next(0, 0);
            for (size_t i = 1; i < 3; ++i)
                if (Qs_next(0, i) > max_q_next)
                    max_q_next = Qs_next(0, i);

            target(0, a_idx) = reward + gamma * max_q_next;

            // 7. Calcular gradiente de pérdida (Q - target) y hacer backward
            utec::algebra::Tensor<T, 2> loss_grad = Qs - target;

            net.backward(loss_grad);

            // 8. Optimizar parámetros
            net.optimize(lr);

            s = s_next;
        }

        if ((ep + 1) % 10 == 0) {
            std::cout << "Episode " << ep + 1 << " finished\n";
        }
    }
}

int main() {
    using namespace utec::neural_network;

    NeuralNetwork<float> net;
    net.add_layer(std::make_unique<Dense<float>>(3, 10));
    net.add_layer(std::make_unique<ReLU<float>>());
    net.add_layer(std::make_unique<Dense<float>>(10, 3));

    PongAgent<float> agent(std::make_unique<Dense<float>>(
        3, 3)); // o pasar net como modelo si quieres

    EnvGym env;

    train_on_policy(net, agent, env, 1000, 0.99f, 0.01f);

    std::cout << "Entrenamiento finalizado\n";
    return 0;
}
