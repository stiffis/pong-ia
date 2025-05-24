#include <iostream>
#include <vector>
#include <numeric>
#include <random> // Added for train_on_policy
#include <iomanip> // For std::fixed and std::setprecision

#include "../include/utec/agent/EnvGym.cpp" // Should ideally be .h and linked
#include "../include/utec/agent/EnvGym.h"
#include "../include/utec/agent/PongAgent.cpp" // Should ideally be .h and linked
#include "../include/utec/agent/PongAgent.h"
#include "../include/utec/agent/state.h" // Already included via PongAgent.h or EnvGym.h

#include "../include/utec/algebra/Tensor.h" // Already included via PongAgent.h or EnvGym.h

#include "../include/utec/nn/activation.h"
#include "../include/utec/nn/dense.h"
#include "../include/utec/nn/layer.h" // Not strictly needed if only using NN
#include "../include/utec/nn/loss.h"   // Not strictly needed if only using NN
#include "../include/utec/nn/neural_network.h"

// Copied from temp/train.cpp
template <typename T>
void train_on_policy(utec::neural_network::NeuralNetwork<T> &net,
                     utec::neural_network::PongAgent<T> &agent, // agent is not used in this version of train_on_policy
                     utec::neural_network::EnvGym &env, size_t episodes,
                     T gamma, T lr) {
    std::default_random_engine rng(std::random_device{}());
    std::uniform_real_distribution<T> dist(0, 1);

    for (size_t ep = 0; ep < episodes; ++ep) {
        utec::neural_network::State s = env.reset();
        bool done = false;

        while (!done) {
            // 1. Obtener vector Q para s
            utec::algebra::Tensor<T, 2> current_state_tensor(1, 5);
            current_state_tensor(0, 0) = static_cast<T>(s.ball_x);
            current_state_tensor(0, 1) = static_cast<T>(s.ball_y);
            current_state_tensor(0, 2) = static_cast<T>(s.ball_vx);
            current_state_tensor(0, 3) = static_cast<T>(s.ball_vy);
            current_state_tensor(0, 4) = static_cast<T>(s.paddle_y);
            utec::algebra::Tensor<T, 2> Qs = net.forward(current_state_tensor);

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
            utec::algebra::Tensor<T, 2> next_state_tensor(1, 5);
            next_state_tensor(0, 0) = static_cast<T>(s_next.ball_x);
            next_state_tensor(0, 1) = static_cast<T>(s_next.ball_y);
            next_state_tensor(0, 2) = static_cast<T>(s_next.ball_vx);
            next_state_tensor(0, 3) = static_cast<T>(s_next.ball_vy);
            next_state_tensor(0, 4) = static_cast<T>(s_next.paddle_y);
            utec::algebra::Tensor<T, 2> Qs_next = net.forward(next_state_tensor);

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

            // 7. Calcular gradiente de pérdida (Q(s,a) - target_value) para la acción tomada
            // El gradiente es 0 para las acciones no tomadas.
            utec::algebra::Tensor<T, 2> loss_grad(Qs.shape()); // Initialize with zeros
            loss_grad.fill(static_cast<T>(0));
            loss_grad(0, a_idx) = Qs(0, a_idx) - target(0, a_idx); // target(0, a_idx) is (r + gamma * max_q_next)

            net.backward(loss_grad);

            // 8. Optimizar parámetros
            net.optimize(lr);

            s = s_next;
        }

        if ((ep + 1) % 100 == 0) { // Print every 100 episodes for less verbose output
            std::cout << "Training Episode " << ep + 1 << " finished\n";
        }
    }
}


int main() {
    using namespace utec::neural_network;

    // Setup
    EnvGym env;
    NeuralNetwork<float> net;
    net.add_layer(std::make_unique<Dense<float>>(5, 10));
    net.add_layer(std::make_unique<ReLU<float>>());
    net.add_layer(std::make_unique<Dense<float>>(10, 3));
    PongAgent<float> agent(net);

    // Training Phase
    std::cout << "Starting training phase..." << std::endl;
    train_on_policy(net, agent, env, 1000, 0.99f, 0.01f); // agent is passed but not used by this train_on_policy
    std::cout << "Training phase completed." << std::endl;

    // Evaluation Phase
    std::cout << "Starting evaluation phase..." << std::endl;
    const int num_eval_episodes = 100;
    int successful_returns = 0;
    int total_potential_returns = 0;
    const int max_steps_per_episode = 300; // Max steps to prevent infinite loops

    for (int i = 0; i < num_eval_episodes; ++i) {
        State current_eval_state = env.reset();
        bool eval_done = false;
        bool return_counted_this_episode = false;
        bool ball_reached_paddle_zone = false;

        for (int step = 0; step < max_steps_per_episode; ++step) {
            int action = agent.act(current_eval_state); // Uses the trained net
            float eval_reward;
            current_eval_state = env.step(action, eval_reward, eval_done);

            if (current_eval_state.ball_x >= 0.95f) {
                ball_reached_paddle_zone = true;
            }

            if (eval_reward > 0 && !return_counted_this_episode) {
                successful_returns++;
                return_counted_this_episode = true; 
            }

            if (eval_done) {
                break;
            }
        }
        if (ball_reached_paddle_zone || eval_done) {
             total_potential_returns++;
        }
    }

    // Calculate and Report Success Rate
    float success_rate = 0.0f;
    if (total_potential_returns > 0) {
        success_rate = static_cast<float>(successful_returns) / total_potential_returns;
    }
    std::cout << "Evaluation Results: " << successful_returns << "/" << total_potential_returns << " successful returns." << std::endl;
    std::cout << std::fixed << std::setprecision(2); // For consistent percentage output
    std::cout << "Success rate: " << success_rate * 100.0f << "%" << std::endl;

    // Assertion for Epic 3
    bool test_passed = (success_rate >= 0.70f);
    std::cout << "Epic 3 Test (>=70% returns): " << (test_passed ? "Passed" : "Failed") << std::endl;
    
    return test_passed ? 0 : 1;
}
