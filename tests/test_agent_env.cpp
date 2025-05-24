#include <iostream>

#include "../include/utec/agent/EnvGym.cpp"
#include "../include/utec/agent/EnvGym.h"
#include "../include/utec/agent/PongAgent.cpp"
#include "../include/utec/agent/PongAgent.h"

#include "../include/utec/nn/activation.h"
#include "../include/utec/nn/dense.h"
#include "../include/utec/nn/layer.h"
#include "../include/utec/nn/loss.h"
#include "../include/utec/nn/neural_network.h"

int main() {
    using namespace utec::neural_network;

    NeuralNetwork<float> net;
    net.add_layer(std::make_unique<Dense<float>>(5, 5));
    net.add_layer(std::make_unique<ReLU<float>>());
    net.add_layer(std::make_unique<Dense<float>>(5, 3));

    // Aquí idealmente cargarías pesos o entrenarías la red

    // Usamos solo la última capa para el agente (ejemplo simple)
    auto model = std::make_unique<Dense<float>>(5, 3);
    PongAgent<float> agent(std::move(model));

    EnvGym env;
    float reward;
    bool done;

    State state = env.reset();

    for (int t = 0; t < 20; ++t) {
        int action = agent.act(state);
        state = env.step(action, reward, done);

        std::cout << "Step " << t << ": action=" << action
                  << ", ball_x=" << state.ball_x << ", ball_y=" << state.ball_y
                  << ", ball_vx=" << state.ball_vx << ", ball_vy=" << state.ball_vy
                  << ", paddle_y=" << state.paddle_y << ", reward=" << reward
                  << ", done=" << done << "\n";

        if (done)
            break;
    }
    return 0;
}
