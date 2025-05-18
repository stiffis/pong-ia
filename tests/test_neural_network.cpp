#include "../include/utec/algebra/Tensor.h"
#include "../include/utec/nn/activation.h"
#include "../include/utec/nn/dense.h"
#include "../include/utec/nn/loss.h"
#include "../include/utec/nn/neural_network.h"

#include <chrono>
#include <iostream>

using Clock = std::chrono::steady_clock;

void test_relu() {
    using T = float;
    auto start = Clock::now();

    utec::algebra::Tensor<T, 2> M(2, 2);
    M(0, 0) = -1;
    M(0, 1) = 2;
    M(1, 0) = 0;
    M(1, 1) = -3;

    utec::neural_network::ReLU<T> relu;
    auto R = relu.forward(M);

    bool res1 = (R(0, 1) == 2);

    utec::algebra::Tensor<T, 2> GR(2, 2);
    GR.fill(1.0f);
    auto dM = relu.backward(GR);

    bool res2 = (dM(0, 0) == 0 && dM(1, 1) == 0);

    auto end = Clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "test_relu forward: " << (res1 ? "Passed" : "Failed") << "\n";
    std::cout << "test_relu backward: " << (res2 ? "Passed" : "Failed") << "\n";
    std::cout << "test_relu time: " << elapsed.count() << " ms\n\n";
}

void test_mse_loss() {
    using T = double;
    auto start = Clock::now();

    utec::algebra::Tensor<T, 2> P(1, 2);
    P(0, 0) = 1;
    P(0, 1) = 2;

    utec::algebra::Tensor<T, 2> Tgt(1, 2);
    Tgt(0, 0) = 0;
    Tgt(0, 1) = 4;

    utec::neural_network::MSELoss<T> loss;
    T L = loss.forward(P, Tgt);

    bool res1 = (L == 2.5);

    auto dP = loss.backward();

    bool res2 = (dP(0, 1) < 0);

    auto end = Clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "test_mse_loss forward: " << (res1 ? "Passed" : "Failed")
              << "\n";
    std::cout << "test_mse_loss backward: " << (res2 ? "Passed" : "Failed")
              << "\n";
    std::cout << "test_mse_loss time: " << elapsed.count() << " ms\n\n";
}

void test_xor_training() {
    using T = float;
    auto start = Clock::now();

    utec::algebra::Tensor<T, 2> X(4, 2);
    X(0, 0) = 0;
    X(0, 1) = 0;
    X(1, 0) = 0;
    X(1, 1) = 1;
    X(2, 0) = 1;
    X(2, 1) = 0;
    X(3, 0) = 1;
    X(3, 1) = 1;

    utec::algebra::Tensor<T, 2> Y(4, 1);
    Y(0, 0) = 0;
    Y(1, 0) = 1;
    Y(2, 0) = 1;
    Y(3, 0) = 0;

    utec::neural_network::NeuralNetwork<T> net;
    net.add_layer(std::make_unique<utec::neural_network::Dense<T>>(2, 4));
    net.add_layer(std::make_unique<utec::neural_network::ReLU<T>>());
    net.add_layer(std::make_unique<utec::neural_network::Dense<T>>(4, 1));

    float final_loss = net.train(X, Y, 1000, 0.1f);

    bool res = (final_loss < 0.1f);

    auto end = Clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "test_xor_training final loss: " << final_loss << "\n";
    std::cout << "test_xor_training: " << (res ? "Passed" : "Failed") << "\n";
    std::cout << "test_xor_training time: " << elapsed.count() << " ms\n\n";
}

int main() {
    test_relu();
    test_mse_loss();
    test_xor_training();
    return 0;
}
