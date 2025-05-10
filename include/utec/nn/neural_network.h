#pragma once
#include "../utec/algebra/Tensor.h"
#include "../utec/algebra/matmul.h"
#include <array>
#include <cmath>
#include <iostream>
#include <random>

using namespace utec::algebra;
class NeuralNetwork {
  public:
    NeuralNetwork(std::vector<int> layer_sizes) {
        layers = layer_sizes.size();
        sizes = layer_sizes;

        // Inicializar los pesos y sesgos
        for (int i = 1; i < layers; ++i) {
            weights.push_back(random_matrix(sizes[i], sizes[i - 1]));
            biases.push_back(random_vector(sizes[i]));
        }
    }

    // Propagación hacia adelante
    Tensor<float, 2> forward(const Tensor<float, 2> &input) {
        Tensor<float, 2> current_output = input;
        outputs.clear();
        outputs.push_back(current_output);

        // Cálculo de las salidas de cada capa
        for (int i = 0; i < layers - 1; ++i) {
            current_output = activation_function(
                matmul(weights[i], current_output) + biases[i]);
            outputs.push_back(current_output);
        }

        return current_output;
    }

    // Backpropagation (actualiza pesos y sesgos)
    void backward(const Tensor<float, 2> &input, const Tensor<float, 2> &target,
                  float learning_rate) {
        // Calcular los gradientes
        Tensor<float, 2> output_error = loss_derivative(outputs.back(), target);

        // Propagar hacia atrás
        for (int i = layers - 2; i >= 0; --i) {
            // Calcular gradientes para la capa actual
            Tensor<float, 2> layer_error =
                output_error * activation_derivative(outputs[i + 1]);

            // Actualizar los pesos y los sesgos
            weights[i] = weights[i] -
                         learning_rate * outer_product(layer_error, outputs[i]);
            biases[i] = biases[i] - learning_rate * layer_error;
        }
    }

    // Función de activación (ReLU)
    Tensor<float, 2> activation_function(const Tensor<float, 2> &input) {
        Tensor<float, 2> result = input;
        for (size_t i = 0; i < input.shape()[0]; ++i) {
            for (size_t j = 0; j < input.shape()[1]; ++j) {
                result(i, j) = std::max(0.0f, input(i, j)); // ReLU
            }
        }
        return result;
    }

    // Derivada de la función de activación (ReLU)
    Tensor<float, 2> activation_derivative(const Tensor<float, 2> &input) {
        Tensor<float, 2> result = input;
        for (size_t i = 0; i < input.shape()[0]; ++i) {
            for (size_t j = 0; j < input.shape()[1]; ++j) {
                result(i, j) =
                    (input(i, j) > 0) ? 1.0f : 0.0f; // Derivada de ReLU
            }
        }
        return result;
    }

  private:
    int layers;
    std::vector<int> sizes;
    std::vector<Tensor<float, 2>> weights; // pesos de cada capa
    std::vector<Tensor<float, 2>> biases;  // sesgos de cada capa
    std::vector<Tensor<float, 2>> outputs; // salidas de cada capa

    // Métodos de utilidad
    Tensor<float, 2> random_vector(int size) {
        Tensor<float, 2> vec(size, 1);
        std::random_device rd;
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < size; ++i) {
            vec(i, 0) = dist(rd);
        }
        return vec;
    }

    Tensor<float, 2> random_matrix(int rows, int cols) {
        Tensor<float, 2> mat(rows, cols);
        std::random_device rd;
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                mat(i, j) = dist(rd);
            }
        }
        return mat;
    }

    Tensor<float, 2> loss_derivative(const Tensor<float, 2> &output,
                                     const Tensor<float, 2> &target) {
        Tensor<float, 2> result = output;
        for (size_t i = 0; i < result.shape()[0]; ++i) {
            for (size_t j = 0; j < result.shape()[1]; ++j) {
                result(i, j) =
                    result(i, j) - target(i, j); // Error cuadrático medio
            }
        }
        return result;
    }

    // Producto externo entre dos tensores
    Tensor<float, 2> outer_product(const Tensor<float, 2> &v1,
                                   const Tensor<float, 2> &v2) {
        Tensor<float, 2> result(v1.shape()[0], v2.shape()[1]);
        for (size_t i = 0; i < v1.shape()[0]; ++i) {
            for (size_t j = 0; j < v2.shape()[1]; ++j) {
                result(i, j) = v1(i, 0) * v2(0, j);
            }
        }
        return result;
    }
};
