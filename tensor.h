#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>
namespace utec::algebra {
template <typename T, size_t Rank> class Tensor {
  private:
    std::array<size_t, Rank> shape_; // Dimensiones del tensor
    std::vector<T> data_;            // Datos del tensor
  public:
    static constexpr size_t rank = Rank; // Constante para el rango del tensor
    Tensor(const std::array<size_t, Rank> &shape) : shape_(shape) {
        size_t total_size = 1;
        for (size_t dim : shape) {
            total_size *= dim;
        }
        data_.resize(total_size);
    }

    template <typename... Dims>
    Tensor(Dims... dims)
        : Tensor(std::array<size_t, Rank>{static_cast<size_t>(dims)...}) {}

    // Método de acceso variadico (unico)
    template <typename... Idxs> T &operator()(Idxs... idxs) {
        static_assert(sizeof...(idxs) == Rank,
                      "Number of indices must match tensor rank");
        size_t index = 0;
        size_t multiplier = 1;
        size_t idx_array[] = {static_cast<size_t>(idxs)...};
        for (size_t i = 0; i < Rank; ++i) {
            index += idx_array[i] * multiplier;
            multiplier *= shape_[i];
        }
        return data_[index];
    }

    // Método de acceso variadico (const)
    template <typename... Idxs> const T &operator()(Idxs... idxs) const {
        static_assert(sizeof...(idxs) == Rank,
                      "Number of indices must match tensor rank");
        size_t index = 0;
        size_t multiplier = 1;
        size_t idx_array[] = {static_cast<size_t>(idxs)...};
        for (size_t i = 0; i < Rank; ++i) {
            index += idx_array[i] * multiplier;
            multiplier *= shape_[i];
        }
        return data_[index];
    }

    // Método para obtener la información de dimensiones
    const std::array<size_t, Rank> &shape() const noexcept { return shape_; }
    void reshape(const std::array<size_t, Rank> &new_shape) {
        size_t total_size = 1;
        for (size_t dim : new_shape) {
            total_size *= dim;
        }
        if (total_size != data_.size()) {
            throw std::invalid_argument("New shape must have the same number "
                                        "of elements as the original shape");
        }
        shape_ = new_shape;
    }
    
    // Método par modificación de forma usndo variadic template
    template <typename... Dims> void reshape(Dims... dims) {
        reshape(std::array<size_t, Rank>{static_cast<size_t>(dims)...});
    }

    // Método para modificacion masiva
    void fill(const T &value) noexcept {
        std::fill(data_.begin(), data_.end(), value);
    }

    // Operadores aritmeticos
    Tensor operator+(const Tensor &other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument(
                "Tensors must have the same shape for addition");
        }
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    Tensor operator-(const Tensor &other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument(
                "Tensors must have the same shape for subtraction");
        }
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }

Tensor operator*(const Tensor &other) const {
    // Asegúrese de que las dimensiones de las matrices sean compatibles
    if (rank != 2 || other.rank != 2) {
        throw std::invalid_argument("Matrix multiplication is only supported for 2D tensors");
    }

    // Verificar la multiplicación de matrices regular (sin broadcasting)
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication");
    }

    // Realización de la multiplicación de matrices
    Tensor result(std::array<size_t, 2>{shape_[0], other.shape_[1]});
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < other.shape_[1]; ++j) {
            result(i, j) = 0;
            for (size_t k = 0; k < shape_[1]; ++k) {
                result(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    }
    return result;
}

    Tensor operator*(const T &scalar) const {
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    // Utilidades

    // (solo para el caso de rank 2)
    Tensor transpose_2d() const {
        if (Rank != 2) {
            throw std::invalid_argument("Transpose is only supported for 2D "
                                        "tensors");
        }
        Tensor result(std::array<size_t, Rank>{shape_[1], shape_[0]});
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    // Operador para accder a los datos
    T& operator[](size_t index) {
        if (index >= data_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    const T& operator[](size_t index) const {
        if (index >= data_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }
};
} // namespace utec::algebra
