#pragma once
#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace utec::algebra {

template <typename T, std::size_t Rank> class Tensor {
    static_assert(Rank > 0, "Rank must be > 0");
    std::array<std::size_t, Rank> shape_{};
    std::vector<T> data_;

    static std::size_t total(const std::array<std::size_t, Rank> &s) {
        return std::accumulate(s.begin(), s.end(), std::size_t{1},
                               std::multiplies<>{});
    }

    template <std::size_t... Is>
    static std::array<std::size_t, Rank>
    make_shape_from_pack(std::index_sequence<Is...>, std::size_t dims...) {
        return {(static_cast<void>(Is), dims)...};
    }

    template <typename... Idxs> std::size_t index_of(Idxs... idxs) const {
        static_assert(sizeof...(Idxs) == Rank,
                      "number of indices must equal Rank");
        std::array<std::size_t, Rank> idx{static_cast<std::size_t>(idxs)...};
        std::size_t linear = 0, stride = 1;
        for (std::size_t i = Rank; i-- > 0;) {
            if (idx[i] >= shape_[i])
                throw std::out_of_range("index out of bounds");
            linear += idx[i] * stride;
            stride *= shape_[i];
        }
        return linear;
    }

  public:
    using shape_type = std::array<std::size_t, Rank>;

    Tensor() = default;

    explicit Tensor(const shape_type &shape)
        : shape_(shape), data_(total(shape)) {}

    template <typename... Dims,
              typename = std::enable_if_t<(sizeof...(Dims) == Rank)>>
    explicit Tensor(Dims... dims)
        : Tensor(shape_type{static_cast<std::size_t>(dims)...}) {}

    const shape_type &shape() const noexcept { return shape_; }

    template <typename... Idxs> T &operator()(Idxs... idxs) {
        return data_[index_of(idxs...)];
    }

    template <typename... Idxs> const T &operator()(Idxs... idxs) const {
        return data_[index_of(idxs...)];
    }

    T &operator[](std::size_t idx) { return data_.at(idx); }
    const T &operator[](std::size_t idx) const { return data_.at(idx); }

    void reshape(const shape_type &new_shape) {
        if (total(new_shape) != data_.size())
            throw std::invalid_argument("reshape changes total elements");
        shape_ = new_shape;
    }

    template <typename... Dims,
              typename = std::enable_if_t<(sizeof...(Dims) == Rank)>>
    void reshape(Dims... dims) {
        reshape(shape_type{static_cast<std::size_t>(dims)...});
    }

    void fill(const T &v) noexcept { std::fill(data_.begin(), data_.end(), v); }

  private:
    static shape_type broadcast_shape(const shape_type &a,
                                      const shape_type &b) {
        shape_type out{};
        for (std::size_t i = 0; i < Rank; ++i) {
            if (a[i] == b[i])
                out[i] = a[i];
            else if (a[i] == 1)
                out[i] = b[i];
            else if (b[i] == 1)
                out[i] = a[i];
            else
                throw std::invalid_argument(
                    "incompatible shapes for broadcasting");
        }
        return out;
    }

    std::size_t coord_to_linear(const shape_type &shp,
                                const shape_type &coord) const {
        std::size_t idx = 0, stride = 1;
        for (std::size_t i = Rank; i-- > 0;) {
            idx += coord[i] * stride;
            stride *= shp[i];
        }
        return idx;
    }

    template <typename F>
    static Tensor apply_elementwise(const Tensor &A, const Tensor &B, F op) {
        auto out_shape = broadcast_shape(A.shape_, B.shape_);
        Tensor C(out_shape);

        shape_type coord{};
        const std::size_t total_elems = total(out_shape);
        for (std::size_t linear = 0; linear < total_elems; ++linear) {
            // compute coordinate from linear index
            std::size_t rem = linear;
            for (std::size_t i = Rank; i-- > 0;) {
                coord[i] = rem % out_shape[i];
                rem /= out_shape[i];
            }
            shape_type coordA = coord;
            shape_type coordB = coord;
            for (std::size_t i = 0; i < Rank; ++i) {
                if (A.shape_[i] == 1)
                    coordA[i] = 0;
                if (B.shape_[i] == 1)
                    coordB[i] = 0;
            }
            const T &a = A.data_[A.coord_to_linear(A.shape_, coordA)];
            const T &b = B.data_[B.coord_to_linear(B.shape_, coordB)];
            C.data_[linear] = op(a, b);
        }
        return C;
    }

  public:
    Tensor operator+(const Tensor &other) const {
        return apply_elementwise(*this, other, std::plus<>{});
    }
    Tensor operator-(const Tensor &other) const {
        return apply_elementwise(*this, other, std::minus<>{});
    }
    Tensor operator*(const Tensor &other) const {
        return apply_elementwise(*this, other, std::multiplies<>{});
    }

    Tensor operator*(const T &scalar) const {
        Tensor C = *this;
        for (auto &v : C.data_)
            v = v * scalar;
        return C;
    }
    friend Tensor operator*(const T &scalar, const Tensor &t) {
        return t * scalar;
    }

    template <std::size_t R = Rank>
    std::enable_if_t<R == 2, Tensor> transpose_2d() const {
        Tensor out({shape_[1], shape_[0]});
        for (std::size_t i = 0; i < shape_[0]; ++i)
            for (std::size_t j = 0; j < shape_[1]; ++j)
                out(j, i) = (*this)(i, j);
        return out;
    }
};

} // namespace utec::algebra
