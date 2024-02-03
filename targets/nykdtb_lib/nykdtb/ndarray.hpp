#ifndef NYKDTB_NDARRAY_HPP
#define NYKDTB_NDARRAY_HPP

#include <variant>

#include "nykdtb/psvector.hpp"
#include "nykdtb/types.hpp"
#include "nykdtb/utils.hpp"

namespace nykdtb {

template<typename T>
concept NDArrayLike = requires(T a) {
    { a[Index{0}] } -> std::same_as<T>;
};

template<typename NDT>
class NDArraySlice;

template<typename T, typename Params>
class NDArrayBase {
public:
    using Type       = T;
    using Shape      = PSVec<Size, Params::SHAPE_STACK_SIZE>;
    using Strides    = PSVec<Size, Params::SHAPE_STACK_SIZE>;
    using Position   = PSVec<Index, Params::SHAPE_STACK_SIZE>;
    using Storage    = PSVec<T, Params::STACK_SIZE>;
    using SliceShape = PSVec<IndexRange, Params::SHAPE_STACK_SIZE>;
    using Parameters = Params;

    NYKDTB_DEFINE_EXCEPTION_CLASS(ShapeDoesNotMatchSize, LogicException)

public:
    NDArrayBase() = default;
    NDArrayBase(Storage input)
        : m_storage(mmove(input)),
          m_shape({static_cast<Size>(m_storage.size())}),
          m_strides(calculateStrides(m_shape)) {}
    NDArrayBase(Storage input, Shape shape)
        : m_storage(mmove(input)), m_shape(mmove(shape)), m_strides(calculateStrides(m_shape)) {
        if (calculateSize(m_shape) != size()) {
            throw ShapeDoesNotMatchSize();
        }
    }

    NDArrayBase(NDArrayBase&&)            = default;
    NDArrayBase& operator=(NDArrayBase&&) = default;

    NDArrayBase clone() const { return {*this}; }

    bool empty() const { return m_storage.empty(); }
    const Shape& shape() const { return m_shape; }
    const Strides& strides() const { return m_strides; }
    Size size() const { return static_cast<Size>(m_storage.size()); }

    T& operator[](Index index) { return m_storage[index]; }
    const T& operator[](Index index) const { return m_storage[index]; }
    T& operator[](const Position& pos) { return this->operator[](calculateRawIndexUnchecked(m_strides, pos)); }
    const T& operator[](const Position& pos) const {
        return this->operator[](calculateRawIndexUnchecked(m_strides, pos));
    }
    NDArraySlice<NDArrayBase> slice(SliceShape shape) { return {*this, mmove(shape)}; }
    NDArraySlice<const NDArrayBase> slice(SliceShape shape) const { return {*this, mmove(shape)}; }

public:
    static constexpr Size calculateSize(const Shape& shape) {
        if (shape.empty()) {
            return 0;
        }

        Size result = 1;
        for (auto size : shape) {
            result *= size;
        }
        return result;
    }

    static constexpr Strides calculateStrides(const Shape& shape) {
        Strides strides(shape);
        strides.last() = 1;

        for (Index i = strides.size() - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }

        return mmove(strides);
    }

    static constexpr Index calculateRawIndexUnchecked(const Strides& strides, const Position& position) {
        Index result = 0;
        for (Index i = 0; i < strides.size(); ++i) {
            result += strides[i] * position[i];
        }
        return result;
    }

private:
    NDArrayBase(const NDArrayBase&)            = default;
    NDArrayBase& operator=(const NDArrayBase&) = default;

private:
    Storage m_storage;
    Shape m_shape;
    Strides m_strides;
};

template<typename NDT>
class NDArraySlice {
public:
    using NDArray                      = NDT;
    using SliceShape                   = typename NDArray::SliceShape;
    using Shape                        = typename NDArray::Shape;
    using Strides                      = typename NDArray::Strides;
    using Position                     = typename NDArray::Position;
    static constexpr bool isConstArray = std::is_const_v<NDArray>;
    using Type = std::conditional<isConstArray, std::add_const_t<typename NDArray::Type>, typename NDArray::Type>;

    NYKDTB_DEFINE_EXCEPTION_CLASS(InvalidSliceShape, LogicException)

public:
    NDArraySlice(NDArray& array, SliceShape shape)
        : m_ndarray{array},
          m_sliceShape{mmove(shape)},
          m_shape(calculateShape(m_ndarray.shape(), m_sliceShape), m_strides(NDArray::calculateStrides(m_shape))) {}

    bool empty() const { return NDArray::calculateSize(m_shape); }
    const Shape& shape() const { return m_shape; }
    const SliceShape& sliceShape() const { return m_sliceShape; }
    const Strides& strides() const { return m_strides; }
    Size size() const { return NDArray::calculateSize(m_shape); }

    Type& operator[](const Index index) { return m_ndarray[calculateRawIndexFromSliceIndexUnchecked(index)]; }
    const Type& operator[](const Index index) const {
        return m_ndarray[calculateRawIndexFromSliceIndexUnchecked(index)];
    }

    Type& operator[](const Position& position) { return m_ndarray[calculateRawIndexFromPositionUnchecked(position)]; }
    const Type& operator[](const Position& position) const {
        return m_ndarray[calculateRawIndexFromPositionUnchecked(position)];
    }

    Index calculateRawIndexFromSliceIndexUnchecked(Index index) const {
        return calculateRawIndexFromSliceIndexUnchecked(m_ndarray.strides(), m_strides, m_sliceShape, index);
    }

    Index calculateRawIndexFromPositionUnchecked(const Position& position) const {
        return calculateRawIndexFromPositionUnchecked(m_ndarray.strides(), m_sliceShape, position);
    }

    static constexpr Shape calculateShape(const Shape& original, const SliceShape& sliceShape) {
        if (original.size() != sliceShape.size()) {
            throw InvalidSliceShape();
        }

        Shape result(original);

        for (Index i = 0; i < original.size(); ++i) {
            result[i] = sliceShape[i].effectiveSize(original[i]);
        }

        return mmove(result);
    }

    static constexpr Index calculateRawIndexFromPositionUnchecked(const Strides& arrayStrides,
                                                                  const SliceShape& sliceShape,
                                                                  const Position& position) {
        Index result = 0;
        for (Index i = 0; i < arrayStrides.size(); ++i) {
            result += arrayStrides[i] * (sliceShape[i].begin() + position[i]);
        }
        return result;
    }

    static constexpr Index calculateRawIndexFromSliceIndexUnchecked(const Strides& arrayStrides,
                                                                    const Strides& sliceStrides,
                                                                    const SliceShape& sliceShape,
                                                                    Index index) {
        Index result = 0;
        for (Index i = 0; i < arrayStrides.size(); ++i) {
            const auto sliceStride = sliceStrides[i];
            const auto arrayStride = arrayStrides[i];
            const auto arrayBegin  = sliceShape[i].begin();
            const Index dimStrides = index / sliceStride;
            index                  = index % sliceStride;
            result += (dimStrides + arrayBegin) * arrayStride;
        }
        return result;
    }

private:
    NDArray& m_ndarray;
    const SliceShape m_sliceShape;
    const Shape m_shape;
    const Strides m_strides;
};

struct DefaultNDArrayParams {
    static constexpr Size STACK_SIZE       = 8;
    static constexpr Size SHAPE_STACK_SIZE = 4;
};

template<typename T>
using NDArray = NDArrayBase<T, DefaultNDArrayParams>;

}  // namespace nykdtb

#endif