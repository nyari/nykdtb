#ifndef NYKDTB_NDARRAY_HPP
#define NYKDTB_NDARRAY_HPP

#include <variant>

#include "nykdtb/psvector.hpp"
#include "nykdtb/types.hpp"
#include "nykdtb/utils.hpp"

namespace nykdtb {

struct IndexRange {
    enum Endpoint { E };
    using Element = std::variant<Index, Endpoint>;

    constexpr IndexRange()
        : begin(0), end(0) {}

    static constexpr IndexRange e2e() { return IndexRange{E, E}; }
    static constexpr IndexRange none() { return IndexRange{0, 0}; }
    static constexpr IndexRange until(Element end) { return IndexRange{E, end}; }
    static constexpr IndexRange after(Element begin) { return IndexRange{begin, E}; }
    static constexpr IndexRange between(Element begin, Element end) { return IndexRange{begin, end}; }

    Element begin;
    Element end;

private:
    constexpr IndexRange(Element _begin, Element _end)
        : begin{_begin}, end{_end} {}
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
        : m_storage(mmove(input)), m_shape({static_cast<Size>(m_storage.size())}) {}
    NDArrayBase(Storage input, Shape shape)
        : m_storage(mmove(input)), m_shape(mmove(shape)) {
        if (calculateSize(m_shape) != size()) {
            throw ShapeDoesNotMatchSize();
        }
    }

    NDArrayBase(const NDArrayBase&)            = delete;
    NDArrayBase(NDArrayBase&&)                 = default;
    NDArrayBase& operator=(const NDArrayBase&) = delete;
    NDArrayBase& operator=(NDArrayBase&&)      = default;

    bool empty() const { return m_storage.empty(); }
    const Shape& shape() const { return m_shape; }
    Size size() const { return static_cast<Size>(m_storage.size()); }

    T& operator[](Index index) { return m_storage[index]; }
    const T& operator[](Index index) const { return m_storage[index]; }
    T& operator[](const Position& pos) { return this->operator[](calculateRawIndexUnchecked(m_shape, pos)); }
    const T& operator[](const Position& pos) const {
        return this->operator[](calculateRawIndexUnchecked(m_shape, pos));
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

    static constexpr Index calculateRawIndexUnchecked(const Shape& shape, const Position& position) {
        Index currentStride = 1;
        Index result        = 0;
        for (Index i = shape.size() - 1; i >= 0; --i) {
            result += position[i] * currentStride;
            currentStride *= shape[i];
        }
        return result;
    }

private:
    Storage m_storage;
    Shape m_shape;
};

template<typename NDT>
class NDArraySlice {
public:
    using NDArray                      = NDT;
    using SliceShape                   = typename NDArray::SliceShape;
    using Shape                        = typename NDArray::Shape;
    using Strides                      = typename NDArray::Strides;
    static constexpr bool isConstArray = std::is_const_v<NDArray>;
    using Type = std::conditional<isConstArray, std::add_const_t<typename NDArray::Type>, typename NDArray::Type>;

    struct RawJump {
        Index begin;
        Index end;
        Size stride;
    };

    using JumpTable = PSVec<RawJump, NDArray::Parameters::SHAPE_STACK_SIZE>;

public:
    NDArraySlice(NDArray& array, SliceShape shape)
        : m_ndarray{array}, m_sliceShape{mmove(shape)}, m_jumps(constructJumpTable(m_sliceShape)) {}

    static JumpTable constructJumpTable(const SliceShape& /*sliceShape*/) { return {}; }

private:
    NDArray& m_ndarray;
    const SliceShape m_sliceShape;
    const Shape m_shape;
    const JumpTable m_jumps;
};

struct DefaultNDArrayParams {
    static constexpr Size STACK_SIZE       = 8;
    static constexpr Size SHAPE_STACK_SIZE = 4;
};

template<typename T>
using NDArray = NDArrayBase<T, DefaultNDArrayParams>;

}  // namespace nykdtb

#endif