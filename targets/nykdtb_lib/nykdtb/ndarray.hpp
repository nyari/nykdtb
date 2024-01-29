#ifndef NYKDTB_NDARRAY_HPP
#define NYKDTB_NDARRAY_HPP

#include "nykdtb/psvector.hpp"
#include "nykdtb/types.hpp"

namespace nykdtb {

template<typename T, typename Params>
class NDArrayBase {
public:
    using Shape      = PSVec<Size, Params::SHAPE_STACK_SIZE>;
    using Storage    = PSVec<T, Params::STACK_SIZE>;
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

    bool empty() const { return m_storage == nullptr; }
    const Shape& shape() const { return m_shape; }
    Size size() const { return static_cast<Size>(m_storage.size()); }

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

private:
private:
    Storage m_storage;
    Shape m_shape;
};

struct DefaultNDArrayParams {
    static constexpr Size STACK_SIZE       = 8;
    static constexpr Size SHAPE_STACK_SIZE = 4;
};

template<typename T>
using NDArray = NDArrayBase<T, DefaultNDArrayParams>;

}  // namespace nykdtb

#endif