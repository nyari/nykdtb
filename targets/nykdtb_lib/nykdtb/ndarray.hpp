#ifndef NYKDTB_NDARRAY_HPP
#define NYKDTB_NDARRAY_HPP

#include "nykdtb/types.hpp"

namespace nykdtb {

template<typename T>
class NDArray {
public:
    using Shape   = Vec<Size>;
    using Storage = Vec<T>;

    NYKDTB_DEFINE_EXCEPTION_CLASS(ShapeDoesNotMatchSize, LogicException)

public:
    NDArray() = default;
    NDArray(Storage input)
        : m_storage(mmove(input)), m_shape({static_cast<Size>(m_storage.size())}) {}
    NDArray(Storage input, Shape shape)
        : m_storage(mmove(input)), m_shape(mmove(shape)) {
        if (calculateSize(m_shape) != size()) {
            throw ShapeDoesNotMatchSize();
        }
    }

    NDArray(const NDArray&)            = delete;
    NDArray(NDArray&&)                 = default;
    NDArray& operator=(const NDArray&) = delete;
    NDArray& operator=(NDArray&&)      = default;

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

}  // namespace nykdtb

#endif