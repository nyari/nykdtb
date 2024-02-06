#ifndef NYKDTB_NDARRAY_HPP
#define NYKDTB_NDARRAY_HPP

#include <variant>

#include "nykdtb/psvector.hpp"
#include "nykdtb/types.hpp"
#include "nykdtb/utils.hpp"

namespace nykdtb {

template<Size STACK_SIZE>
struct NDArrayShape : public PSVec<Size, STACK_SIZE> {
    NDArrayShape() = default;
    NDArrayShape(std::initializer_list<Size> list)
        : PSVec<Size, STACK_SIZE>(list) {}

    bool operator==(const NDArrayShape& other) const {
        const NDArrayShape& me = *this;

        for (Index i = 0, j = 0; i < me.size() && j < other.size(); ++i, ++j) {
            if (me[i] != other[j]) {
                if (me[i] == 1 && other[j] != 1) [[unlikely]] {
                    --j;
                } else if (me[i] != 1 && other[j] == 1) [[unlikely]] {
                    --i;
                } else {
                    return false;
                }
            }
        }

        return true;
    }

    bool operator!=(const NDArrayShape& other) const { return !(*this == other); }

    Size shapeSize() const {
        if (PSVec<Size, STACK_SIZE>::empty()) {
            return 0;
        }

        Size result = 1;
        for (auto size : *this) {
            result *= size;
        }
        return result;
    }
};

template<typename T>
concept NDArrayLike = requires(T a) {
    typename T::Type;
    typename T::SliceShape;
    typename T::Shape;
    typename T::Strides;
    typename T::Position;
    typename T::Iterator;
    typename T::ConstIterator;

    { a[Index{0}] } -> std::common_reference_with<typename T::Type>;
    { a[typename T::Position{0}] } -> std::common_reference_with<typename T::Type>;
    { a.empty() } -> std::same_as<bool>;
    { a.shape() } -> std::common_reference_with<typename T::Shape>;
    { a.strides() } -> std::common_reference_with<typename T::Strides>;
    { a.size() } -> std::same_as<Size>;

    { a.begin() } -> std::same_as<typename T::Iterator>;
    { const_cast<const T&>(a).begin() } -> std::same_as<typename T::ConstIterator>;
    { a.end() } -> std::same_as<typename T::Iterator>;
    { const_cast<const T&>(a).end() } -> std::same_as<typename T::ConstIterator>;
};

template<NDArrayLike NDT>
class NDArraySlice;

template<typename T, typename Params>
class NDArrayBase {
public:
    using Type          = T;
    using Shape         = NDArrayShape<Params::SHAPE_STACK_SIZE>;
    using Strides       = PSVec<Size, Params::SHAPE_STACK_SIZE>;
    using Position      = PSVec<Index, Params::SHAPE_STACK_SIZE>;
    using Storage       = PSVec<T, Params::STACK_SIZE>;
    using SliceShape    = PSVec<IndexRange, Params::SHAPE_STACK_SIZE>;
    using Parameters    = Params;
    using Iterator      = Type*;
    using ConstIterator = const Type*;

    NYKDTB_DEFINE_EXCEPTION_CLASS(ShapeDoesNotMatchSize, LogicException)

public:
    NDArrayBase() = default;
    NDArrayBase(Storage input)
        : m_storage(mmove(input)),
          m_shape({static_cast<Size>(m_storage.size())}),
          m_strides(calculateStrides(m_shape)) {}
    NDArrayBase(Storage input, Shape shape)
        : m_storage(mmove(input)), m_shape(mmove(shape)), m_strides(calculateStrides(m_shape)) {
        if (m_shape.shapeSize() != size()) {
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

    Iterator begin() { return m_storage.begin(); }
    ConstIterator begin() const { return m_storage.begin(); }
    Iterator end() { return m_storage.end(); }
    ConstIterator end() const { return m_storage.end(); }

    T& operator[](Index index) { return m_storage[index]; }
    const T& operator[](Index index) const { return m_storage[index]; }
    T& operator[](const Position& pos) { return this->operator[](calculateRawIndexUnchecked(m_strides, pos)); }
    const T& operator[](const Position& pos) const {
        return this->operator[](calculateRawIndexUnchecked(m_strides, pos));
    }

    void reshape(Shape shape) {
        if (shape.shapeSize() != m_shape.shapeSize()) {
            throw ShapeDoesNotMatchSize();
        }

        m_shape   = mmove(shape);
        m_strides = calculateStrides(m_shape);
    }

    void resize(Shape newShape, T init) {
        m_storage.resize(newShape.shapeSize(), mmove(init));
        m_shape   = mmove(newShape);
        m_strides = calculateStrides(m_shape);
    }

public:
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
    NDArrayBase& operator=(const NDArrayBase&) = delete;

private:
    Storage m_storage;
    Shape m_shape;
    Strides m_strides;
};

template<NDArrayLike NDT>
class NDArraySlice {
public:
    using NDArray     = NDT;
    using NDArrayType = std::remove_cvref_t<NDArray>;
    using Type        = typename NDArray::Type;
    using SliceShape  = typename NDArray::SliceShape;
    using Shape       = typename NDArray::Shape;
    using Strides     = typename NDArray::Strides;
    using Position    = typename NDArray::Position;

    static constexpr bool isConstArray = std::is_const_v<NDArray>;
    using MutType                      = std::conditional_t<isConstArray, std::add_const_t<Type>, Type>;
    using ConstType                    = const std::remove_cvref_t<Type>;

    template<typename T>
    class IteratorBase;

    using Iterator      = IteratorBase<NDArraySlice>;
    using ConstIterator = IteratorBase<const std::remove_cvref_t<NDArraySlice>>;

    NYKDTB_DEFINE_EXCEPTION_CLASS(InvalidSliceShape, LogicException)

public:
    NDArraySlice(NDArray& array, SliceShape shape)
        : m_ndarray{array},
          m_sliceShape{mmove(shape)},
          m_shape(calculateShape(m_ndarray.shape(), m_sliceShape)),
          m_strides(NDArray::calculateStrides(m_shape)) {}

    bool empty() const { return m_shape.shapeSize(); }
    const Shape& shape() const { return m_shape; }
    const SliceShape& sliceShape() const { return m_sliceShape; }
    const Strides& strides() const { return m_strides; }
    Size size() const { return m_shape.shapeSize(); }

    NDArrayType materialize() const { return NDArrayType{{begin(), end()}, m_shape}; }

    Iterator begin() { return Iterator(*this); }
    ConstIterator begin() const { return ConstIterator(*this); }
    Iterator end() { return Iterator(*this, Iterator::End); }
    ConstIterator end() const { return ConstIterator(*this, ConstIterator::End); }

    MutType& operator[](const Index index) { return m_ndarray[calculateRawIndexFromSliceIndexUnchecked(index)]; }
    ConstType& operator[](const Index index) const {
        return m_ndarray[calculateRawIndexFromSliceIndexUnchecked(index)];
    }

    MutType& operator[](const Position& position) {
        return m_ndarray[calculateRawIndexFromPositionUnchecked(position)];
    }
    ConstType& operator[](const Position& position) const {
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

public:
    template<typename T>
    class IteratorBase {
    public:
        enum EndPlacement { End };

        using Type      = typename T::Type;
        using ConstType = const Type;
        using Position  = typename T::Position;

    public:
        IteratorBase(T& slice)
            : m_slice(slice),
              m_pos{Position::constructFilled(m_slice.shape().size(), 0)},
              m_rawIndex{m_slice.calculateRawIndexFromPositionUnchecked(m_pos)} {}
        IteratorBase(T& slice, EndPlacement)
            : m_slice(slice),
              m_pos{m_slice.transformed([](const auto& item) { return item - 1; })},
              m_rawIndex{m_slice.calculateRawIndexFromPositionUnchecked(m_pos)} {}

        IteratorBase& operator++() {
            advanceOne();
            return *this;
        }

        IteratorBase operator++(int) {
            IteratorBase copy(*this);
            advanceOne();
            return mmove(copy);
        }

        Type& operator*() { return m_slice.m_ndarray[m_rawIndex]; }
        ConstType& operator*() const { return m_slice.m_ndarray[m_rawIndex]; }
        Type* operator->() { return &m_slice.m_ndarray[m_rawIndex]; }
        ConstType* operator->() const { return &m_slice.m_ndarray[m_rawIndex]; }

        bool operator==(const IteratorBase& other) const { return m_rawIndex == other.m_rawIndex; }
        bool operator!=(const IteratorBase& other) const { return m_rawIndex != other.m_rawIndex; }
        bool operator<(const IteratorBase& other) const { return m_rawIndex < other.m_rawIndex; }
        bool operator<=(const IteratorBase& other) const { return m_rawIndex <= other.m_rawIndex; }

    private:
        void advanceOne() {
            bool recalculateRaw = false;
            for (Index i = m_pos.size() - 1; i >= 0; --i) {
                if (++m_pos[i] >= m_slice.shape()[i]) [[unlikely]] {
                    recalculateRaw = true;
                    m_pos[i]       = 0;
                } else [[likely]] {
                    ++m_rawIndex;
                    break;
                }
            }
            if (recalculateRaw) [[unlikely]] {
                m_rawIndex = m_slice.calculateRawIndexFromPositionUnchecked(m_pos);
            }
        }

    private:
        T & m_slice;
        Position m_pos;
        Index m_rawIndex;
    };
};

struct DefaultNDArrayParams {
    static constexpr Size STACK_SIZE       = 8;
    static constexpr Size SHAPE_STACK_SIZE = 4;
};

template<typename T>
using NDArray = NDArrayBase<T, DefaultNDArrayParams>;

}  // namespace nykdtb

#endif