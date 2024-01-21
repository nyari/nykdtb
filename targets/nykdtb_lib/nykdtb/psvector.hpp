#ifndef NYKDTB_PSVECTOR_HPP
#define NYKDTB_PSVECTOR_HPP

#include <type_traits>
#include <utility>

#include "nykdtb/types.hpp"

namespace nykdtb {

struct CopyItems {
    template<typename T>
    inline constexpr const std::remove_reference_t<T>& operator()(const T& v) noexcept {
        return v;
    }
};

struct MoveItems {
    template<typename T>
    inline constexpr std::remove_reference_t<T>&& operator()(T&& subject) noexcept {
        return mmove(subject);
    }
};

template<typename T, Size STACK_SIZE>
class PartialStackStorageVector {
public:
    NYKDTB_DEFINE_EXCEPTION_CLASS(IncorrectSizeAllocation, LogicException)

    using value_type   = T;
    using ValueType    = T;
    using Pointer      = T*;
    using ConstPointer = const T*;

public:
    inline PartialStackStorageVector();
    template<typename Iter, typename Op = CopyItems>
    inline PartialStackStorageVector(Iter _begin, Iter _end, Op op = Op());
    inline PartialStackStorageVector(std::initializer_list<T> init);
    inline PartialStackStorageVector(const PartialStackStorageVector& other);
    inline PartialStackStorageVector(PartialStackStorageVector&& other);
    inline PartialStackStorageVector& operator=(const PartialStackStorageVector& other);
    inline PartialStackStorageVector& operator=(PartialStackStorageVector&& other);

    inline ~PartialStackStorageVector();

    inline Size size() const { return m_currentSize; }

    inline Pointer begin() {
        if (onStack()) {
            return stackBegin();
        } else {
            return &m_heapStorage[0];
        }
    }

    inline ConstPointer begin() const {
        if (onStack()) {
            return stackBegin();
        } else {
            return &m_heapStorage[0];
        }
    }

    inline Pointer end() { return &begin()[m_currentSize]; }
    inline ConstPointer end() const { return &begin()[m_currentSize]; }

    inline void push_back(T elem) {
        ensureAllocatedSize(m_currentSize + 1);
        new (&begin()[m_currentSize]) T{mmove(elem)};
        ++m_currentSize;
    }

    template<typename... Args>
    inline void emplace_back(Args&&... args) {
        ensureAllocatedSize(m_currentSize + 1);
        new (&begin()[m_currentSize]) T{std::forward<Args>(args)...};
        ++m_currentSize;
    }

    template<typename Iter>
    inline Iter erase(Iter intervalBegin, Iter intervalEnd);

    template<typename Iter>
    inline Iter erase(Iter elementIt) {
        return erase(elementIt, elementIt + 1);
    }

    inline T& operator[](const Index i) { return begin()[i]; }
    inline const T& operator[](const Index i) const { return begin()[i]; }

    inline bool onStack() const { return m_heapStorage == nullptr; }
    inline bool empty() const { return m_currentSize == 0; }

public:
    inline Pointer stackBegin() { return reinterpret_cast<Pointer>(&m_stackStorage[0]); }
    inline ConstPointer stackBegin() const { return reinterpret_cast<ConstPointer>(&m_stackStorage[0]); }

    inline void ensureAllocatedSize(const Size size) {
        if (m_currentSize > size) throw IncorrectSizeAllocation();

        if (onStack()) {
            if (size > STACK_SIZE) {
                moveStackToHeapWithAllocatedSize(size * 2);
            }
        } else {
            if (size > STACK_SIZE) {
                if (size > m_allocatedSize) {
                    moveHeapToNewHeapWithAllocatedSize(size * 2);
                }
            } else {
                moveHeapToStack();
            }
        }
    }

    inline void moveStackToHeapWithAllocatedSize(const Size allocatedSize) {
        m_allocatedSize = allocatedSize;
        m_heapStorage   = reinterpret_cast<T*>(std::malloc(m_allocatedSize * sizeof(T)));
        T* stack        = stackBegin();
        for (Index i = 0; i < m_currentSize; ++i) {
            new (&m_heapStorage[i]) T{mmove(stack[i])};
            stack[i].~T();
        }
    }

    inline void moveHeapToNewHeapWithAllocatedSize(const Size allocatedSize) {
        m_allocatedSize = allocatedSize;
        auto newHeap    = reinterpret_cast<T*>(std::malloc(m_allocatedSize * sizeof(T)));
        for (Index i = 0; i < m_currentSize; ++i) {
            new (&newHeap[i]) T{mmove(m_heapStorage[i])};
            m_heapStorage[i].~T();
        }
        m_heapStorage = newHeap;
    }

    inline void moveHeapToStack() {
        T* stack = stackBegin();
        for (Index i = 0; i < m_currentSize; ++i) {
            new (&stack[i]) T{mmove(m_heapStorage[i])};
            m_heapStorage[i].~T();
        }
        m_allocatedSize = STACK_SIZE;
        std::free(m_heapStorage);
        m_heapStorage = nullptr;
    }

private:
    Size m_currentSize;
    Size m_allocatedSize;
    T* m_heapStorage;
    uint8_t m_stackStorage[sizeof(T) * STACK_SIZE];
};

template<typename T, Size STACK_SIZE>
inline PartialStackStorageVector<T, STACK_SIZE>::PartialStackStorageVector()
    : m_currentSize(0), m_allocatedSize(STACK_SIZE), m_heapStorage(nullptr) {}

template<typename T, Size STACK_SIZE>
template<typename Iter, typename Op>
inline PartialStackStorageVector<T, STACK_SIZE>::PartialStackStorageVector(Iter _begin, Iter _end, Op op)
    : m_currentSize(0), m_allocatedSize(STACK_SIZE), m_heapStorage(nullptr) {
    const Size targetSize = static_cast<Size>(_end - _begin);
    ensureAllocatedSize(targetSize);
    auto ptr = begin();
    for (auto it = _begin; it < _end; ++it) {
        new (ptr++) T{op(*it)};
    }
    m_currentSize = targetSize;
}

template<typename T, Size STACK_SIZE>
inline PartialStackStorageVector<T, STACK_SIZE>::PartialStackStorageVector(std::initializer_list<T> init)
    : PartialStackStorageVector(init.begin(), init.end()) {}

template<typename T, Size STACK_SIZE>
inline PartialStackStorageVector<T, STACK_SIZE>::PartialStackStorageVector(const PartialStackStorageVector& other)
    : m_currentSize(0), m_allocatedSize(STACK_SIZE), m_heapStorage(nullptr) {
    ensureAllocatedSize(other.m_currentSize);
    auto ptr = begin();
    for (const auto& item : other) {
        new (ptr++) T{item};
    }
    m_currentSize = other.m_currentSize;
}

template<typename T, Size STACK_SIZE>
inline PartialStackStorageVector<T, STACK_SIZE>::PartialStackStorageVector(PartialStackStorageVector&& other)
    : m_currentSize(other.m_currentSize), m_allocatedSize(STACK_SIZE), m_heapStorage(nullptr) {
    if (other.onStack()) {
        auto ptr = begin();
        for (auto& item : other) {
            new (ptr++) T{mmove(item)};
            item.~T();
        }
    } else {
        m_heapStorage   = other.m_heapStorage;
        m_allocatedSize = other.m_allocatedSize;
    }
    other.m_heapStorage   = nullptr;
    other.m_currentSize   = 0;
    other.m_allocatedSize = 0;
}

template<typename T, Size STACK_SIZE>
inline PartialStackStorageVector<T, STACK_SIZE>& PartialStackStorageVector<T, STACK_SIZE>::operator=(
    const PartialStackStorageVector& other) {
    ensureAllocatedSize(other.m_currentSize);
    const Size commonPartSize = std::min(m_currentSize, other.m_currentSize);
    for (Index i = 0; i < commonPartSize; ++i) {
        this->operator[](i) = other[i];
    }
    for (Index i = commonPartSize; i < m_currentSize; ++i) {
        begin()[i].~T();
    }
    for (Index i = commonPartSize; i < other.m_currentSize; ++i) {
        new (&begin()[i]) T{other[i]};
    }
    m_currentSize = other.m_currentSize;
    return *this;
}

template<typename T, Size STACK_SIZE>
inline PartialStackStorageVector<T, STACK_SIZE>& PartialStackStorageVector<T, STACK_SIZE>::operator=(
    PartialStackStorageVector&& other) {
    if (other.onStack()) {
        ensureAllocatedSize(other.m_currentSize);
        const Size commonPartSize = std::min(m_currentSize, other.m_currentSize);
        for (Index i = 0; i < commonPartSize; ++i) {
            this->operator[](i) = mmove(other[i]);
            other[i].~T();
        }
        for (Index i = commonPartSize; i < m_currentSize; ++i) {
            begin()[i].~T();
        }
        for (Index i = commonPartSize; i < other.m_currentSize; ++i) {
            new (&begin()[i]) T{mmove(other[i])};
            other[i].~T();
        }
    } else {
        for (auto& item : *this) {
            item.~T();
        }
        std::free(m_heapStorage);
        m_heapStorage   = other.m_heapStorage;
        m_allocatedSize = other.m_allocatedSize;
    }
    m_currentSize         = other.m_currentSize;
    other.m_heapStorage   = nullptr;
    other.m_currentSize   = 0;
    other.m_allocatedSize = 0;
    return *this;
}

template<typename T, Size STACK_SIZE>
inline PartialStackStorageVector<T, STACK_SIZE>::~PartialStackStorageVector() {
    for (auto& item : *this) {
        item.~T();
    }
    if (!onStack()) {
        std::free(m_heapStorage);
    }
}

template<typename T, Size STACK_SIZE>
template<typename Iter>
inline Iter PartialStackStorageVector<T, STACK_SIZE>::erase(Iter intervalBegin, Iter intervalEnd) {
    Pointer end         = this->end();
    auto offsetDistance = intervalEnd - intervalBegin;
    Pointer movementEnd = end - offsetDistance;
    for (auto it = intervalBegin; it < intervalEnd; ++it) {
        *it = mmove(it[offsetDistance]);
        it[offsetDistance].~T();
    }
    for (auto it = intervalEnd; it < movementEnd; ++it) {
        new (it) T{mmove(it[offsetDistance])};
        it[offsetDistance].~T();
    }
    for (auto it = movementEnd; it < end; ++it) {
        it->~T();
    }
    m_currentSize -= offsetDistance;
    ensureAllocatedSize(m_currentSize);
    return intervalBegin;
}

template<typename T, Size STACK_SIZE>
using PSVec = PartialStackStorageVector<T, STACK_SIZE>;

template<typename T>
using PSVec8 = PartialStackStorageVector<T, 8>;

template<typename T>
using PSVec4 = PartialStackStorageVector<T, 4>;

}  // namespace nykdtb

#endif