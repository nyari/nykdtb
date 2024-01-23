#ifndef NYKDTB_PSVECTOR_HPP
#define NYKDTB_PSVECTOR_HPP

#include <type_traits>
#include <utility>

#include "nykdtb/types.hpp"

namespace nykdtb {

template<typename T, Size STACK_SIZE>
class PartialStackStorageVector {
public:
    NYKDTB_DEFINE_EXCEPTION_CLASS(IncorrectSizeAllocation, LogicException)

    using value_type   = T;
    using ValueType    = T;
    using Pointer      = T*;
    using ConstPointer = const T*;

    struct CopyConstruct;
    struct MoveConstruct;
    struct MoveAssign; 
    struct CopyAssign;

public:
    inline PartialStackStorageVector();
    template<typename Iter, typename Op = CopyConstruct>
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
        transfer(&stack[0], &stack[m_currentSize], m_heapStorage, MoveConstruct{});
    }

    inline void moveHeapToNewHeapWithAllocatedSize(const Size allocatedSize) {
        m_allocatedSize = allocatedSize;
        auto newHeap    = reinterpret_cast<T*>(std::malloc(m_allocatedSize * sizeof(T)));
        transfer(&m_heapStorage[0], &m_heapStorage[m_currentSize], newHeap, MoveConstruct{});
        std::free(m_heapStorage);
        m_heapStorage = newHeap;
    }

    inline void moveHeapToStack() {
        T* stack = stackBegin();
        transfer(&m_heapStorage[0], &m_heapStorage[m_currentSize], stack, MoveConstruct{});
        m_allocatedSize = STACK_SIZE;
        std::free(m_heapStorage);
        m_heapStorage = nullptr;
    }

    struct MoveConstruct {
        inline void operator()(T& lhs, T&& rhs)
        {
            new (&lhs) T{mmove(rhs)};
            rhs.~T();
        }
    };

    struct MoveAssign {
        inline void operator()(T& lhs, T&& rhs)
        {
            lhs = mmove(rhs);
            rhs.~T();
        }
    };

    struct CopyConstruct {
        inline void operator()(T& lhs, const T& rhs)
        {
            new (&lhs) T{rhs};
        }
    };

    struct CopyAssign {
        inline void operator()(T& lhs, const T& rhs)
        {
            lhs = rhs;
        }
    };

    template<typename PS, typename PT, typename Op>
    inline void transfer(PS begin, PS end, PT target, Op op)
    {
        for (auto i = begin; i < end; ++i) {
            op(*(target++), std::move(*i));
        }
    }

    inline void destruct(Pointer begin, Pointer end) {
        for (auto it = begin; it < end; ++it) {
            it->~T();
        }
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
    transfer(_begin, _end, ptr, op);
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
    transfer(other.begin(), other.end(), ptr, CopyConstruct{});
    m_currentSize = other.m_currentSize;
}

template<typename T, Size STACK_SIZE>
inline PartialStackStorageVector<T, STACK_SIZE>::PartialStackStorageVector(PartialStackStorageVector&& other)
    : m_currentSize(other.m_currentSize), m_allocatedSize(STACK_SIZE), m_heapStorage(nullptr) {
    if (other.onStack()) {
        transfer(other.begin(), other.end(), begin(), MoveConstruct{});
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
    transfer(&other[0], &other[commonPartSize], begin(), CopyAssign{});
    destruct(&begin()[commonPartSize], &begin()[m_currentSize]);
    transfer(&other[commonPartSize], &other[other.m_currentSize], &begin()[commonPartSize], CopyConstruct{});
    m_currentSize = other.m_currentSize;
    return *this;
}

template<typename T, Size STACK_SIZE>
inline PartialStackStorageVector<T, STACK_SIZE>& PartialStackStorageVector<T, STACK_SIZE>::operator=(
    PartialStackStorageVector&& other) {
    if (other.onStack()) {
        ensureAllocatedSize(other.m_currentSize);
        const Size commonPartSize = std::min(m_currentSize, other.m_currentSize);
        transfer(&other[0], &other[commonPartSize], begin(), MoveAssign{});
        destruct(&begin()[commonPartSize], &begin()[m_currentSize]);
        transfer(&other[commonPartSize], &other[other.m_currentSize], &begin()[commonPartSize], MoveConstruct{});
    } else {
        destruct(begin(), end());
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
    destruct(begin(), end());
    if (!onStack()) {
        std::free(m_heapStorage);
    }
}

template<typename T, Size STACK_SIZE>
template<typename Iter>
inline Iter PartialStackStorageVector<T, STACK_SIZE>::erase(Iter intervalBegin, Iter intervalEnd) {
    Pointer originalEnd             = this->end();
    Size erasedElemCount            = static_cast<Size>(intervalEnd - intervalBegin);
    Pointer endIteratorAfterMove    = originalEnd - erasedElemCount;

    transfer(intervalEnd, intervalEnd + erasedElemCount, intervalBegin, MoveAssign{});
    transfer(intervalEnd + erasedElemCount, originalEnd, intervalEnd, MoveConstruct{});
    destruct(endIteratorAfterMove, originalEnd);

    m_currentSize -= erasedElemCount;
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