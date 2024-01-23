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

    static constexpr auto moveConstruct = [] (T& lhs, T&& rhs) { new (&lhs) T{mmove(rhs)}; rhs.~T();};
    static constexpr auto moveAssign = [] (T& lhs, T&& rhs) { lhs = mmove(rhs); rhs.~T(); };
    static constexpr auto copyConstruct = [] (T& lhs, const T& rhs) { new (&lhs) T{rhs}; };
    static constexpr auto copyAssign = [] (T& lhs, const T& rhs) { lhs = rhs; };

public:
    inline PartialStackStorageVector();
    template<typename Iter, typename Op = decltype(copyConstruct)>
    inline PartialStackStorageVector(Iter _begin, Iter _end, Op op = copyConstruct);
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

    inline Pointer ptr(const Index idx) { return &begin()[idx]; }
    inline ConstPointer ptr(const Index idx) const { return &begin()[idx]; }

    inline Pointer end() { return ptr(m_currentSize); }
    inline ConstPointer end() const { return ptr(m_currentSize); }

    inline void push_back(T elem) {
        ensureAllocatedSize(m_currentSize + 1);
        new (ptr(m_currentSize)) T{mmove(elem)};
        ++m_currentSize;
    }

    template<typename... Args>
    inline void emplace_back(Args&&... args) {
        ensureAllocatedSize(m_currentSize + 1);
        new (ptr(m_currentSize)) T{std::forward<Args>(args)...};
        ++m_currentSize;
    }

    inline Pointer erase(Pointer intervalBegin, Pointer intervalEnd);
    inline Pointer erase(Pointer elementIt) {
        return erase(elementIt, elementIt + 1);
    }

    template<typename SIter>
    inline Pointer insert(Pointer before, SIter first, SIter last);
    inline Pointer insert(Pointer before, T value) { return insert(before, &value, (&value) + 1); }

    inline T& operator[](const Index i) { return *ptr(i); }
    inline const T& operator[](const Index i) const { return *ptr(i); }

    inline bool onStack() const { return m_heapStorage == nullptr; }
    inline bool empty() const { return m_currentSize == 0; }

private:
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

    static inline Pointer allocateMemory(const Size elemCount) {
        return reinterpret_cast<Pointer>(std::malloc(elemCount * sizeof(T)));
    }

    static inline void free(Pointer ptr) {
        std::free(ptr);
    }

    inline void moveStackToHeapWithAllocatedSize(const Size allocatedSize) {
        m_allocatedSize = allocatedSize;
        m_heapStorage   = allocateMemory(m_allocatedSize);
        T* stack        = stackBegin();
        transfer(&stack[0], &stack[m_currentSize], m_heapStorage, moveConstruct);
    }

    inline void moveHeapToNewHeapWithAllocatedSize(const Size allocatedSize) {
        m_allocatedSize = allocatedSize;
        Pointer newHeap    = allocateMemory(m_allocatedSize);
        transfer(&m_heapStorage[0], &m_heapStorage[m_currentSize], newHeap, moveConstruct);
        free(m_heapStorage);
        m_heapStorage = newHeap;
    }

    inline void moveHeapToStack() {
        transfer(&m_heapStorage[0], &m_heapStorage[m_currentSize], stackBegin(), moveConstruct);
        m_allocatedSize = STACK_SIZE;
        free(m_heapStorage);
        m_heapStorage = nullptr;
    }

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
    transfer(_begin, _end, begin(), op);
    m_currentSize = targetSize;
}

template<typename T, Size STACK_SIZE>
inline PartialStackStorageVector<T, STACK_SIZE>::PartialStackStorageVector(std::initializer_list<T> init)
    : PartialStackStorageVector(init.begin(), init.end()) {}

template<typename T, Size STACK_SIZE>
inline PartialStackStorageVector<T, STACK_SIZE>::PartialStackStorageVector(const PartialStackStorageVector& other)
    : m_currentSize(0), m_allocatedSize(STACK_SIZE), m_heapStorage(nullptr) {
    ensureAllocatedSize(other.m_currentSize);
    transfer(other.begin(), other.end(), begin(), copyConstruct);
    m_currentSize = other.m_currentSize;
}

template<typename T, Size STACK_SIZE>
inline PartialStackStorageVector<T, STACK_SIZE>::PartialStackStorageVector(PartialStackStorageVector&& other)
    : m_currentSize(other.m_currentSize), m_allocatedSize(STACK_SIZE), m_heapStorage(nullptr) {
    if (other.onStack()) {
        transfer(other.begin(), other.end(), begin(), moveConstruct);
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

    const Size commonPartSize = std::min(m_currentSize, other.m_currentSize);

    ensureAllocatedSize(other.m_currentSize);
    transfer(other.begin(), other.ptr(commonPartSize), begin(), copyAssign);
    destruct(ptr(commonPartSize), ptr(m_currentSize));
    transfer(other.ptr(commonPartSize), other.ptr(other.m_currentSize), ptr(commonPartSize), copyConstruct);

    m_currentSize = other.m_currentSize;
    return *this;
}

template<typename T, Size STACK_SIZE>
inline PartialStackStorageVector<T, STACK_SIZE>& PartialStackStorageVector<T, STACK_SIZE>::operator=(
    PartialStackStorageVector&& other) {
    if (other.onStack()) {
        const Size commonPartSize = std::min(m_currentSize, other.m_currentSize);

        ensureAllocatedSize(other.m_currentSize);
        transfer(other.begin(), other.ptr(commonPartSize), begin(), moveAssign);
        destruct(ptr(commonPartSize), ptr(m_currentSize));
        transfer(other.ptr(commonPartSize), other.ptr(other.m_currentSize), ptr(commonPartSize), moveConstruct);
    } else {
        destruct(begin(), end());
        free(m_heapStorage);
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
        free(m_heapStorage);
    }
}

template<typename T, Size STACK_SIZE>
inline typename PartialStackStorageVector<T, STACK_SIZE>::Pointer PartialStackStorageVector<T, STACK_SIZE>::erase(Pointer intervalBegin, Pointer intervalEnd) {
    Pointer originalEnd             = this->end();
    Size erasedElemCount            = static_cast<Size>(intervalEnd - intervalBegin);
    Pointer endIteratorAfterMove    = originalEnd - erasedElemCount;

    transfer(intervalEnd, intervalEnd + erasedElemCount, intervalBegin, moveAssign);
    transfer(intervalEnd + erasedElemCount, originalEnd, intervalEnd, moveConstruct);
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