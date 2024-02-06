#ifndef NYKDTB_NDARRAY_OPS_HPP
#define NYKDTB_NDARRAY_OPS_HPP

#include <cmath>

#include "nykdtb/ndarray.hpp"

namespace nykdtb::nda {

NYKDTB_DEFINE_EXCEPTION_CLASS(ShapesDoNotMatch, LogicException);
NYKDTB_DEFINE_EXCEPTION_CLASS(SizesDoNotMatch, LogicException);

template<NDArrayLike LHS, NDArrayLike RHS>
inline static void addAssign(LHS& lhs, const RHS& rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw ShapesDoNotMatch();
    }

    auto lhsBegin = lhs.begin();
    auto lhsEnd   = lhs.end();
    auto rhsIt    = rhs.begin();

    for (auto lhsIt = lhsBegin; lhsIt < lhsEnd; ++lhsIt, ++rhsIt) {
        *lhsIt += *rhsIt;
    }
}

template<NDArrayLike LHS, NDArrayLike RHS>
inline static LHS add(LHS lhs, const RHS& rhs) {
    addAssign(lhs, rhs);
    return mmove(lhs);
}

template<NDArrayLike LHS, NDArrayLike RHS>
inline static void assign(LHS& lhs, const RHS& rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw ShapesDoNotMatch();
    }

    auto lhsBegin = lhs.begin();
    auto lhsEnd   = lhs.end();
    auto rhsIt    = rhs.begin();

    for (auto lhsIt = lhsBegin; lhsIt < lhsEnd; ++lhsIt, ++rhsIt) {
        *lhsIt = *rhsIt;
    }
}

template<NDArrayLike T>
inline static void normalize(T& elem) {
    typename T::Type lengthsq = 0;
    for (const auto& e : elem) {
        lengthsq += e * e;
    }
    auto mtp = 1.0F / std::sqrt(lengthsq);
    for (auto& e : elem) {
        e *= div;
    }
}

template<NDArrayLike LHS, NDArrayLike RHS>
inline static typename LHS::Type dot(const LHS& lhs, const RHS& rhs) {
    static_assert(std::is_same_v<typename LHS::T, typename RHS::T>,
                  "This function needs to be called with NDArrays of the same internal type");

    if (lhs.size() != rhs.size()) {
        throw SizesDoNotMatch();
    }
    auto lhsBegin = lhs.begin();
    auto lhsEnd   = lhs.end();
    auto rhsIt    = rhs.begin();

    typename LHS::Type result = 0;

    for (auto lhsIt = lhsBegin; lhsIt < lhsEnd; ++lhsIt, ++rhsIt) {
        result += (*lhsIt) * (*rhsIt);
    }

    return result;
}

template<NDArrayLike T>
inline static T normalized(T elem) {
    normalize(elem);
    return mmove(elem);
}

/*
TODO list:
* Matrix inverse
* Subtract
* Cross product
* Axis, rotation matrix creation
*/

}  // namespace nykdtb::nda

#endif