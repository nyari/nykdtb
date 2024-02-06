#ifndef NYKDTB_NDARRAY_OPS_HPP
#define NYKDTB_NDARRAY_OPS_HPP

#include <cmath>

#include "nykdtb/ndarray.hpp"

namespace nykdtb::nda {

NYKDTB_DEFINE_EXCEPTION_CLASS(ShapesDoNotMatch, LogicException)
NYKDTB_DEFINE_EXCEPTION_CLASS(SizesDoNotMatch, LogicException)
NYKDTB_DEFINE_EXCEPTION_CLASS(DivisionByZero, RuntimeException)

template<NDArrayLike LHS, NDArrayLike RHS, typename F>
inline static void baseAssignWithSameShape(LHS& lhs, const RHS& rhs, F op) {
    if (lhs.shape() != rhs.shape()) {
        throw ShapesDoNotMatch();
    }

    auto lhsBegin = lhs.begin();
    auto lhsEnd   = lhs.end();
    auto rhsIt    = rhs.begin();

    for (auto lhsIt = lhsBegin; lhsIt < lhsEnd; ++lhsIt, ++rhsIt) {
        op(*lhsIt, *rhsIt);
    }
}

template<NDArrayLike LHS, NDArrayLike RHS>
inline static void addAssign(LHS& lhs, const RHS& rhs) {
    baseAssignWithSameShape(lhs, rhs, [](auto& lhs, const auto& rhs) { lhs += rhs; });
}

template<NDArrayLike LHS, NDArrayLike RHS>
inline static LHS add(LHS lhs, const RHS& rhs) {
    addAssign(lhs, rhs);
    return mmove(lhs);
}

template<NDArrayLike LHS, NDArrayLike RHS>
inline static void subAssign(LHS& lhs, const RHS& rhs) {
    baseAssignWithSameShape(lhs, rhs, [](auto& lhs, const auto& rhs) { lhs -= rhs; });
}

template<NDArrayLike LHS, NDArrayLike RHS>
inline static LHS sub(LHS lhs, const RHS& rhs) {
    subAssign(lhs, rhs);
    return mmove(lhs);
}

template<NDArrayLike LHS, NDArrayLike RHS>
inline static void ewMulAssign(LHS& lhs, const RHS& rhs) {
    baseAssignWithSameShape(lhs, rhs, [](auto& lhs, const auto& rhs) { lhs *= rhs; });
}

template<NDArrayLike LHS, NDArrayLike RHS>
inline static LHS ewMul(LHS lhs, const RHS& rhs) {
    ewMulAssign(lhs, rhs);
    return mmove(lhs);
}

template<NDArrayLike LHS, NDArrayLike RHS>
inline static void ewDivAssign(LHS& lhs, const RHS& rhs) {
    baseAssignWithSameShape(lhs, rhs, [](auto& lhs, const auto& rhs) { lhs /= rhs; });
}

template<NDArrayLike LHS, NDArrayLike RHS>
inline static LHS ewDiv(LHS lhs, const RHS& rhs) {
    ewDivAssign(lhs, rhs);
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

template<NDArrayLike T, typename F>
inline static void baseAssignWithScalar(T& lhs, typename T::Type& rhs, F op) {
    auto lhsBegin = lhs.begin();
    auto lhsEnd   = lhs.end();
    auto rhsIt    = rhs.begin();

    for (auto lhsIt = lhsBegin; lhsIt < lhsEnd; ++lhsIt, ++rhsIt) {
        op(*lhsIt, rhs);
    }
}

template<NDArrayLike T>
inline static void addAssignScalar(T& lhs, typename T::Type& rhs) {
    baseAssignWithScalar(lhs, rhs, [](auto& lhs, const auto& rhs) { lhs += rhs; });
}

template<NDArrayLike T>
inline static void addScalar(T lhs, typename T::Type& rhs) {
    addAssignScalar(lhs, rhs);
    return mmove(lhs);
}

template<NDArrayLike T>
inline static void subAssignScalar(T& lhs, typename T::Type& rhs) {
    baseAssignWithScalar(lhs, rhs, [](auto& lhs, const auto& rhs) { lhs -= rhs; });
}

template<NDArrayLike T>
inline static void subScalar(T lhs, typename T::Type& rhs) {
    subAssignScalar(lhs, rhs);
    return mmove(lhs);
}

template<NDArrayLike T>
inline static void mulAssignScalar(T& lhs, typename T::Type& rhs) {
    baseAssignWithScalar(lhs, rhs, [](auto& lhs, const auto& rhs) { lhs *= rhs; });
}

template<NDArrayLike T>
inline static void mulScalar(T lhs, typename T::Type& rhs) {
    mulAssignScalar(lhs, rhs);
    return mmove(lhs);
}

template<NDArrayLike T>
inline static void divAssignScalar(T& lhs, typename T::Type& rhs) {
    baseAssignWithScalar(lhs, rhs, [](auto& lhs, const auto& rhs) { lhs /= rhs; });
}

template<NDArrayLike T>
inline static void divScalar(T lhs, typename T::Type& rhs) {
    divAssignScalar(lhs, rhs);
    return mmove(lhs);
}

template<NDArrayLike T>
inline static void normalize(T& elem) {
    typename T::Type lengthsq = 0;
    for (const auto& e : elem) {
        lengthsq += e * e;
    }
    if (lengthsq == 0.0) {
        throw DivisionByZero();
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

namespace d2 {

NYKDTB_DEFINE_EXCEPTION_CLASS(Matrix2DError, RuntimeException)

template<NDArrayLike T>
inline static bool isSquare(const typename T::Shape& shape) {
    return (shape.dims() == 2) && (shape[0] == shape[1]);
}

template<NDArrayLike T>
inline static typename T::MaterialType identity(const typename T::Shape& shape) {
    using Mx = typename T::MaterialType;
    if (!isSquare(shape)) {
        throw Matrix2DError("Only 2D square matrices have identity");
    }

    Mx result(Mx::zeros(shape));

    for (Index i = 0; i < shape[0]; ++i) {
        result[{i, i}];
    }

    return result;
}

template<NDArrayLike T>
inline static typename T::MaterialType inverse(T input) {
    using Mx = typename T::MaterialType;
    if (!isSquare(input.shape())) {
        throw Matrix2DError("Only 2D square matrices are invertable");
    }

    Mx result      = identity(input.shape());
    const Size dim = input.shape(0);

    // Bottom half Gauss elimination
    for (Index leading = 0; leading < dim; ++leading) {
        const auto leadingRowSelect = typename T::SliceShape{IndexRange::single(leading), IndexRange::e2e()};
        auto leadInputRow           = input.slice(leadingRowSelect);
        auto leadResultRow          = result.slice(leadingRowSelect);
        const auto leadValue        = 1.0 / leadInputRow[{0, leading}];
        mulAssignScalar(leadInputRow, leadValue);
        mulAssignScalar(leadResultRow, leadValue);

        for (Index remaining = leading + 1; remaining < dim; ++remaining) {
            const auto remainingRowSelect = typename T::SliceShape{IndexRange::single(remaining), IndexRange::e2e()};
            auto remainingInputRow        = input.slice(remainingRowSelect);
            auto remainingResultRow       = result.slice(remainingRowSelect);
            const auto remainingValue     = remainingInputRow[{0, leading}];
            subAssign(remainingInputRow, ewMul(leadInputRow.materialize(), remainingValue));
            subAssign(remainingResultRow, ewMul(leadResultRow.materialize(), remainingValue));
        }
    }

    return mmove(result);
}

}  // namespace d2

template<NDArrayLike T>
inline static T normalized(T elem) {
    normalize(elem);
    return mmove(elem);
}

/*
TODO list:
* Matrix inverse
* Cross product
* Axis, rotation matrix creation
*/

}  // namespace nykdtb::nda

#endif