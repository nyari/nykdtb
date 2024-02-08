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
inline static void baseAssignWithScalar(T& lhs, const typename T::Type& rhs, F op) {
    auto lhsBegin = lhs.begin();
    auto lhsEnd   = lhs.end();

    for (auto lhsIt = lhsBegin; lhsIt < lhsEnd; ++lhsIt) {
        op(*lhsIt, rhs);
    }
}

template<NDArrayLike T>
inline static void addAssignScalar(T& lhs, const typename T::Type& rhs) {
    baseAssignWithScalar(lhs, rhs, [](auto& lhs, const auto& rhs) { lhs += rhs; });
}

template<NDArrayLike T>
inline static T addScalar(T lhs, const typename T::Type& rhs) {
    addAssignScalar(lhs, rhs);
    return mmove(lhs);
}

template<NDArrayLike T>
inline static void subAssignScalar(T& lhs, const typename T::Type& rhs) {
    baseAssignWithScalar(lhs, rhs, [](auto& lhs, const auto& rhs) { lhs -= rhs; });
}

template<NDArrayLike T>
inline static T subScalar(T lhs, const typename T::Type& rhs) {
    subAssignScalar(lhs, rhs);
    return mmove(lhs);
}

template<NDArrayLike T>
inline static void mulAssignScalar(T& lhs, const typename T::Type& rhs) {
    baseAssignWithScalar(lhs, rhs, [](auto& lhs, const auto& rhs) { lhs *= rhs; });
}

template<NDArrayLike T>
inline static T mulScalar(T lhs, const typename T::Type& rhs) {
    mulAssignScalar(lhs, rhs);
    return mmove(lhs);
}

template<NDArrayLike T>
inline static void divAssignScalar(T& lhs, const typename T::Type& rhs) {
    baseAssignWithScalar(lhs, rhs, [](auto& lhs, const auto& rhs) { lhs /= rhs; });
}

template<NDArrayLike T>
inline static T divScalar(T lhs, const typename T::Type& rhs) {
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
    if (!isSquare<T>(shape)) {
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
    if (!isSquare<T>(input.shape())) {
        throw Matrix2DError("Only 2D square matrices are invertable");
    }

    Mx result      = identity<T>(input.shape());
    const Size dim = input.shape(0);

    for (Index round = 0; round < 2; ++round) {
        const bool isBottomTriangle = round == 0;
        const Index leadingStart    = isBottomTriangle ? 0 : dim - 1;
        const Index cycleEnd        = isBottomTriangle ? dim : 0;
        const auto cycleEndCompare  = isBottomTriangle ? [](const Index lhs, const Index rhs) { return lhs < rhs; }
                                                       : [](const Index lhs, const Index rhs) { return lhs >= rhs; };
        const auto cycleAdvance     = isBottomTriangle ? [](Index& value) { ++value; } : [](Index& value) { --value; };
        const Index remainingOffset = isBottomTriangle ? 1 : -1;

        for (Index leading = leadingStart; cycleEndCompare(leading, cycleEnd); cycleAdvance(leading)) {
            const auto leadingRowSelect = typename T::SliceShape{IR::single(leading), IR::e2e()};
            NDArraySlice<T> leadInputRow(input, leadingRowSelect);
            NDArraySlice<T> leadResultRow(result, leadingRowSelect);
            const typename T::Type leadValue = 1.0 / leadInputRow[{0, leading}];
            mulAssignScalar(leadInputRow, leadValue);
            mulAssignScalar(leadResultRow, leadValue);

            for (Index remaining = leading + remainingOffset; cycleEndCompare(remaining, cycleEnd);
                 cycleAdvance(remaining)) {
                const auto remainingRowSelect = typename T::SliceShape{IR::single(remaining), IR::e2e()};
                NDArraySlice<T> remainingInputRow(input, remainingRowSelect);
                NDArraySlice<T> remainingResultRow(result, remainingRowSelect);
                const auto remainingValue = remainingInputRow[{0, leading}];
                subAssign(remainingInputRow, mulScalar(leadInputRow.materialize(), remainingValue));
                subAssign(remainingResultRow, mulScalar(leadResultRow.materialize(), remainingValue));
            }
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
* Matrix multiplication
* Cross product
* Axis, rotation matrix creation
*/

}  // namespace nykdtb::nda

#endif