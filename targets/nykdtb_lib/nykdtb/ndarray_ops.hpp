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
inline static typename T::Type magnitude(const T& elem) {
    typename T::Type lengthsq = 0;
    for (const auto& e : elem) {
        lengthsq += e * e;
    }

    return std::sqrt(lengthsq);
}

template<NDArrayLike T>
inline static void normalize(T& elem) {
    const auto mag = magnitude(elem);
    if (mag == 0) {
        throw DivisionByZero();
    }
    auto mtp = 1.0F / mag;
    for (auto& e : elem) {
        e *= mtp;
    }
}

template<NDArrayLike T>
inline static T normalized(T elem) {
    normalize(elem);
    return mmove(elem);
}

template<NDArrayLike LHS, NDArrayLike RHS>
inline static typename LHS::Type dot(const LHS& lhs, const RHS& rhs) {
    static_assert(std::is_same_v<typename LHS::Type, typename RHS::Type>,
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

template<NDArrayLike LHS, NDArrayLike RHS>
inline static bool eq(const LHS& lhs, const RHS& rhs) {
    if (lhs.shape() != rhs.shape()) {
        return false;
    }

    auto lhsBegin = lhs.begin();
    auto lhsEnd   = lhs.end();
    auto rhsIt    = rhs.begin();

    for (auto lhsIt = lhsBegin; lhsIt < lhsEnd; ++lhsIt, ++rhsIt) {
        if ((*lhsIt) != (*rhsIt)) {
            return false;
        }
    }

    return true;
}

namespace d2 {

NYKDTB_DEFINE_EXCEPTION_CLASS(Matrix2DError, RuntimeException)

template<NDArrayLike T>
inline static bool is2d(const typename T::Shape& shape) {
    return (shape.dims() == 2);
}

template<NDArrayLike T>
inline static bool isSquare(const typename T::Shape& shape) {
    return is2d<T>(shape) && (shape[0] == shape[1]);
}

template<NDArrayLike T>
inline static typename T::MaterialType identity(const typename T::Shape& shape) {
    using Mx = typename T::MaterialType;
    if (!isSquare<T>(shape)) {
        throw Matrix2DError("Only 2D square matrices have identity");
    }

    Mx result(Mx::zeros(shape));

    for (Index i = 0; i < shape[0]; ++i) {
        result[{i, i}] = 1;
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

template<NDArrayLike LHS, NDArrayLike RHS, Size INTERLEAVE = 4>
inline static typename LHS::MaterialType matMul(const LHS& lhs, const RHS& rhs) {
    if (!is2d<LHS>(lhs.shape()) || !is2d<RHS>(rhs.shape())) {
        throw Matrix2DError("Only 2D matrices are multipliable");
    }
    if (lhs.shape(1) != rhs.shape(0)) {
        throw Matrix2DError("Incorrect shape for matrix multiplication");
    }

    const typename LHS::Shape resultShape{lhs.shape(0), rhs.shape(1)};
    auto result = LHS::MaterialType::zeros(resultShape);

    for (Index resultRow = 0; resultRow < resultShape[0]; ++resultRow) {
        for (Index resultColumn = 0; resultColumn < resultShape[1]; ++resultColumn) {
            for (Index sourceColumn = 0; sourceColumn < lhs.shape(1); ++sourceColumn) {
                const auto lhsValue = lhs[{resultRow, sourceColumn}];
                const auto rhsValue = rhs[{sourceColumn, resultColumn}];
                result[{resultRow, resultColumn}] += lhsValue * rhsValue;
            }
        }
    }

    return mmove(result);
}

template<NDArrayLike LHS, NDArrayLike RHS>
inline static typename LHS::MaterialType cross3(const LHS& a, const RHS& b) {
    const typename LHS::Shape resultShape{1, 3};

    const auto i = a[1] * b[2] - a[2] * b[1];
    const auto j = a[0] * b[2] - a[2] * b[0];
    const auto k = a[0] * b[1] - a[1] * b[0];

    return {{i, j, k}, resultShape};
};

template<NDArrayLike T>
inline static T::MaterialType rotAngleMx(const T& axis, const typename T::Type angle) {
    const auto x = axis[0];
    const auto y = axis[1];
    const auto z = axis[2];
    const auto s = std::sin(angle);
    const auto c = std::cos(angle);
    const auto a = 1 - c;

    return {{c + x * x * a,
             x * y * a - z * s,
             x * z * a + y * s,
             y * x * a + z * s,
             c + y * y * a,
             y * z * a - x * s,
             z * x * a - y * s,
             z * y * a + x * s,
             c + z * z * a},
            {3, 3}};
}

}  // namespace d2

}  // namespace nykdtb::nda

#endif