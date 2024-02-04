#ifndef NYKDTB_NDARRAY_OPS_HPP
#define NYKDTB_NDARRAY_OPS_HPP

#include "nykdtb/ndarray.hpp"

namespace nykdtb::nda {

NYKDTB_DEFINE_EXCEPTION_CLASS(ShapesDoNotMatch, LogicException);

template<NDArrayLike LHS, NDArrayLike RHS>
void addAssign(LHS& lhs, const RHS& rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw ShapesDoNotMatch();
    }

    for (Index i = 0; i < lhs.size(); ++i) {
        lhs[i] += rhs[i];
    }
}

template<NDArrayLike LHS, NDArrayLike RHS>
LHS add(const LHS& lhs, const RHS& rhs) {
    LHS result{lhs.clone()};
    addAssign(result, rhs);
    return mmove(result);
}

}  // namespace nykdtb::nda

#endif