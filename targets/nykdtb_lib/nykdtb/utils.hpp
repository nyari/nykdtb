#ifndef NYKDTB_UTILS_HPP
#define NYKDTB_UTILS_HPP

namespace nykdtb {

template<typename T>
static constexpr T sign(const T val) {
    if (val < T(0)) {
        return T(-1);
    } else if (val > T(0)) {
        return T(1);
    }
    return T(0);
}

template<typename T>
static constexpr bool betweenCO(const T& lhs, const T& val, const T& rhs) {
    return lhs <= val && val < rhs;
}

}  // namespace nykdtb

#endif