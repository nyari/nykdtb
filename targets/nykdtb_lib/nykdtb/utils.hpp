#ifndef NYKDTB_UTILS_HPP
#define NYKDTB_UTILS_HPP

namespace nykdtb {

template<typename T>
static inline constexpr T sign(const T val) {
    if (val < T(0)) {
        return T(-1);
    } else if (val > T(0)) {
        return T(1);
    }
    return T(0);
}

}  // namespace nykdtb

#endif