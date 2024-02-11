#ifndef NYKDTB_UTILS_HPP
#define NYKDTB_UTILS_HPP

#include <type_traits>

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

class IndexRange {
public:
    enum Endpoint { E };
    using EndElement = std::variant<Index, Endpoint>;

    constexpr IndexRange()
        : m_begin(0), m_end(0) {}

    static constexpr IndexRange e2e() { return IndexRange{0, E}; }
    static constexpr IndexRange none() { return IndexRange{0, 0}; }
    static constexpr IndexRange until(EndElement end) { return IndexRange{0, end}; }
    static constexpr IndexRange after(Index begin) { return IndexRange{begin, E}; }
    static constexpr IndexRange between(Index begin, EndElement end) { return IndexRange{begin, end}; }
    static constexpr IndexRange single(Index elem) { return IndexRange{elem, elem + 1}; }

    inline Size effectiveSize(Size maxValue) const { return end(maxValue) - begin(); }

    inline Index begin() const { return m_begin; }
    inline Index end(Size maxValue) const {
        return std::visit(
            [maxValue](auto&& arg) -> Index {
                using T = std::decay_t<decltype(arg)>;
                if (std::is_same_v<T, Index>) {
                    return arg;
                } else {
                    return maxValue;
                }
            },
            m_end);
    }

private:
    Index m_begin;
    EndElement m_end;

private:
    constexpr IndexRange(Index _begin, EndElement _end)
        : m_begin{_begin}, m_end{_end} {}
};

using IR = IndexRange;

template<typename NewArray, typename OldArray>
static constexpr NewArray arrayPrepend(std::remove_cvref_t<decltype(NewArray()[0])> item, const OldArray& input) {
    NewArray result;
    auto resultIt = result.begin();
    *(resultIt++) = mmove(item);
    for (const auto& item : input) {
        *(resultIt++) = item;
    }

    return mmove(result);
}

}  // namespace nykdtb

#endif