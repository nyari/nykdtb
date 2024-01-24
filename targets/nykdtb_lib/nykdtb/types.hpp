#ifndef NYKDTB_TYPE_HPP
#define NYKDTB_TYPE_HPP

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nykdtb {

using Index        = int32_t;
using Size         = int32_t;
using DefaultFloat = float;

template<typename T>
using Vec = std::vector<T>;

template<typename Key, typename Value>
using UnorderedMap = std::unordered_map<Key, Value>;

template<typename Key>
using UnorderedSet = std::unordered_set<Key>;

template<typename T>
using Optional                = std::optional<T>;
static constexpr auto nullopt = std::nullopt;

template<typename T>
using SharedPtr = std::shared_ptr<T>;

template<typename T>
inline static constexpr std::remove_reference_t<T>&& mmove(T&& subject) noexcept {
    static_assert(!std::is_const_v<std::remove_reference_t<T>>,
                  "Trying to move const reference. Is this intended? If yes, use std::move.");
    return static_cast<std::remove_reference_t<T>&&>(subject);
}

template<typename T, typename... Args>
static constexpr inline SharedPtr<T> makeShared(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

template<typename T>
using UniquePtr = std::unique_ptr<T>;

template<typename T, typename... Args>
static constexpr inline UniquePtr<T> makeUnique(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

namespace type_info {

using Ref = std::reference_wrapper<const std::type_info>;

struct Hasher {
    std::size_t operator()(Ref code) const { return code.get().hash_code(); }
};

struct EqualTo {
    bool operator()(Ref lhs, Ref rhs) const { return lhs.get() == rhs.get(); }
};

}  // namespace type_info

#define NYKDTB_DEFINE_EXCEPTION_CLASS(classname, superclass)                                        \
    struct classname : public superclass {                                                           \
        classname()                                                                                  \
            : superclass(std::string("[") + std::string(#classname) + std::string("]")) {}           \
        classname(const std::string& message)                                                        \
            : superclass(std::string("[") + std::string(#classname) + std::string("]") + message) {} \
    };

NYKDTB_DEFINE_EXCEPTION_CLASS(RuntimeException, std::runtime_error);
NYKDTB_DEFINE_EXCEPTION_CLASS(LogicException, std::logic_error);
NYKDTB_DEFINE_EXCEPTION_CLASS(NotImplementedException, std::logic_error);

namespace consts {
static constexpr float PIf  = 3.1415926535897932384626433F;
static constexpr double PId = 3.1415926535897932384626433;
}  // namespace consts

}  // namespace nykdtb

#endif