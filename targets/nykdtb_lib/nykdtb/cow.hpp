#ifndef NYKDTB_COW_HPP
#define NYKDTB_COW_HPP

#include "nykdtb/types.hpp"

namespace nykdtb {

template<typename T>
class CowPtr {
public:
    CowPtr()
        : mp(nullptr) {}

    CowPtr(SharedPtr<T> p)
        : mp(mmove(p)) {}

    CowPtr(const CowPtr& p)            = default;
    CowPtr(CowPtr&& p)                 = default;
    CowPtr& operator=(const CowPtr& p) = default;
    CowPtr& operator=(CowPtr&& p)      = default;

    inline const T& use() const { return *mp; }
    inline const T& ref() { return *mp; }
    inline const T& ref() const { return *mp; }
    inline const T* operator*() const { return mp.get(); }
    inline const T* operator->() const { return mp.get(); }

    inline T& use() {
        cow();
        return *mp;
    }

    inline T* operator*() {
        cow();
        return *mp;
    }

    inline T* operator->() {
        cow();
        return mp.get();
    }

    inline SharedPtr<T> takeShared() { return {mmove(mp)}; }

private:
    void cow() {
        if (mp.use_count() > 1) {
            mp = makeShared<T>(mp);
        }
    }

private:
    SharedPtr<T> mp;
};

}  // namespace nykdtb

#endif