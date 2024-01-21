#ifndef NYKDTB_CONTTESETAPP_HPP
#define NYKDTB_CONTTESETAPP_HPP

#include <memory>

#include "nykdtb/types.hpp"

namespace nykdtb {

class ContainerTestAppliance {
public:
    enum class Op { Default, Copy, Copied, CopyA, Move, Moved, MoveA, Destructed };
    using OpList = ArrayList<Op>;

public:
    inline ContainerTestAppliance()
        : m_Ops(makeShared<OpList>()) {
        m_Ops->emplace_back(Op::Default);
    }

    inline ContainerTestAppliance(SharedPtr<OpList> input)
        : m_Ops(mmove(input)) {}

    inline ContainerTestAppliance(const ContainerTestAppliance& other)
        : m_Ops(makeShared<OpList>(*other.m_Ops)) {
        m_Ops->emplace_back(Op::Copy);
        other.m_Ops->emplace_back(Op::Copied);
    }

    inline ContainerTestAppliance(ContainerTestAppliance&& other)
        : m_Ops(mmove(other.m_Ops)) {
        m_Ops->emplace_back(Op::Moved);
    }

    inline ContainerTestAppliance& operator=(const ContainerTestAppliance& other) {
        m_Ops->emplace_back(Op::CopyA);
        m_Ops = makeShared<OpList>(*other.m_Ops);
        m_Ops->emplace_back(Op::Copy);
        other.m_Ops->emplace_back(Op::Copied);
        return *this;
    }

    inline ContainerTestAppliance& operator=(ContainerTestAppliance&& other) {
        m_Ops->emplace_back(Op::MoveA);
        other.m_Ops->emplace_back(Op::Moved);
        m_Ops = mmove(other.m_Ops);
        return *this;
    }

    inline ~ContainerTestAppliance() {
        if (m_Ops != nullptr) {
            m_Ops->emplace_back(Op::Destructed);
        }
    }

    SharedPtr<OpList> track() const { return m_Ops; }

    bool compare(const std::vector<Op>& refOps) { return *m_Ops == refOps; }

    bool operator==(const ContainerTestAppliance& other) const { return m_Ops == other.m_Ops; }

private:
    SharedPtr<OpList> m_Ops;
};

}  // namespace nykdtb

#endif