#include "nykdtb/psvector.hpp"

#include "catch2/catch.hpp"
#include "conttestapp.hpp"

using namespace nykdtb;

using Op = ContainerTestAppliance::Op;

template<Size SIZE>
using TestVec = PSVec<ContainerTestAppliance, SIZE>;

template<Size SIZE>
using RefArr = std::array<ContainerTestAppliance, SIZE>;

using RefVec = std::vector<ContainerTestAppliance>;

TEST_CASE("PSVec DefaultConstruct", "[psvec]") {
    TestVec<4> test;
    REQUIRE(test.empty());
    REQUIRE(test.size() == 0);
    REQUIRE(test.onStack());
}

TEST_CASE("PSVec Init list on stack only 1 element in larger container") {
    TestVec<4> test{{}};

    REQUIRE_FALSE(test.empty());
    REQUIRE(test.size() == 1);
    REQUIRE(test.onStack());
    REQUIRE(test[0].compare({Op::Default, Op::Copy}));
}

TEST_CASE("PSVec Init list on stack only 2 element in 2 sized container") {
    TestVec<2> test{{}, {}};

    REQUIRE_FALSE(test.empty());
    REQUIRE(test.size() == 2);
    REQUIRE(test.onStack());
    REQUIRE(test[0].compare({Op::Default, Op::Copy}));
    REQUIRE(test[1].compare({Op::Default, Op::Copy}));
}

TEST_CASE("PSVec Init list on stack only 3 element in 2 sized container") {
    TestVec<2> test{{}, {}, {}};

    REQUIRE_FALSE(test.empty());
    REQUIRE(test.size() == 3);
    REQUIRE_FALSE(test.onStack());
    REQUIRE(test[0].compare({Op::Default, Op::Copy}));
    REQUIRE(test[1].compare({Op::Default, Op::Copy}));
    REQUIRE(test[2].compare({Op::Default, Op::Copy}));
}

TEST_CASE("PSVec Test emplace_back on stack") {
    TestVec<2> test{{}};

    REQUIRE_FALSE(test.empty());
    REQUIRE(test.size() == 1);
    REQUIRE(test.onStack());
    REQUIRE(test[0].compare({Op::Default, Op::Copy}));
    test.emplace_back();
    REQUIRE(test.size() == 2);
    REQUIRE(test.onStack());
    REQUIRE(test[0].compare({Op::Default, Op::Copy}));
    REQUIRE(test[1].compare({Op::Default}));
}

TEST_CASE("PSVec Test push_back on stack") {
    TestVec<2> test{{}};

    REQUIRE_FALSE(test.empty());
    REQUIRE(test.size() == 1);
    REQUIRE(test.onStack());
    REQUIRE(test[0].compare({Op::Default, Op::Copy}));
    test.push_back({});
    REQUIRE(test.size() == 2);
    REQUIRE(test.onStack());
    REQUIRE(test[0].compare({Op::Default, Op::Copy}));
    REQUIRE(test[1].compare({Op::Default, Op::Moved}));
}

TEST_CASE("PSVec Test emplace_back with move to heap") {
    TestVec<1> test{{}};

    REQUIRE_FALSE(test.empty());
    REQUIRE(test.size() == 1);
    REQUIRE(test.onStack());
    REQUIRE(test[0].compare({Op::Default, Op::Copy}));

    test.emplace_back();
    REQUIRE(test.size() == 2);
    REQUIRE_FALSE(test.onStack());
    REQUIRE(test[0].compare({Op::Default, Op::Copy, Op::Moved}));
    REQUIRE(test[1].compare({Op::Default}));

    SECTION("Reallocate heap, by 3 new items") {
        test.emplace_back();
        test.emplace_back();
        test.emplace_back();
        REQUIRE(test.size() == 5);
        REQUIRE_FALSE(test.onStack());
        REQUIRE(test[0].compare({Op::Default, Op::Copy, Op::Moved, Op::Moved}));
        REQUIRE(test[1].compare({Op::Default, Op::Moved}));
        REQUIRE(test[2].compare({Op::Default, Op::Moved}));
        REQUIRE(test[3].compare({Op::Default, Op::Moved}));
        REQUIRE(test[4].compare({Op::Default}));
    }
}

TEST_CASE("PSVec Test push_back with move to heap") {
    TestVec<1> test{{}};

    REQUIRE_FALSE(test.empty());
    REQUIRE(test.size() == 1);
    REQUIRE(test.onStack());
    REQUIRE(test[0].compare({Op::Default, Op::Copy}));

    test.push_back({});
    REQUIRE(test.size() == 2);
    REQUIRE_FALSE(test.onStack());
    REQUIRE(test[0].compare({Op::Default, Op::Copy, Op::Moved}));
    REQUIRE(test[1].compare({Op::Default, Op::Moved}));
    SECTION("Reallocate heap, by 3 new items") {
        test.push_back({});
        test.push_back({});
        test.push_back({});
        REQUIRE(test.size() == 5);
        REQUIRE_FALSE(test.onStack());
        REQUIRE(test[0].compare({Op::Default, Op::Copy, Op::Moved, Op::Moved}));
        REQUIRE(test[1].compare({Op::Default, Op::Moved, Op::Moved}));
        REQUIRE(test[2].compare({Op::Default, Op::Moved, Op::Moved}));
        REQUIRE(test[3].compare({Op::Default, Op::Moved, Op::Moved}));
        REQUIRE(test[4].compare({Op::Default, Op::Moved}));
    }
}

TEST_CASE("PSVec Erase one element on stack") {
    RefArr<4> ref;
    TestVec<4> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    test.erase(test.begin() + 1);
    REQUIRE(test.onStack());
    REQUIRE(test.size() == 3);
    REQUIRE(ref[0] == test[0]);
    REQUIRE(ref[2] == test[1]);
    REQUIRE(ref[3] == test[2]);
    REQUIRE(ref[0].compare({Op::Default}));
    REQUIRE(ref[1].compare({Op::Default, Op::MoveA}));
    REQUIRE(ref[2].compare({Op::Default, Op::Moved}));
    REQUIRE(ref[3].compare({Op::Default, Op::Moved}));
}

TEST_CASE("PSVec Erase two elements on stack") {
    RefArr<6> ref;
    TestVec<6> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    test.erase(test.begin() + 1, test.begin() + 3);
    REQUIRE(test.onStack());
    REQUIRE(test.size() == 4);
    REQUIRE(ref[0] == test[0]);
    REQUIRE(ref[3] == test[1]);
    REQUIRE(ref[4] == test[2]);
    REQUIRE(ref[5] == test[3]);
    REQUIRE(ref[0].compare({Op::Default}));
    REQUIRE(ref[1].compare({Op::Default, Op::MoveA}));
    REQUIRE(ref[2].compare({Op::Default, Op::MoveA}));
    REQUIRE(ref[3].compare({Op::Default, Op::Moved}));
    REQUIRE(ref[4].compare({Op::Default, Op::Moved}));
    REQUIRE(ref[5].compare({Op::Default, Op::Moved}));
}

TEST_CASE("PSVec Erase two elements on heap with container 2 size") {
    RefArr<6> ref;
    TestVec<2> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    test.erase(test.begin() + 1, test.begin() + 3);
    REQUIRE_FALSE(test.onStack());
    REQUIRE(test.size() == 4);
    REQUIRE(ref[0] == test[0]);
    REQUIRE(ref[3] == test[1]);
    REQUIRE(ref[4] == test[2]);
    REQUIRE(ref[5] == test[3]);
    REQUIRE(ref[0].compare({Op::Default, Op::Moved}));
    REQUIRE(ref[1].compare({Op::Default, Op::Moved, Op::MoveA}));
    REQUIRE(ref[2].compare({Op::Default, Op::MoveA}));
    REQUIRE(ref[3].compare({Op::Default, Op::Moved}));
    REQUIRE(ref[4].compare({Op::Default, Op::Moved}));
    REQUIRE(ref[5].compare({Op::Default, Op::Moved}));
}

TEST_CASE("PSVec Erase two elements on heap with container 4 size, move back to stack") {
    RefArr<6> ref;
    TestVec<4> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    test.erase(test.begin() + 1, test.begin() + 3);
    REQUIRE(test.onStack());
    REQUIRE(test.size() == 4);
    REQUIRE(ref[0] == test[0]);
    REQUIRE(ref[3] == test[1]);
    REQUIRE(ref[4] == test[2]);
    REQUIRE(ref[5] == test[3]);
    REQUIRE(ref[0].compare({Op::Default, Op::Moved, Op::Moved}));
    REQUIRE(ref[1].compare({Op::Default, Op::Moved, Op::MoveA}));
    REQUIRE(ref[2].compare({Op::Default, Op::Moved, Op::MoveA}));
    REQUIRE(ref[3].compare({Op::Default, Op::Moved, Op::Moved, Op::Moved}));
    REQUIRE(ref[4].compare({Op::Default, Op::Moved, Op::Moved}));
    REQUIRE(ref[5].compare({Op::Default, Op::Moved, Op::Moved}));
}

TEST_CASE("PSVec copy construct stack vector") {
    RefArr<2> ref;
    TestVec<2> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    RefVec refV;
    refV.reserve(2);
    {
        TestVec<2> copy(test);
        for (const auto& elem : copy) {
            refV.emplace_back(elem.track());
        }

        REQUIRE(copy.onStack());
        REQUIRE(copy.size() == 2);
        REQUIRE_FALSE(copy.empty());
    }

    REQUIRE(ref[0] == test[0]);
    REQUIRE(ref[1] == test[1]);
    REQUIRE(ref[0].compare({Op::Default, Op::Copied}));
    REQUIRE(ref[1].compare({Op::Default, Op::Copied}));
    REQUIRE(refV[0].compare({Op::Default, Op::Copy, Op::Destructed}));
    REQUIRE(refV[1].compare({Op::Default, Op::Copy, Op::Destructed}));
}

TEST_CASE("PSVec copy construct heap vector") {
    RefArr<2> ref;
    TestVec<1> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    RefVec refV;
    refV.reserve(2);
    {
        TestVec<1> copy(test);
        for (const auto& elem : copy) {
            refV.emplace_back(elem.track());
        }

        REQUIRE_FALSE(copy.onStack());
        REQUIRE(copy.size() == 2);
        REQUIRE_FALSE(copy.empty());
    }
    REQUIRE(ref[0] == test[0]);
    REQUIRE(ref[1] == test[1]);
    REQUIRE(ref[0].compare({Op::Default, Op::Moved, Op::Copied}));
    REQUIRE(ref[1].compare({Op::Default, Op::Copied}));
    REQUIRE(refV[0].compare({Op::Default, Op::Moved, Op::Copy, Op::Destructed}));
    REQUIRE(refV[1].compare({Op::Default, Op::Copy, Op::Destructed}));
}

TEST_CASE("PSVec move construct stack vector") {
    RefArr<2> ref;
    TestVec<2> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    RefVec refV;
    refV.reserve(2);
    {
        TestVec<2> copy(mmove(test));
        REQUIRE(test.empty());
        for (const auto& elem : copy) {
            refV.emplace_back(elem.track());
        }

        REQUIRE(copy.onStack());
        REQUIRE(copy.size() == 2);
        REQUIRE_FALSE(copy.empty());
    }

    REQUIRE(ref[0] == refV[0]);
    REQUIRE(ref[1] == refV[1]);
    REQUIRE(ref[0].compare({Op::Default, Op::Moved, Op::Destructed}));
    REQUIRE(ref[1].compare({Op::Default, Op::Moved, Op::Destructed}));
}

TEST_CASE("PSVec move construct heap vector") {
    RefArr<2> ref;
    TestVec<1> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    RefVec refV;
    refV.reserve(2);
    {
        TestVec<1> copy(mmove(test));
        REQUIRE(test.empty());
        for (const auto& elem : copy) {
            refV.emplace_back(elem.track());
        }

        REQUIRE_FALSE(copy.onStack());
        REQUIRE(copy.size() == 2);
        REQUIRE_FALSE(copy.empty());
    }

    REQUIRE(ref[0] == refV[0]);
    REQUIRE(ref[1] == refV[1]);
    REQUIRE(ref[0].compare({Op::Default, Op::Moved, Op::Destructed}));
    REQUIRE(ref[1].compare({Op::Default, Op::Destructed}));
}

TEST_CASE("PSVec copy assign construct stack vector") {
    RefArr<2> ref;
    TestVec<2> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    RefVec refV;
    refV.reserve(2);
    RefVec refO;
    refO.reserve(2);
    {
        TestVec<2> copy{{}};
        for (const auto& elem : copy) {
            refO.emplace_back(elem.track());
        }
        copy = test;
        for (const auto& elem : copy) {
            refV.emplace_back(elem.track());
        }

        REQUIRE(copy.onStack());
        REQUIRE(copy.size() == 2);
        REQUIRE_FALSE(copy.empty());
    }

    REQUIRE(refO[0].compare({Op::Default, Op::Copy, Op::CopyA}));
    REQUIRE(ref[0] == test[0]);
    REQUIRE(ref[1] == test[1]);
    REQUIRE(ref[0].compare({Op::Default, Op::Copied}));
    REQUIRE(ref[1].compare({Op::Default, Op::Copied}));
    REQUIRE(refV[0].compare({Op::Default, Op::Copy, Op::Destructed}));
    REQUIRE(refV[1].compare({Op::Default, Op::Copy, Op::Destructed}));
}

TEST_CASE("PSVec copy assign construct heap vector") {
    RefArr<2> ref;
    TestVec<1> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    RefVec refV;
    refV.reserve(2);
    RefVec refO;
    refO.reserve(2);
    {
        TestVec<1> copy{{}};
        for (const auto& elem : copy) {
            refO.emplace_back(elem.track());
        }
        copy = test;
        for (const auto& elem : copy) {
            refV.emplace_back(elem.track());
        }

        REQUIRE_FALSE(copy.onStack());
        REQUIRE(copy.size() == 2);
        REQUIRE_FALSE(copy.empty());
    }

    // TODO: The "Moved" operation is inefficient here. Since we already need to move it, we could just destruct it in
    // place and then allocate heap
    REQUIRE(refO[0].compare({Op::Default, Op::Copy, Op::Moved, Op::CopyA}));
    REQUIRE(ref[0] == test[0]);
    REQUIRE(ref[1] == test[1]);
    REQUIRE(ref[0].compare({Op::Default, Op::Moved, Op::Copied}));
    REQUIRE(ref[1].compare({Op::Default, Op::Copied}));
    REQUIRE(refV[0].compare({Op::Default, Op::Moved, Op::Copy, Op::Destructed}));
    REQUIRE(refV[1].compare({Op::Default, Op::Copy, Op::Destructed}));
}

TEST_CASE("PSVec move assign construct stack vector") {
    RefArr<2> ref;
    TestVec<2> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    RefVec refV;
    refV.reserve(2);
    RefVec refO;
    refO.reserve(2);
    {
        TestVec<2> copy{{}};
        for (const auto& elem : copy) {
            refO.emplace_back(elem.track());
        }
        copy = mmove(test);
        for (const auto& elem : copy) {
            refV.emplace_back(elem.track());
        }

        REQUIRE(copy.onStack());
        REQUIRE(copy.size() == 2);
        REQUIRE_FALSE(copy.empty());
    }

    REQUIRE(refO[0].compare({Op::Default, Op::Copy, Op::MoveA}));
    REQUIRE(ref[0] == refV[0]);
    REQUIRE(ref[1] == refV[1]);
    REQUIRE(ref[0].compare({Op::Default, Op::Moved, Op::Destructed}));
    REQUIRE(ref[1].compare({Op::Default, Op::Moved, Op::Destructed}));
}

TEST_CASE("PSVec move assign construct heap vector") {
    RefArr<2> ref;
    TestVec<1> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    RefVec refV;
    refV.reserve(2);
    RefVec refO;
    refO.reserve(2);
    {
        TestVec<1> copy{{}};
        for (const auto& elem : copy) {
            refO.emplace_back(elem.track());
        }
        copy = mmove(test);
        for (const auto& elem : copy) {
            refV.emplace_back(elem.track());
        }

        REQUIRE_FALSE(copy.onStack());
        REQUIRE(copy.size() == 2);
        REQUIRE_FALSE(copy.empty());
    }

    REQUIRE(refO[0].compare({Op::Default, Op::Copy, Op::Destructed}));
    REQUIRE(ref[0] == refV[0]);
    REQUIRE(ref[1] == refV[1]);
    REQUIRE(ref[0].compare({Op::Default, Op::Moved, Op::Destructed}));
    REQUIRE(ref[1].compare({Op::Default, Op::Destructed}));
}


TEST_CASE("PSVec insert one element on stack") {
    RefArr<2> ref;
    TestVec<4> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    ContainerTestAppliance inserted;

    test.insert(test.ptr(1), inserted);
    REQUIRE(test.onStack());
    REQUIRE(test.size() == 3);
    REQUIRE(ref[0] == test[0]);
    REQUIRE(ref[1] == test[2]);
    REQUIRE(ref[0].compare({Op::Default}));
    REQUIRE(ref[1].compare({Op::Default, Op::Moved}));
    REQUIRE(test[1].compare({Op::Default, Op::Copy}));
    REQUIRE(inserted.compare({Op::Default, Op::Copied}));
}

TEST_CASE("PSVec insert one element on heap") {
    RefArr<2> ref;
    TestVec<1> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    ContainerTestAppliance inserted;

    test.insert(test.ptr(1), inserted);
    REQUIRE_FALSE(test.onStack());
    REQUIRE(test.size() == 3);
    REQUIRE(ref[0] == test[0]);
    REQUIRE(ref[1] == test[2]);
    REQUIRE(ref[0].compare({Op::Default, Op::Moved}));
    REQUIRE(ref[1].compare({Op::Default, Op::Moved}));
    REQUIRE(test[1].compare({Op::Default, Op::Copy}));
    REQUIRE(inserted.compare({Op::Default, Op::Copied}));
}

TEST_CASE("PSVec insert one element while moving from stack to heap") {
    RefArr<2> ref;
    TestVec<2> test;
    for (const auto& r : ref) {
        test.emplace_back(r.track());
    }

    ContainerTestAppliance inserted;

    test.insert(test.ptr(1), inserted);
    REQUIRE_FALSE(test.onStack());
    REQUIRE(test.size() == 3);
    REQUIRE(ref[0] == test[0]);
    REQUIRE(ref[1] == test[2]);
    REQUIRE(ref[0].compare({Op::Default, Op::Moved}));
    REQUIRE(ref[1].compare({Op::Default, Op::Moved, Op::Moved}));
    REQUIRE(test[1].compare({Op::Default, Op::Copy}));
    REQUIRE(inserted.compare({Op::Default, Op::Copied}));
}
