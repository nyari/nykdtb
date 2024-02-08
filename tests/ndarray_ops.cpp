#include "nykdtb/ndarray_ops.hpp"

#include <catch2/catch.hpp>

using namespace nykdtb;

using TestArray = NDArray<float>;
using TestSlice = NDArraySlice<TestArray>;

TEST_CASE("NDArray matrix inverse", "[ndarray][matrix]") {
    TestArray arr{{1, 2, 3, 4}, {2, 2}};
    auto result = nda::d2::inverse(arr.clone());
    REQUIRE(result[{0, 0}] == -2);
    REQUIRE(result[{0, 1}] == 1);
    REQUIRE(result[{1, 0}] == 1.5);
    REQUIRE(result[{1, 1}] == -0.5);
}

TEST_CASE("NDArray matrix multiplication", "[ndarray][matrix]") {
    /*
                6   5   4
                3   2   1
        1   2   12  9   6
        3   4   30  23  16
        5   6   48  37  26
    */

    const TestArray lhs{{1, 2, 3, 4, 5, 6}, {3, 2}};
    const TestArray rhs{{6, 5, 4, 3, 2, 1}, {2, 3}};

    const auto result = nda::d2::matMul(lhs, rhs);

    REQUIRE(nda::eq(result, TestArray{{12, 9, 6, 30, 23, 16, 48, 37, 26}, {3, 3}}));
}

TEST_CASE("NDArray cross procuct", "[ndarray][matrix]") {
    const TestArray lhs{{1, 0, 0}, {1, 3}};
    const TestArray rhs{{0, 1, 0}, {1, 3}};

    const auto result = nda::d2::cross3(lhs, rhs);

    REQUIRE(nda::eq(result, TestArray{{0, 0, 1}, {1, 3}}));
}
