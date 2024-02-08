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
