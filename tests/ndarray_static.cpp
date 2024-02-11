#include <catch2/catch.hpp>

#include "nykdtb/ndarray.hpp"
#include "nykdtb/ndarray_ops.hpp"

using namespace nykdtb;

struct TestNDArrayStaticParams {
    static constexpr Size STORAGE_ALIGNMENT = 512;
};

using TestArray = NDArrayStatic<float, TestNDArrayStaticParams, 4, 4>;

TEST_CASE("NDArray static default construct", "[ndarray][static]") {
    TestArray arr;
    static_cast<void>(arr);
}