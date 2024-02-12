#include <catch2/catch.hpp>

#include "nykdtb/ndarray.hpp"
#include "nykdtb/ndarray_ops.hpp"

using namespace nykdtb;

struct TestNDArrayStaticParams {
    static constexpr Size STORAGE_ALIGNMENT = 256;
};

template<Size... sizes>
using StaticTestArray  = NDArrayStatic<float, TestNDArrayStaticParams, sizes...>;
using DynamicTestArray = NDArray<float>;

TEST_CASE("NDArray default constructed", "[ndarray][static]") {
    using Arr = StaticTestArray<2, 4, 3>;

    REQUIRE(Arr::Meta::storageSize == 24);
    REQUIRE(Arr::Meta::depth == 3);
    REQUIRE(Arr::Meta::stride == 12);
    REQUIRE(Arr::Meta::strides == Arr::Meta::DimensionStorage{12, 3, 1});
    REQUIRE(Arr::Meta::shape == Arr::Meta::DimensionStorage{2, 4, 3});
}

TEST_CASE("NDArray static matrix multiplication", "[ndarray][static]") {
    const DynamicTestArray lhs{{1, 2, 3, 4, 5, 6}, {3, 2}};
    const StaticTestArray<2, 4> rhs{{6, 5, 4, -1, 3, 2, 1, -2}};

    const auto result = nda::d2::matMul(lhs, rhs);

    REQUIRE(nda::eq(result, DynamicTestArray{{12, 9, 6, -5, 30, 23, 16, -11, 48, 37, 26, -17}, {3, 4}}));
}

TEST_CASE("NDArray static matMul", "[ndarray][static]") {
    const DynamicTestArray lhs{{1, 2, 3, 4, 5, 6}, {3, 2}};
    const StaticTestArray<2, 4> rhs{{6, 5, 4, -1, 3, 2, 1, -2}};

    const auto result = nda::d2::matMul(lhs, rhs);

    REQUIRE(nda::eq(result, DynamicTestArray{{12, 9, 6, -5, 30, 23, 16, -11, 48, 37, 26, -17}, {3, 4}}));
}

TEST_CASE("NDArray static matrix inverse", "[ndarray][static]") {
    StaticTestArray<2, 2> arr{{1, 2, 3, 4}};
    auto result = nda::d2::inverse(arr.clone());
    REQUIRE(result[{0, 0}] == -2);
    REQUIRE(result[{0, 1}] == 1);
    REQUIRE(result[{1, 0}] == 1.5);
    REQUIRE(result[{1, 1}] == -0.5);
}