#include "nykdtb/ndarray.hpp"

#include "catch2/catch.hpp"
#include "nykdtb/ndarray_ops.hpp"

using namespace nykdtb;

using TestArray = NDArray<float>;
using TestSlice = NDArraySlice<TestArray>;

TEST_CASE("NDArray default construct", "[ndarray]") {
    TestArray arr;

    REQUIRE(arr.empty());
    REQUIRE(arr.size() == 0);
    REQUIRE(arr.shape() == TestArray::Shape{});
}

TEST_CASE("NDArray with one element", "[ndarray]") {
    TestArray arr({1});

    REQUIRE(!arr.empty());
    REQUIRE(arr.size() == 1);
    REQUIRE(arr.shape() == TestArray::Shape{1});

    REQUIRE(arr[0] == 1);
    REQUIRE(arr[TestArray::Position{0}] == 1);
}

TEST_CASE("NDArray with two elements and correct shape", "[ndarray]") {
    TestArray arr({1, 2}, {2});

    REQUIRE(!arr.empty());
    REQUIRE(arr.size() == 2);
    REQUIRE(arr.shape() == TestArray::Shape{2});

    REQUIRE(arr[0] == 1);
    REQUIRE(arr[1] == 2);

    REQUIRE(arr[TestArray::Position{0}] == 1);
    REQUIRE(arr[TestArray::Position{1}] == 2);
}

TEST_CASE("NDArray with four elements and correct 2D shape", "[ndarray]") {
    TestArray arr({1, 2, 3, 4}, {2, 2});

    REQUIRE(!arr.empty());
    REQUIRE(arr.size() == 4);
    REQUIRE(arr.shape() == TestArray::Shape{2, 2});
    REQUIRE(arr.strides() == TestArray::Shape{2, 1});

    REQUIRE(arr[0] == 1);
    REQUIRE(arr[1] == 2);
    REQUIRE(arr[2] == 3);
    REQUIRE(arr[3] == 4);

    REQUIRE(arr[{0, 0}] == 1);
    REQUIRE(arr[{0, 1}] == 2);
    REQUIRE(arr[{1, 0}] == 3);
    REQUIRE(arr[{1, 1}] == 4);
}

TEST_CASE("NDArray with four elements and incorrect 2D shape", "[ndarray]") {
    REQUIRE_THROWS_AS(TestArray({1, 2, 3, 4}, {2, 1}), TestArray::ShapeDoesNotMatchSize);
}

TEST_CASE("NDArray calculateStrides tests", "[ndarray]") {
    SECTION("One dimensional shape") {
        REQUIRE(TestArray::calculateStrides({7}) == TestArray::Strides{1});
    }
    SECTION("Multi dimensional shape") {
        REQUIRE(TestArray::calculateStrides({7, 5, 3, 2}) == TestArray::Strides{30, 6, 2, 1});
    }
}

TEST_CASE("NDArraySlice calculateRawIndex") {
    using IR = IndexRange;
    SECTION("One dimensional shape E2E slice") {
        REQUIRE(TestSlice::calculateRawIndexFromSliceIndexUnchecked({1}, {1}, {IR::e2e()}, 100) == 100);
    }
    SECTION("One dimensional shape narrowed slice") {
        REQUIRE(TestSlice::calculateRawIndexFromSliceIndexUnchecked({1}, {1}, {IR::between(10, 100)}, 50) == 60);
    }
    SECTION("Four dimensional shape E2E slice") {
        TestSlice::Shape originalShape{5, 3, 2};
        auto originalStrides = TestSlice::NDArray::calculateStrides(originalShape);

        TestSlice::SliceShape sliceShape{IR::e2e(), IR::e2e(), IR::e2e()};
        auto calcShape   = TestSlice::calculateShape(originalShape, sliceShape);
        auto calcStrides = TestSlice::NDArray::calculateStrides(calcShape);

        REQUIRE(TestSlice::calculateRawIndexFromSliceIndexUnchecked(originalStrides, calcStrides, sliceShape, 10) ==
                10);
    }
    SECTION("Three dimensional shape narrow slices slice") {
        TestSlice::Shape originalShape{5, 3, 2};
        auto originalStrides = TestSlice::NDArray::calculateStrides(originalShape);

        TestSlice::SliceShape sliceShape{IR::between(2, 4), IR::single(1), IR::e2e()};
        auto calcShape   = TestSlice::calculateShape(originalShape, sliceShape);
        auto calcStrides = TestSlice::NDArray::calculateStrides(calcShape);

        REQUIRE(TestSlice::calculateRawIndexFromSliceIndexUnchecked(originalStrides, calcStrides, sliceShape, 3) == 21);
    }
}

TEST_CASE("NDArray add same shape arrays", "[ndarray]") {
    TestArray arr1({1, 2, 3, 4}, {2, 2});
    TestArray arr2({4, 3, 2, 1}, {2, 2});

    nda::addAssign(arr1, arr2);

    for (Index i = 0; i < 4; ++i) {
        REQUIRE(arr1[i] == 5);
    }
}

TEST_CASE("NDArray add slice to array", "[ndarray]") {
    TestArray arr1({1, 2, 3, 4}, {2, 2});
    TestArray arr2({4, 3, 2, 1, 1, 2, 3, 4}, {4, 2});

    SECTION("Slice begin") {
        nda::addAssign(arr1, TestSlice{arr2, {IndexRange::until(2), IndexRange::e2e()}});

        for (Index i = 0; i < 4; ++i) {
            REQUIRE(arr1[i] == 5);
        }
    }
    SECTION("Slice end") {
        nda::addAssign(arr1, TestSlice{arr2, {IndexRange::after(2), IndexRange::e2e()}});
        for (Index i = 0; i < 4; ++i) {
            REQUIRE(arr1[i] == ((i + 1) * 2));
        }
    }
}