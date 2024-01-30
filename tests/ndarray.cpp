#include "nykdtb/ndarray.hpp"

#include "catch2/catch.hpp"

using namespace nykdtb;

TEST_CASE("NDArray default construct", "[ndarray]") {
    NDArray arr;

    REQUIRE(arr.empty());
    REQUIRE(arr.size() == 0);
    REQUIRE(arr.shape() == NDArray::Shape{});
}

TEST_CASE("NDArray with one element", "[ndarray]") {
    NDArray<float> arr({1});

    REQUIRE(!arr.empty());
    REQUIRE(arr.size() == 1);
    REQUIRE(arr.shape() == NDArray::Shape{1});

    REQUIRE(arr[0] == 1);

    REQUIRE(arr[{0}] == 1);
}

TEST_CASE("NDArray with two elements and correct shape", "[ndarray]") {
    NDArray<float> arr({1, 2}, {2});

    REQUIRE(!arr.empty());
    REQUIRE(arr.size() == 2);
    REQUIRE(arr.shape() == NDArray::Shape{2});

    REQUIRE(arr[0] == 1);
    REQUIRE(arr[1] == 2);

    REQUIRE(arr[{0}] == 1);
    REQUIRE(arr[{1}] == 2);
}

TEST_CASE("NDArray with two elements and correct shape", "[ndarray]") {
    NDArray<float> arr({1, 2, 3, 4}, {2, 2});

    REQUIRE(!arr.empty());
    REQUIRE(arr.size() == 4);
    REQUIRE(arr.shape() == NDArray::Shape{2, 2});

    REQUIRE(arr[0] == 1);
    REQUIRE(arr[1] == 2);
    REQUIRE(arr[2] == 3);
    REQUIRE(arr[3] == 4);

    REQUIRE(arr[{0, 0}] == 1);
    REQUIRE(arr[{0, 1}] == 2);
    REQUIRE(arr[{1, 0}] == 3);
    REQUIRE(arr[{1, 1}] == 4);
}

TEST_CASE("NDArray with two elements and incorrect shape", "[ndarray]") {
    REQUIRE_THROWS_AS(NDArray<float> arr({1, 2, 3, 4}, {2, 1}), NDArray<float>::ShapeDoesNotMatchSize);
}