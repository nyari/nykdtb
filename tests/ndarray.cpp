#include "nykdtb/ndarray.hpp"

#include "catch2/catch.hpp"

using namespace nykdtb;

using TestArray = NDArray<float>;

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