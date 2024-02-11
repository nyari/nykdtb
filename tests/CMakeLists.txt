include(FetchContent)
include(CTest)

FetchContent_Declare(
  catch
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.6)

# CMake 3.14+
FetchContent_MakeAvailable(catch)

add_executable(nykdtb_tests
main.cpp
psvector.cpp
ndarray.cpp
ndarray_static.cpp
ndarray_ops.cpp
)

set_property(TARGET nykdtb_tests PROPERTY CXX_STANDARD 20)

target_link_libraries(nykdtb_tests
  PRIVATE
  Catch2::Catch2
  nykdtb_lib)

add_test(NAME nykdtb_tests COMMAND nykdtb_tests)
