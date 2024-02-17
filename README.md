# NYKDTB
This is simple C++20 base library used for some personal projects.

It implements a couple of functionalities not present in the C++20 STL that proved useful.

## Features
### PartialStackStorageVector
Vector implementation with the following features
* Use stack allocation until a limit specified in template argument
  * Storage is moved to heap when limit is reached
  * Storage is moved back to stack if enough elements are removed to reach limit.
* Explicitly specified storage alignment
  * Both for stack and heap allocations
  * Uses `std::assume_aligned` functionality to allow for aligned storage compiler optimizations

### n-dimension arrays
`NDArray` implementations similar in concept to Python `numpy` library arrays.
* All implementations conform to the `NDArrayLike` concept used in the library.
  * This concept is also used for the implementation of operations on the arrays
* There is a dynamic implementation through the `NDArrayBase` type.
  * Configurable template parameters
    * `STACK_SHAPE_SIZE` - using `PartialStackStorageVector` the amount of shape values to store on the stack
    * `STACK_SIZE` - The amount of elements to keep on the stack before moving the array to heap storage.
    * `STORAGE_ALIGNMENT` - How to align the internal storage
  * This may be assigned any size and shape during runtime
* There is a static implementation with dimensions known at compile time: `NDArrayStatic`
* There is a slice implementation called `NDArraySlice` that allows slicing of any `NDArrayLike` object.
  * For each dimension of the NDArray a index range may be specified. Hence the underlying memory does not need to be contiguous.

Operations are implemented in a separate header: `ndarray_ops.hpp`. This includes the following:
* Element-wise arithmetic operations
* 2D Matrix inverse
* 2D Matrix multiplication

### Command line argument parsing
Simple helper class to parse command line arguments for an application.

### Copy-on-write pointer
Pointer implementation to allow owned object passing with lazy-copy mechanism.

### Mutable move
This is an "improvement" on the std::move. It ensures through static assertion that the r-value reference cast happens on a mutable object or reference. This is to avoid accidental copies when using a `const` first development mindset for variables.


## Usage

Recommended usage is through CMake as that is the build system used for this project.

```CMake
include(FetchContent)

FetchContent_Declare(
  nykdtb
  GIT_REPOSITORY git@github.com:nyari/nykdtb.git
  GIT_TAG <tag to use>
)

FetchContent_MakeAvailable(nykdtb nykdtb)

# ...

target_link_libraries(<target_using_the_ligrary>
  PUBLIC nykdtb_lib
  # ...
)
```

Other build environments are not supported yet.