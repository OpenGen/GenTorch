# GenTorch

This repository is based on the hypothesis that a port of [Gen](gen.dev) to C++, using LibTorch for numerical types and operations and automatic differentiation, will be useful for industry research use cases, will be a potential backend for a port of Gen to Python, will provide a balance of ergonomics and performance for learning and inference in structured generative models with stochastic structure, and and will be useful for real-time applications in robotics, computer vision, perception, and human-understanding.

## External dependencies

### C++ compiler supporting C++17

For example, either [clang++-13](https://clang.llvm.org/get_started.html) or [g++-11](https://gcc.gnu.org/).

### CMake 3.7 or above

See [Installing CMake](https://cmake.org/install/)

### libtorch

This library depends on the C++ distribution of PyTorch, called [libtorch](https://pytorch.org/cppdocs/).

So far, the code has only been tested with the CPU-version of libtorch (version 1.9.1+cpu).

Follow instructions at [Installing C++ distributions of PyTorch](https://pytorch.org/cppdocs/installing.html) to install LibTorch, and record the absolute path to the resulting `libtorch` directory on your filesystem.

### Doxygen

This is only required for generating documentation.

See [Doxygen Installation](https://www.doxygen.nl/manual/install.html).

## Building 

Currently this project builds a static library, tests, and documentation.

To build, either open the `libgen-experimental` directory in a C++ IDE (e.g. [https://www.jetbrains.com/clion/](CLion)) or use `cmake` directly.
In either case, you need to add an option `-DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch` when configuring cmake.

To configure:
```shell
cmake -S . -B build -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch
```

To build:
```shell
cmake --build build
```

To run tests:
```shell
cmake --build build --target test
```

To build documentation:
```shell
cmake --build build --target docs
```