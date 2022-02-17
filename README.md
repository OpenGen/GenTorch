# GenTorch

[![Test](https://github.com/OpenGen/GenTorch/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/OpenGen/GenTorch/actions/workflows/test.yml)

**Work-in-progress** probabilistic programming language for use with [GenTL](https://github.com/OpenGen/GenTL), building on [LibTorch](https://pytorch.org/cppdocs/installing.html).

[**Documentation**](https://opengen.github.io/gentorch-docs/latest/)

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

To build, either open the `GenTorch` directory in a C++ IDE (e.g. [https://www.jetbrains.com/clion/](CLion)) or use `cmake` directly.
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

## Roadmap

[ ] Implement involutive MCMC
[ ] Implement `choice_gradient`
[ ] Implement arithmetic for `ChoiceTrie`s whose values are `Tensor`s (use `std::any_cast<Tensor>`)
