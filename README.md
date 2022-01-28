# libgen-experimental

This repository is based on the hypothesis that a port of [Gen](gen.dev) to C++, using LibTorch for numerical types and operations and automatic differentiation, will be useful for industry research use cases, will be a potential backend for a port of Gen to Python, will provide a balance of ergonomics and performance for learning and inference in structured generative models with stochastic structure, and and will be useful for real-time applications in robotics, computer vision, perception, and human-understanding.

## Dependencies

### C++ compiler supporting C++20

For example, either [clang++-13](https://clang.llvm.org/get_started.html) or [g++-11](https://gcc.gnu.org/).

### CMake 3.7 or above

See [Installing CMake](https://cmake.org/install/).

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

## Milestones and planning

(Also see `docs/design_notes.md`).

### Milestone 1: Complete DML implementation that supports multi-threaded learning and inference

[x] Implement `Address` type.

[x] Implement `Trie` type.

[x] Implement `simulate`, `generate`, `get_score`, `get_choice_trie`, and `get_return_value` for DML.

[~] Implement `update` for DML, without incremental computation.

[x] Add ability to register `torch::nn::Module`s that are called directly within a DML function, and to recursively compute a vector of trainable parameters, like supported by `torch::nn::Module`).

[x] Implement an example of multi-threaded simple Monte Carlo and importance sampling that exhibits near-linear scaling up to 32 cores on a c6i.metal instance.

[~] Implement an example of multi-threaded SGD-based training of a `torch::nn::Module` that exhibits good near-linear scaling up to 32 cores on a c6i.metal instance.

[x] Implement threadsafe `gradients` for DML with support for compound data
types in arguments and return values, but not support for parameter gradients

[x] Add support for parameter gradients to the DML gradients implementation

[ ] Add example of running `gradients` on a generative function that calls a `torch::nn::Module` that is defined in C++ with a custom gradient.

[ ] Add  example of running `gradients` on a generative function that calls a `torch::jit::script::Module` that was defined in Python.

[x] Add some abstraction to make it easy to add a new primitive distribution.

[x] Implement `Normal` and `Bernoulli` and `UniformContinuous` distributions based on that abstraction.

[ ] Implement continous integration on GitHub

### Milestone 2: Initial version of a multi-threaded inference and learning library

[ ] Implement a library function for multi-threaded SGD-based learning of a generative function, illustrated using an inference model with stochastic structure.

[ ] Implement a library function for multi-threaded importance sampling with a custom proposal, illustrated using a proposal trained using multi-threaded SGD.

[ ] Implement a Metropolis-Hastings library function.

### Milestone 3: Incremental computation and sequential Monte Carlo

[ ] Add an `Unfold` combinator and incremental computation for lists.

[ ] Implement a multi-threaded sequential Monte Carlo inference library function that supports custom proposals, rejuvenation moves, and has approximately linear scaling behavior.

### Milestone 4: Involutive MCMC

[ ] Implement library function for involutive MCMC.
