# libgen-experimental-sandbox

NOTE: The code in this repository is not currently intended for external users. It is a personal sandbox for experimenting with implementations of Gen in C++ (currently based on LibTorch, the PyTorch C++ library).

This repository is based on the hypothesis that a port of [Gen](gen.dev) to C++, using LibTorch for numerical types and operations and automatic differentiation, will be useful for industry research use cases, will be a potential backend for a port of Gen to Python, will provide a balance of ergonomics and performance for learning and inference in structured generative models with stochastic structure, and and will be useful for real-time applications in robotics, computer vision, perception, and human-understanding.

NOTE: Uses C++17.

## Installing LibTorch dependency

This library depends on the C++ distribution of PyTorch, called LibTorch.

So far, the code has only been tested with the CPU-version of LibTorch (version 1.9.1+cpu).

Follow instructions at [Installing C++ distributions of PyTorch](https://pytorch.org/cppdocs/installing.html) to install LibTorch, and record the path.

## Building with cmake


Currently this project just builds a few executables that run some test code.
It does not build a library.

To build, either open the `libgen-experiment-sandbox` directory in a C++ IDE (e.g. [https://www.jetbrains.com/clion/](CLion)) or use `cmake` directly.
In either case, you need to add an option `-DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch` when running cmake.

```shell
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build .
```

This should generate some executables in the build directly.
See `CMakeLists.txt` for the list of executables.

## Milestones and planning

(Also see `docs/design_notes.md`).

### Milestone 1: Complete DML implementation that supports multi-threaded learning and inference

[x] Implement `Address` type.

[x] Implement `Trie` type.

[x] Implement `simulate`, `generate`, `get_score`, `get_choice_trie`, and `get_return_value` for DML.

[~] Implement `update` for DML, without incremental computation.

[ ] Add ability to register `torch::nn::Module`s that are called directly within a DML function, and to recursively compute a vector of trainable parameters, like supported by `torch::nn::Module`).

[~] Implement an example of multi-threaded SGD-based training of a `torch::nn::Module` that exhibits good near-linear scaling up to 32 cores on a c6i.metal instance.

[ ] Implement `gradients` for DML, with support for separate gradient accumulators per-thread and aggregation across threads, but only for generative functions whose return value is a `torch::Tensor` and whose arguments are a tuple of `torch::Tensor`s.

[ ] Add example of running `gradients` on a generative function that calls a `torch::nn::Module` that is defined in C++ with a custom gradient.

[ ] Add  example of running `gradients` on a generative function that calls a `torch::jit::script::Module` that was defined in Python.

[ ] Add some abstraction to make it easy to add a new primitive distribution.

[ ] Implement `Normal` and `Bernoulli` and `Discrete` distributions based on that abstraction.

[ ] Add support for using `gradients` with generative function that have return values and arguments that are not only `torch::Tensors`, but include tuples, lists, and dictionaries, and compositions of these. (This will likely involve defining Gen-specific compound data types). Also add support for registering new user-defined data types that also inter-operate cleanly with `gradients`.

[ ] Implement proper tests and continous integration.

### Milestone 2: Initial version of a multi-threaded inference and learning library

[ ] Implement a library function for multi-threaded SGD-based learning of a generative function, illustrated using an inference model with stochastic structure.

[ ] Implement a library function for multi-threaded importance sampling with a custom proposal, illustrated using a proposal trained using multi-threaded SGD.

[ ] Implement a Metropolis-Hastings library function.

### Milestone 3: Incremental computation and sequential Monte Carlo

[ ] Add an `Unfold` combinator and incremental computation for lists.

[ ] Implement a multi-threaded sequential Monte Carlo inference library function that supports custom proposals, rejuvenation moves, and has approximately linear scaling behavior.

### Milestone 4: Involutive MCMC

[ ] Implement library function for involutive MCMC.
