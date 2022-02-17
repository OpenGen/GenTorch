# Design notes

**NOTE: These are out of date**

## Address class

The `Address` class is an immutable type that stores a sequence of keys (which are currently each `std::string`),
representing a hierarchical address of a call site in a probabilistic program.
An *empty* address is indicated by the `empty()` member function, and has no keys.
The first key of a non-empty address is accessible via `first()` and the rest of the address is accessible via `rest()`.

Question: What types should the keys of `Address` be? Of course, they must be immutable and have a hash function and quality. Should they be allowed to be integer types? There may be tradeoffs between performance and expressiveness here.

## Trie class

The `Trie` class implements a mutable trie data structure where each node in the trie can accessed via its `Address`.
This class is primarily intended to implement choice dictionaries with hierarchical addresses, such as those used to define observed data and constraints on generative functions.
A `Trie` can be in one of three states:
(i) empty (`empty` member function),
(ii) contain a value (see `has_value` and `get_value` member functions),
(iii) it has one or more subtries (see `has_subtrie`, `get_subtrie`, and `subtries` member functions).
A `Trie` can also be mutated via `set_value` and `set_subtrie` member functions.
Because types of random choices are heterogeneous, values are stored as `std::any`.
Note that `set_value` will invoke the copy constructor for the provided value.
Note that for `Tensor` values, the copy constructor does not copy the data, but only copies a shared pointer, so mutating a `Tensor` value that has been placed into a `Trie` will mutate the `Trie`.
For most basic C++ types (e.g. `double`, `std::string`), the copy constructor does copy the data.

## Trace abstract class

The `Trace` abstract class contains a set of virtual member functions that constitute the immutable trace abstract data type.

Note that the `get_return_value` virtual member function returns a value of type `std::any`, and that depending on the underlying data type, mutating this value may mutate the value stored in the trace (this is the case for `Tensor` values).
Therefore, users should not mutate values obtained with `get_return_value`.
The same is true for `Tensor` values obtained from `choices`.

## DML trace implementation

DML traces include a key-value store that maps addresses to subtraces.
The types of the subtraces are in general all different and not known until run time.
Therefore, the DML trace implementation uses a container (`std::unordered_map`) that maps keys (for now, `std::string`) to values.
For the value type, we could use a pointer to `void`, or a pointer to `Trace` where `Trace` is an abstract class that all trace implementations must inherit from.
One can also use `std::any`.
We use `std::any` so that we can use the same `Trie` data structure that is used for random choices (see above).
The DML code body knows the concrete type of subtraces, so that when executing the DML code body in during trace operations (such as `update`) we obtain a reference to the `Trace` abstract type via std::any_cast`.

## Use of `torch::Tensor` as an interface data type

The current implementation uses `torch::Tensor` as the standard numerical array data type.
Note that `torch::autograd` functionality is a main feature of `torch` that motivates this, but is only part of the DML language implementation, and so that alone does not necessitate its use as the implementation to use for numerical array data types.
Other considerations are that `torch::Tensor` has a large library of operations and a large user community.
More investigation needed.

## Multithreaded gradient-based learning

A *parameter store* contains its own copy of the values of parameters and its own gradient accumulator and support for gradient-based optimization.
When doing inference with traces, we do not mutate the parameter store, which is a private member variable of the trace that is shared across multiple traces that may be used within multiple threads.
We implement learning with traces using the `gradients` member function of a trace, which
accepts a return value gradient and a scaler for the gradient contribution, increments the gradient accumulators,
and returns gradients with respect to the generative function's arguments.

## Multithreaded SGD with traces

Because `gradients` mutates a parameter store, more care is required for supporting multithreading.
Specifically, SGD-based learning algorithms should create parameter gradient accumulators that are private to each thread that calls `gradients`, and should accumulate the per-thread gradient accumulators into the main gradient accumulator after each trace's gradients have been computed. A typical configuration might be 16 gradient accumulation threads that are tasked with accumluating gradients (using the `gradients` trace member function) for a collection of 64 traces (constituting a minibatch).
Each of the 16 threads will have its own gradient accumulator.
Other designs are also possible, but based on some initial experiments using multi-threaded training with `torch` in C++, this design seems suitable for the type of small-ish neural networks that we would conceivably choose to run on a CPU anyways. (For large neural networks a separate component that automatically batches per-trace computations might be worth the implementation effort and run time overhead.
Such a component could complement the multi-threaded SGD implementation strategy discussed here.).

Here is example code for doing parallel training in `torch`, where `net` is a `torch::nn::Module` that takes as input two `Tensor`s (`x` and `y`) and returns a `Tensor` (`loss`).

    for (int i = start_idx; i < end_idx; ++i) {
        const auto&[x, y] = training_data[minibatch[i]];
        Tensor loss = net.forward(x, y);
        int param_idx = 0;
        for (const auto& grad : torch::autograd::grad({loss}, net_parameters)) {
            parameter_grads_to_accumulate[param_idx++].add_(grad);
        }
    }

Here, `net_parameters` is a const reference to a vector of parameters that is shared by all threads and is obtained via:

    std::vector<Tensor> net_parameters { net.parameters(true) };

and `parameter_grads_to_accumulate` is a per-thread vector of gradient accumulators that can be obtained via:

    std::vector<Tensor> parameter_grads_to_accumulate;
    for (const auto& param : net_parameters) {
        parameter_grads_to_accumulate.emplace_back(torch::zeros_like(param.grad()).detach());
    }

and `start_idx` and `end_idx` are the beginning and end of the chunks of the minibatch that are assigned to this thread.

While deterministic chunking of traces across threads may be suitable when the traces are all the same size, the intended use case of stochastic control flow means that traces may be of different sizes.
Therefore a thread pool, which dynamically allocates threads to traces, may be better.

### DML and `torch:nn::Module` and `torch::autograd`

DML will use `torch::autograd` as the workhorse for automatic differentiation with its `gradients` implementation.
The implementation will a `torch::nn::Module`s to wrap each generative call site and glue the callee's implementation
of `gradients` into the `torch::autograd` computation graph.
While `torch::nn::Module`s could be wrapped as generative functions, to reduce overhead and boilerplate code required by users, allowing users to inline calls to `torch::nn::Module`s within their DML code seems likely a good idea.

A DML generative function will be associated with a `torch::nn::Module` (or similar) that gives a list of all the parameters (`torch::Tensor`s) used by all `torch::nn:Module`s called directly by this DML function (not through the generative function interface). The user will need to register these modules.

### Simulate, generate, and update member functions for DML

If a user knows that they are going to call `generate` on a trace produced from `simulate`, `generate` or `update`,
then it would be good to include a flag to `simulate`, `generate` or `update` that will cause a `torch` computation graph
to be constructed, so that an additional forward pass through the function in `gradients` is not necessary.

These functions should accept the parameter store that they should read the parameters of `torch::nn::Module`s from, and optionally

### Gradient member function for DML

DML uses LibTorch's autograd implementation.
To do this, we need to insert a special node into the computation graph for each generative function call that the DML function makes.
We cannot simply extend `torch::autograd::Function` like you would normally do when defining a new node type since the number of input tensors is part of the static type signature of the `forward` member function.
We require the number of input tensors to vary at run time, because the arguments to generative functions can be compound data structures that contain a dynamic number of tensors.
Therefore, we add nodes to the LibTorch computation graph using the lower-level `torch::autograd::Node` class.
Specifically, we create a class for a `Node` that represents the forward computation, a class for a `Node` that represents the gradient (backward) computation, and the forward computation establishes the edges in the computation graph that are later used in the backward pass.

Some relevant references:

- http://blog.ezyang.com/2019/05/pytorch-internals/
- https://github.com/pytorch/pytorch/blob/master/torch/csrc/autograd/function.h#L50
- https://discuss.pytorch.org/t/extending-autograd-from-c/76240/5

The `Node` interface uses a dynamic vector of `Tensor`s as the input to the node and the output of the node.
Overloaded `roll` and `unroll` functions are used to translate between these vectors and the broader set of other compound data types that are used in Gen programs.

## TorchScript modules

TorchScript provides a potentially important vehicle making user models (at least the PyTorch parts) defined in a Python-based prototyping environment available for use in Gen generative functions.

Modules defined using TorchScript (`torch::jit::script::Module`) should be supported in the same way that `torch::nn::Module`s are supported.

## Supporting AD through composite data types and custom data types

Rolling into `std::vector<torch::Tensor>` and unrolling from `std::vector<torch::Tensor>`.

## Parallel Sequential Monte Carlo and incremental computation

TODO
