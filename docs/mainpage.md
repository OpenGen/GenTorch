# Documentation for GenTorch                             {#mainpage}

GenTorch is a probabilistic programming language embedded in C++, that uses the PyTorch C++ API (LibTorch) for automatic differentiation.
GenTorch probabilistic programs can represent probabilistic **generative models** (including structured generative models and deep generative models) as well as **inference models** (including proposal distributions and amortized inference networks).

## Defining a probabilistic program
Users write a GenTorch probabilistic program by defining a new class that derives from the gentorch::DMLGenFn class.

### Declaring the input type, return type, and parameter type
As part of the declaration of your class, you indicate the **input type**, **return type**, and **parameter type** of your probabilistic program.
Your class should have a constructor that invokes the base class constructor, which takes a single argument called the **input**.

## Defining the forward function
Your class should have a template member function called **forward**, that takes a single template argument (the **tracer**; more on this later) and returns the **return type**.
The body of this member function (which should be marked as `const` because it does not modify the inputs) defines the probabilistic semantics of your program by making random choices.
You can use whatever deterministic code you want (with restrictions if you want support for gradients; see below), combined with special calls to other probabilistic programs made via **tracer.call**.
You can call other GenTorch probabilistic programs, primitive probabilistic programs provided in this library, or other objects that implement the same GenTL interface.

## Using your probabilistic program

Why write your model as a probabilistic program? Because GenTorch will automatically implement a number of functions that are useful for doing inference and learning with the model.
These functions together implement the interface expected by the [GenTL](https://github.com/OpenGen/GenTL/) library (see [interface](https://opengen.github.io/gentl-docs/latest/#interface)), which includes implementations of several composable inference and learning algorithms that you can apply to your model.
You can also write your own inference and learning algorithms using the functions provided by GenTorch.
