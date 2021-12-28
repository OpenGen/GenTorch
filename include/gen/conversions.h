#pragma once

#include <vector>
#include <torch/torch.h>

using torch::Tensor;
using std::vector;
using std::pair;

// Data types that can be used as arguments and return values of generative functions

// ***********************
// ** unsupported types **
// ***********************

// TODO: consider emitting a warning

template <typename T>
void unroll(vector<Tensor>& unrolled, const T& value) {
}

template <typename T>
pair<size_t,T> roll(const vector<Tensor>& unrolled, size_t start, const T& value) {
    return {start, value};
}

// ************
// ** Tensor **
// ************

pair<size_t,Tensor> roll(const vector<Tensor>& unrolled, size_t start, const Tensor& value);
void unroll(vector<Tensor>& unrolled, const Tensor& value);

// ***************
// ** pair<T,U> **
// ***************

template <typename T, typename U>
void unroll(vector<Tensor>& unrolled, const pair<T,U>& value) {
    unroll(unrolled, value.first);
    unroll(unrolled, value.second);
}

template <typename T, typename U>
pair<size_t,pair<T,U>> roll(const vector<Tensor>& unrolled, size_t start, const pair<T,U>& value) {
    auto [after_first, rolled_first] = roll(unrolled, start, value.first);
    auto [after_second, rolled_second] = roll(unrolled, after_first, value.second);
    return {after_second, {rolled_first, rolled_second}};
}

// ***************
// ** vector<T> **
// ***************

template <typename T>
void unroll(vector<Tensor>& unrolled, const vector<T>& value) {
    for (const auto element: value) {
        unroll(unrolled, element);
    }
}

template <typename T>
pair<size_t,vector<T>> roll(const vector<Tensor>& unrolled, size_t start, const vector<T>& value) {
    vector<T> rolled;
    for (const auto element: value) {
        auto [new_start, rolled_element] = roll(unrolled, start, element);
        start = new_start;
        rolled.emplace_back(rolled_element);
    }
    return {start, rolled};
}


// *************
// ** general **
// *************

template <typename T>
vector<Tensor> unroll(const T& value) {
    std::vector<Tensor> unrolled;
    unroll(unrolled, value);
    return unrolled;
}

template <typename T>
T roll(const vector<Tensor>& unrolled, const T& value) {
    auto [num_tensors, rolled] = roll(unrolled, 0, value);
    return rolled;
}

