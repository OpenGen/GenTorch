/* Copyright 2021-2022 Massachusetts Institute of Technology

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/


#ifndef GENTORCH_CONVERSIONS_H
#define GENTORCH_CONVERSIONS_H

#include <vector>
#include <torch/torch.h>

using torch::Tensor;
using std::vector;
using std::pair;

namespace gentorch {

    std::string __attribute__ ((noinline)) print(const Tensor& tensor);

// Data types that can be used as arguments and return values of generative functions

// **************************************
// ** default behavior for other types **
// **************************************

// TODO: consider emitting a warning

    template<typename T>
    void unroll(vector<Tensor> &unrolled, const T &value) {
    }

    template<typename T>
    pair<size_t, T> roll(const vector<Tensor> &unrolled, size_t start, const T &value) {
        return {start, value};
    }

// *************
// ** Nothing **
// *************

    typedef std::nullptr_t Nothing;
    constexpr Nothing nothing = nullptr;

    Nothing zero_gradient(const Nothing& value);

// ************
// ** Tensor **
// ************

    pair<size_t, Tensor> roll(const vector<Tensor> &unrolled, size_t start, const Tensor &value);

    void unroll(vector<Tensor> &unrolled, const Tensor &value);

    Tensor zero_gradient(const Tensor& value); // TODO implement

// ***************
// ** pair<T,U> **
// ***************

    template<typename T, typename U>
    void unroll(vector<Tensor> &unrolled, const pair<T, U> &value) {
        unroll(unrolled, value.first);
        unroll(unrolled, value.second);
    }

    template<typename T, typename U>
    pair<size_t, pair<T, U>> roll(const vector<Tensor> &unrolled, size_t start, const pair<T, U> &value) {
        auto[after_first, rolled_first] = roll(unrolled, start, value.first);
        auto[after_second, rolled_second] = roll(unrolled, after_first, value.second);
        return {after_second, {rolled_first, rolled_second}};
    }

// ***************
// ** vector<T> **
// ***************

    template<typename T>
    void unroll(vector<Tensor> &unrolled, const vector<T> &value) {
        for (const auto element: value) {
            unroll(unrolled, element);
        }
    }

    template<typename T>
    pair<size_t, vector<T>> roll(const vector<Tensor> &unrolled, size_t start, const vector<T> &value) {
        vector<T> rolled;
        for (const auto element: value) {
            auto[new_start, rolled_element] = roll(unrolled, start, element);
            start = new_start;
            rolled.emplace_back(rolled_element);
        }
        return {start, rolled};
    }


// *************
// ** general **
// *************

    template<typename T>
    vector<Tensor> unroll(const T &value) {
        std::vector<Tensor> unrolled;
        unroll(unrolled, value);
        return unrolled;
    }

    template<typename T>
    T roll(const vector<Tensor> &unrolled, const T &value) {
        auto[num_tensors, rolled] = roll(unrolled, 0, value);
        return rolled;
    }

}

#endif // GENTORCH_CONVERSIONS_H