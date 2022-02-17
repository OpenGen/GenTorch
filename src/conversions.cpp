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

#include <gentorch/conversions.h>

#include <torch/torch.h>

#include <vector>
#include <string>
#include <sstream>

using torch::Tensor;
using std::vector;
using std::pair;

namespace gentorch {

    std::string __attribute__ ((noinline)) print(const Tensor& tensor) {
        std::stringstream ss;
        ss << tensor;
        std::cout << tensor << std::flush;
        return ss.str();
    }

// *************
// ** Nothing **
// *************

    Nothing zero_gradient(const Nothing& value) {
        return nothing;
    }

// *************************************
// ** Tensor (scalar numerical value) **
// *************************************

pair<size_t, Tensor> roll(const vector<Tensor> &unrolled, size_t start, const Tensor &value) {
    return {start + 1, unrolled[start]};
}

void unroll(vector<Tensor> &unrolled, const Tensor &value) {
    unrolled.emplace_back(value);
}

}
