/* Copyright 2021 The LibGen Authors

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

#include <catch2/catch.hpp>
#include <gen/conversions.h>

#include <cassert>

#include <torch/torch.h>

using std::vector, std::pair;
using torch::Tensor;
using torch::TensorOptions;
using torch::tensor;
using gen::unroll;
using gen::roll;

TEST_CASE("Tensor", "[conversion]") {

    // unroll
    Tensor value = tensor(1.0);
    vector<Tensor> unrolled = unroll(value);
    REQUIRE(unrolled.size() == 1);
    REQUIRE(unrolled[0].equal(tensor(1.0)));

    // roll
    Tensor value_rt = roll(unrolled, value);
    REQUIRE(value_rt.equal(tensor(1.0)));
}

TEST_CASE("pair of Tensors", "[conversion]") {

    // unroll
    pair<Tensor,Tensor> value {tensor(1.0), tensor(2.0)};
    vector<Tensor> unrolled = unroll(value);
    REQUIRE(unrolled.size() == 2);
    REQUIRE(unrolled[0].equal(tensor(1.0)));
    REQUIRE(unrolled[1].equal(tensor(2.0)));

    // roll
    pair<Tensor,Tensor> value_rt = roll(unrolled, value);
    REQUIRE(value_rt.first.equal(tensor(1.0)));
    REQUIRE(value_rt.second.equal(tensor(2.0)));
}

TEST_CASE("vector of Tensors", "[conversion]") {

    // unroll
    vector<Tensor> value {tensor(1.0), tensor(2.0)};
    vector<Tensor> unrolled = unroll(value);
    REQUIRE(unrolled.size() == 2);
    REQUIRE(unrolled[0].equal(tensor(1.0)));
    REQUIRE(unrolled[1].equal(tensor(2.0)));

    // roll
    vector<Tensor> value_rt = roll(unrolled, value);
    REQUIRE(value_rt.size() == 2);
    REQUIRE(value_rt[0].equal(tensor(1.0)));
    REQUIRE(value_rt[1].equal(tensor(2.0)));
}

TEST_CASE("nested compound data type", "[conversion]") {

    // unroll
    vector<vector<Tensor>> value {{tensor(1.0), tensor(2.0)}, {tensor(3.0)}};
    vector<Tensor> unrolled = unroll(value);
    REQUIRE(unrolled.size() == 3);
    REQUIRE(unrolled[0].equal(tensor(1.0)));
    REQUIRE(unrolled[1].equal(tensor(2.0)));
    REQUIRE(unrolled[2].equal(tensor(3.0)));

    // roll
    vector<vector<Tensor>> value_rt = roll(unrolled, value);
    REQUIRE(value_rt.size() == 2);
    REQUIRE(value_rt[0].size() == 2);
    REQUIRE(value_rt[1].size() == 1);
    REQUIRE(value_rt[0][0].equal(tensor(1.0)));
    REQUIRE(value_rt[0][1].equal(tensor(2.0)));
    REQUIRE(value_rt[1][0].equal(tensor(3.0)));
}