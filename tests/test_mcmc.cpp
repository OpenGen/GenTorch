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

#include <catch2/catch.hpp>

#include <cassert>
#include <iostream>

#include <torch/torch.h>

#include <gentorch/dml/dml.h>
#include <gentorch/parameters.h>
#include <gentorch/distributions/normal.h>
#include <gentorch/conversions.h>

#include <gentl/util/randutils.h>
#include <gentl/inference/mcmc.h>

using gentl::randutils::seed_seq_fe128;
using torch::Tensor, torch::tensor;
using std::vector;

using gentorch::dml::DMLGenFn;
using gentorch::EmptyModule;
using gentorch::distributions::normal::Normal;
using gentorch::Nothing, gentorch::nothing;
using std::cout, std::endl;

struct Model;
struct Model : public DMLGenFn<Model, Nothing, Nothing, EmptyModule> {
    explicit Model() : DMLGenFn<M, A, R, P>(nothing) {};
    template<typename T>
    return_type forward(T &tracer) const {
        auto x = tracer.call({"x"}, Normal(tensor(0.0), tensor(1.0)));
        return nothing;
    }
};

TEST_CASE("mh", "[mh]") {

    EmptyModule parameters {};
    seed_seq_fe128 minibatch_seed{1};
    std::mt19937 rng(minibatch_seed);
    Model model{};
    auto proposal = [&model](const gentorch::dml::DMLTrace<Model>&) { return model; };
    auto trace = model.simulate(rng, parameters, SimulateOptions());
    bool accepted = gentl::mcmc::mh<gentorch::dml::DMLTrace<Model>>(rng, *trace, proposal, parameters);
    REQUIRE(accepted);
}
