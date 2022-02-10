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

#include <cassert>
#include <iostream>

#include <torch/torch.h>

#include <gen/dml.h>
#include <gen/parameters.h>
#include <gen/distributions/normal.h>
#include <gen/utils/randutils.h>
#include <gen/conversions.h>

#include <gentl/inference/mcmc.h>

using randutils::seed_seq_fe128;
using torch::Tensor, torch::tensor;
using std::vector;

using gen::dml::DMLGenFn;
using gen::EmptyModule;
using gen::distributions::normal::Normal;
using gen::Nothing, gen::nothing;
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
    auto proposal = [&model](const Model::trace_type&) { return model; };
    auto trace = model.simulate(rng, parameters, SimulateOptions());
    bool accepted = gentl::mcmc::mh(rng, trace, proposal, parameters, false);
    REQUIRE(accepted);
}
