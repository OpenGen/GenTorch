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
#include <gentl/inference/particle_filter.h>

using gentl::randutils::seed_seq_fe128;
using torch::Tensor, torch::tensor;
using std::vector;

using gentorch::dml::DMLGenFn;
using gentorch::EmptyModule;
using gentorch::distributions::normal::Normal;
using gentorch::Nothing, gentorch::nothing;
using std::cout, std::endl;

namespace gentorch::test_particle_filter {

struct Model;

struct Model : public DMLGenFn<Model, std::pair<size_t, Tensor>, Nothing, EmptyModule> {
    explicit Model(size_t num_steps, Tensor init_x) : DMLGenFn<M, A, R, P>({num_steps, init_x}) {};

    template<typename T>
    return_type forward(T& tracer) const {
        const auto&[num_steps, init_x] = tracer.get_args();
        auto x = init_x.clone();
        for (int i = 0; i < num_steps; i++) {
            x = tracer.call({std::to_string(i), "x"}, Normal(x, tensor(1.0)));
            auto y = tracer.call({std::to_string(i), "y"}, Normal(x, tensor(0.1)));
        }
        return nothing;
    }
};

}

TEST_CASE("particle filter") {
    using gentorch::test_particle_filter::Model;
    EmptyModule parameters {};
    seed_seq_fe128 minibatch_seed{1};
    std::mt19937 rng(minibatch_seed);
    Model model{0, tensor(0.0)};
    gentl::smc::ParticleSystem<Model::trace_type,std::mt19937> filter{1, rng};
    filter.init_step(model, parameters, ChoiceTrie{});
    {
        ChoiceTrie constraints;
        constraints.set_value({std::to_string(0), "y"}, tensor(1.0));
        filter.step(gentl::change::UnknownChange<Model>(Model{1, tensor(0.0)}), constraints);
        filter.resample();
    }
    {
        ChoiceTrie constraints;
        constraints.set_value({std::to_string(1), "y"}, tensor(2.0));
        filter.step(gentl::change::UnknownChange<Model>(Model{2, tensor(0.0)}), constraints);
        filter.resample();
    }
}
