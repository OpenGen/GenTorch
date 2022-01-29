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
#include <gen/sgd.h>
#include <gen/utils/randutils.h>
#include <gen/conversions.h> // TODO we may need to rename this to 'types'


using randutils::seed_seq_fe128;
using torch::Tensor, torch::tensor;
using std::vector;
using gen::dml::DMLGenFn;
using gen::EmptyModule;
using gen::distributions::normal::Normal;
using gen::Nothing, gen::nothing;

struct ModelModule : public gen::Parameters {
    Tensor a;
    Tensor b;
    ModelModule() {
        c10::InferenceMode guard{false};
        a = register_parameter("a", tensor(0.0));
        b = register_parameter("b", tensor(0.0));
    };
};

struct Model;

struct Model : public DMLGenFn<Model, Nothing, Nothing, ModelModule> {
    explicit Model() : DMLGenFn<M, A, R, P>(nothing) {};

    template<typename T>
    return_type forward(T &tracer) const {
        auto &parameters = tracer.get_parameters();
        assert(parameters.a.allclose(tensor(0.0)));
        assert(parameters.b.allclose(tensor(0.0)));
        auto mean = parameters.a - parameters.b;
        auto x = tracer.call({"x"}, Normal(mean, tensor(1.0)));
        assert(x.allclose(tensor(1.0)) || x.allclose(tensor(2.0)) || x.allclose(tensor(3.0)));
        return nothing;
    }
};

typedef Tensor datum_t;


TEST_CASE("multi-threaded matches single-threaded", "[sgd]") {


    auto unpack_datum = [](const datum_t &datum) -> std::pair<Model, gen::ChoiceTrie> {
        const auto& x = datum;
        Model model{};
        gen::ChoiceTrie constraints;
        constraints.set_value({"x"}, x);
        return {model, constraints};
    };

    // Note: the parameters of both are initialized deterministically and identically
    ModelModule parameters1 {};
    ModelModule parameters2 {};

    size_t minibatch_size = 10;
    seed_seq_fe128 minibatch_seed{1};

    std::vector<Tensor> data {tensor(1.0), tensor(2.0), tensor(3.0)};

    auto expected_a_gradient = [&data](const std::vector<size_t>& minibatch) {
        // gradient of log standard normal density with respect to mean is:
        // lpdf = C + -0.5 (x - mu)^2
        // deriv wtr mu is: -1 * (x - mu) * (-1) = x - mu
        // total gradient at mu = 0.0 is:
        double expected = 0.0;
        for (auto idx : minibatch) {
            expected += (*data[idx].data_ptr<float>() / static_cast<double>(minibatch.size()));
        }
        return -expected;
    };

    bool callback_was_called;
    std::vector<size_t> minibatch_single;
    std::vector<size_t> minibatch_multi;

    auto callback_single = [&callback_was_called,&minibatch_single](const std::vector<size_t>& minibatch) -> bool {
        callback_was_called = true;
        minibatch_single = minibatch; // copy assignment
        return true; // done (just evaluate the gradient once then return; do not reset gradients)
    };

    auto callback_multi = [&callback_was_called,&minibatch_multi](const std::vector<size_t>& minibatch) -> bool {
        callback_was_called = true;
        minibatch_multi = minibatch; // copy assignment
        return true; // done (just evaluate the gradient once then return; do not reset gradients)
    };

    std::mt19937 rng(minibatch_seed);
    callback_was_called = false;
    gen::sgd::train_supervised_single_threaded(parameters1, callback_single, data, unpack_datum, minibatch_size, rng);
    REQUIRE(callback_was_called);

    size_t num_threads = 3;
    callback_was_called = false;
    gen::sgd::train_supervised(parameters2, callback_multi, data, unpack_datum, minibatch_size, num_threads, minibatch_seed);
    REQUIRE(callback_was_called);

    // check that gradients are nonzero
    REQUIRE(*parameters1.a.grad().abs().data_ptr<float>() > 0.001);
    REQUIRE(!parameters1.a.grad().allclose(parameters1.b.grad()));

    // check that the minibatches are equal (they should use the same the seed)
    REQUIRE(minibatch_single.size() == minibatch_multi.size());
    for (size_t i = 0; i < minibatch_size; i++) {
        REQUIRE(minibatch_single[i]== minibatch_multi[i]);
    }

    // check that they match the expectation
    REQUIRE(parameters1.a.grad().allclose(tensor(expected_a_gradient(minibatch_single))));
    REQUIRE(parameters1.b.grad().allclose(-tensor(expected_a_gradient(minibatch_single))));
    REQUIRE(parameters2.a.grad().allclose(tensor(expected_a_gradient(minibatch_multi))));
    REQUIRE(parameters2.b.grad().allclose(-tensor(expected_a_gradient(minibatch_multi))));

    // check that they match
    REQUIRE(parameters1.a.grad().allclose(parameters2.a.grad()));
    REQUIRE(parameters1.b.grad().allclose(parameters2.b.grad()));


}
