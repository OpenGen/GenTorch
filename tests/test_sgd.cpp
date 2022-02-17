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
#include <gentl/learning/supervised.h>

using gentl::randutils::seed_seq_fe128;
using torch::Tensor, torch::tensor;
using std::vector;

using gentorch::dml::DMLGenFn;
using gentorch::EmptyModule;
using gentorch::distributions::normal::Normal;
using gentorch::Nothing, gentorch::nothing;
using std::cout, std::endl;

struct ModelModule : public gentorch::Parameters {
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


    auto unpack_datum = [](const datum_t &datum) -> std::pair<Model, gentorch::ChoiceTrie> {
        const auto& x = datum;
        Model model{};
        gentorch::ChoiceTrie constraints;
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
    gentl::sgd::train_supervised_single_threaded(parameters1, callback_single, data, unpack_datum, minibatch_size, rng);
    REQUIRE(callback_was_called);

    size_t num_threads = 3;
    callback_was_called = false;
    gentl::sgd::train_supervised(parameters2, callback_multi, data, unpack_datum, minibatch_size, num_threads, minibatch_seed);
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

struct SimpleOptimizationModelModule : public gentorch::Parameters {
    Tensor mean;
    SimpleOptimizationModelModule() {
        c10::InferenceMode guard{false};
        mean = register_parameter("mean", tensor(0.0));
    }
};

struct SimpleOptimizationModel;

struct SimpleOptimizationModel : public DMLGenFn<SimpleOptimizationModel, Nothing, Nothing, SimpleOptimizationModelModule> {
    explicit SimpleOptimizationModel() : DMLGenFn<M, A, R, P>(nothing) {};
    template<typename T>
    return_type forward(T &tracer) const {
        if (tracer.prepare_for_gradients()) {
            assert(!c10::InferenceMode::is_enabled());
        }
        auto &parameters = tracer.get_parameters();
        auto mean = parameters.mean;
        auto x = tracer.call({"x"}, Normal(mean, tensor(1.0)));
        assert(x.sizes().equals({}));
        return nothing;
    }
};

TEST_CASE("simple optimization problem", "[sgd]") {


    auto unpack_datum = [](const Tensor& x) -> std::pair<SimpleOptimizationModel, gentorch::ChoiceTrie> {
        SimpleOptimizationModel model;
        gentorch::ChoiceTrie constraints;
        constraints.set_value({"x"}, x);
        return {model, constraints};
    };

    // data has 50 1s and 50 3s; the optimum mean should be 2.0
    double expected_objective_optimum = 0.0;
    double opt_mean = 2.0;
    std::vector<Tensor> data;
    for (size_t i = 0; i < 50; i++) {
        double x = 1.0;
        data.emplace_back(tensor(x));
        expected_objective_optimum += gentorch::distributions::normal::log_density(opt_mean, 1.0, x);
    }
    for (size_t i = 0; i < 50; i++) {
        double x = 3.0;
        data.emplace_back(tensor(x));
        expected_objective_optimum += gentorch::distributions::normal::log_density(opt_mean, 1.0, x);
    }
    expected_objective_optimum /= static_cast<double>(data.size());

    double learning_rate = 0.01;

    seed_seq_fe128 seed_seq {1};
    std::mt19937 rng {seed_seq};
    size_t num_iters = 400;
    size_t minibatch_size = 64;

    auto evaluate = [&data,&unpack_datum,&rng](size_t iter, SimpleOptimizationModelModule& parameters) -> double {
        double objective = gentl::sgd::estimate_objective(rng, parameters, data, unpack_datum);
        return objective;
    };

    // single threaded
    {
        SimpleOptimizationModelModule parameters {};
        torch::optim::SGD sgd {
                parameters.all_parameters(),
                torch::optim::SGDOptions(learning_rate).dampening(0.0).momentum(0.0)};
        size_t iter = 0;
        double objective;
        auto callback = [&iter,&evaluate,&parameters,&objective,num_iters,&sgd](const std::vector<size_t>& minibatch) -> bool {
            c10::InferenceMode guard {true};
            sgd.step();
            sgd.zero_grad();
            if (iter % 100 == 0)
                objective = evaluate(iter, parameters);
            return (iter++) == num_iters - 1;
        };
        evaluate(iter++, parameters);
        gentl::sgd::train_supervised_single_threaded(parameters, callback, data, unpack_datum, minibatch_size, rng);
        REQUIRE(parameters.mean.allclose(tensor(opt_mean), 0.05, 0.05));
        REQUIRE(std::abs(objective - expected_objective_optimum) < 0.01);
    }

    // multi threaded
    {
        size_t num_threads = 2;
        SimpleOptimizationModelModule parameters {};
        torch::optim::SGD sgd {
                parameters.all_parameters(),
                torch::optim::SGDOptions(learning_rate).dampening(0.0).momentum(0.0)};
        size_t iter = 0;
        double objective;
        auto callback = [&iter,&evaluate,&parameters,&objective,num_iters,&sgd](const std::vector<size_t>& minibatch) -> bool {
            c10::InferenceMode guard {true};
            sgd.step();
            sgd.zero_grad();
            if (iter % 100 == 0)
                objective = evaluate(iter, parameters);
            return (iter++) == num_iters - 1;
        };
        evaluate(iter++, parameters);
        gentl::sgd::train_supervised(parameters, callback, data, unpack_datum, minibatch_size, num_threads, seed_seq);
        REQUIRE(parameters.mean.allclose(tensor(opt_mean), 0.05, 0.05));
        REQUIRE(std::abs(objective - expected_objective_optimum) < 0.01);
    }


}
