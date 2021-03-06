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
#include <gentorch/trie.h>
#include <gentorch/dml/dml.h>
#include <gentorch/distributions/normal.h>

#include <gentl/types.h>

#include <cassert>
#include <iostream>
#include <memory>

#include <torch/torch.h>

using torch::Tensor;
using torch::TensorOptions;
using torch::tensor;

using gentorch::dml::DMLGenFn;
using gentorch::distributions::normal::Normal;
using gentorch::distributions::normal::NormalDist;
using gentorch::ChoiceTrie;
using gentorch::EmptyModule;

using gentl::SimulateOptions;
using gentl::GenerateOptions;
using gentl::UpdateOptions;

namespace gentorch::test::dml {

bool tensor_scalar_bool(Tensor a) {
    if (a.ndimension() != 0) {
        throw std::logic_error("bool error");
    }
    return *(a.data_ptr<bool>());
}

class Foo : public DMLGenFn<Foo, std::pair<Tensor, int>, Tensor, EmptyModule> {
public:
    explicit Foo(Tensor z, int depth) : DMLGenFn<M, A, R, P>({z, depth}) {}

    // NOTE: the default copy constructor does not call the copy constructor of the base class..
    // this is annoying boilerplate
    Foo(const Foo &other) : DMLGenFn(other) {}

    template<typename Tracer>
    return_type forward(Tracer &tracer) const {
        auto &parameters = tracer.get_parameters();
        const auto &args = tracer.get_args();
        const auto&[z, depth] = tracer.get_args();
        auto x = tensor(0.0);
        auto y = tensor(1.0) + z + x;
        auto a = tensor(0.0);
        auto z1 = tracer.call({"z1"}, Normal(a, tensor(1.0)));
        auto z2 = tracer.call({"z2"}, Normal(z1, tensor(0.1)));
        Tensor result;
        if (tensor_scalar_bool(z1 < 0.0)) {
            auto z3 = tracer.call({"recursive"}, Foo(z2, depth + 1), parameters);
            result = x + z1 + y + z2 + z3;
        } else {
            result = x + z1 + y + z2;
        }
        return result;// + w;
    }
};

}

TEST_CASE("update", "[dml]") {
    using gentorch::test::dml::Foo;
    c10::InferenceMode guard{true};

    std::random_device rd{};
    std::mt19937 rng{rd()};
    EmptyModule parameters;
    Foo model {tensor(1.0), 0};

    // obtain initial trace with a recursive call
    ChoiceTrie constraints1 {};
    constraints1.set_value({"z1"}, tensor(-1.0));
    constraints1.set_value({"z2"}, tensor(2.0));
    constraints1.set_value({"recursive", "z1"}, tensor(1.0));
    constraints1.set_value({"recursive", "z2"}, tensor(3.0));
    auto [trace, generate_log_weight] = model.generate(rng, parameters, constraints1,
                                                       GenerateOptions().precompute_gradient(false));
    std::cout << "choices before update: " << std::endl;
    std::cout << trace->choices() << std::endl;

    // no change to the model, just a change to the choices that removes the recursive call
    ChoiceTrie constraints2 {};
    constraints2.set_value({"z1"}, tensor(1.0)); // this causes the recursive call to not be made
    double update_log_weight = trace->update(rng, gentl::change::UnknownChange<Foo>(model), constraints2,
                                            UpdateOptions().save(true));
    ChoiceTrie choices = trace->choices();
    std::cout << "choices after update: " << std::endl;
    std::cout << choices << std::endl;
    REQUIRE(choices.get_subtrie({"z1"}).has_value());
    REQUIRE(choices.get_subtrie({"z2"}).has_value());
    REQUIRE(!choices.has_subtrie({"recursive"}));

    const ChoiceTrie& backward_constraints = trace->backward_constraints();
    std::cout << "backward coinstraints: " << std::endl;
    std::cout <<backward_constraints << std::endl;
    REQUIRE(backward_constraints.get_subtrie({"z1"}).has_value());
    REQUIRE(!backward_constraints.has_subtrie({"z2"}));
    REQUIRE(backward_constraints.get_subtrie({"recursive", "z1"}).has_value());
    REQUIRE(backward_constraints.get_subtrie({"recursive", "z2"}).has_value());

    // revert it
    trace->revert();
    ChoiceTrie choices_after_reversion = trace->choices();
    REQUIRE(choices_after_reversion.get_subtrie({"z1"}).has_value());
    REQUIRE(choices_after_reversion.get_subtrie({"z2"}).has_value());
    REQUIRE(choices_after_reversion.get_subtrie({"recursive", "z1"}).has_value());
    REQUIRE(choices_after_reversion.get_subtrie({"recursive", "z2"}).has_value());

}

TEST_CASE("simulate", "[dml]") {
    using gentorch::test::dml::Foo;
    c10::InferenceMode guard{true};
    EmptyModule parameters;
    Foo model {tensor(1.0), 0};
    std::random_device rd{};
    std::mt19937 gen{rd()};
    auto trace = model.simulate(gen, parameters, SimulateOptions().precompute_gradient(true));
    ChoiceTrie choices = trace->choices();
    REQUIRE(choices.get_subtrie({"z1"}).has_value());
    REQUIRE(choices.get_subtrie({"z2"}).has_value());
}

TEST_CASE("generate", "[dml]") {
    using gentorch::test::dml::Foo;
    c10::InferenceMode guard{true};
    EmptyModule parameters;
    Foo model {tensor(1.0), 0};
    std::random_device rd{};
    std::mt19937 gen{rd()};
    ChoiceTrie constraints {};
    constraints.set_value({"z1"}, tensor(-1.0));
    constraints.set_value({"z2"}, tensor(2.0));
    constraints.set_value({"recursive", "z1"}, tensor(1.0));
    constraints.set_value({"recursive", "z2"}, tensor(3.0));
    auto [trace, log_weight] = model.generate(gen, parameters, constraints, GenerateOptions().precompute_gradient(true));
    ChoiceTrie choices = trace->choices();
    REQUIRE(any_cast<Tensor>(choices.get_value({"z1"})).equal(tensor(-1.0)));
    REQUIRE(any_cast<Tensor>(choices.get_value({"z2"})).equal(tensor(2.0)));
    REQUIRE(any_cast<Tensor>(choices.get_value({"recursive", "z1"})).equal(tensor(1.0)));
    REQUIRE(any_cast<Tensor>(choices.get_value({"recursive", "z2"})).equal(tensor(3.0)));
}

void do_simulate(int idx, int n, std::vector<double>& scores, EmptyModule& parameters) {
    using gentorch::test::dml::Foo;
    c10::InferenceMode guard{true};
    std::random_device rd{};
    std::mt19937 gen{rd()};
    double score = 0.0;
    for (int j = 0; j < n; j++) {
        Tensor z = tensor(1.0);
        auto model = Foo(z, 0);
        auto trace = model.simulate(gen, parameters, SimulateOptions());
        score += trace->score();
    }
    scores[idx] = score;
}

void do_generate(int idx, int n, std::vector<double>& scores, const ChoiceTrie& constraints, EmptyModule& parameters) {
    using gentorch::test::dml::Foo;
    c10::InferenceMode guard{true};
    std::random_device rd{};
    std::mt19937 gen{rd()};
    double total_log_weight = 0.0;
    for (int j = 0; j < n; j++) {
        Tensor z = tensor(1.0);
        assert(z.is_inference());
        auto model = Foo(z, 0);
        auto [trace, log_weight] = model.generate(gen, parameters, constraints, GenerateOptions());
        total_log_weight += log_weight;
    }
    scores[idx] = total_log_weight;
}

TEST_CASE("multithreaded_simulate", "[multithreading, dml]") {
    EmptyModule parameters;
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    int n = 1000;
    int n_threads = 1;
    std::vector<double> scores(n_threads, 0.0);
    std::vector<std::thread> threads;
    for (int i = 0; i < n_threads; i++) {
        threads.push_back(std::thread(do_simulate, i, n, std::ref(scores), std::ref(parameters)));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count()/1000000.0 << " seconds" << std::endl;
    std::cout << scores << std::endl;
}


TEST_CASE("multithreaded_generate", "[multithreading, dml]") {
    c10::InferenceMode guard{true};
    EmptyModule parameters;
    ChoiceTrie constraints {};
    constraints.set_value({"z2"}, tensor(1.0));
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    int n = 1000;
    int n_threads = 1;
    std::vector<double> scores(n_threads, 0.0);
    std::vector<std::thread> threads;
    for (int i = 0; i < n_threads; i++) {
        do_generate(i, n, scores, constraints, parameters);
        threads.push_back(
                std::thread(do_generate, i, n, std::ref(scores), std::ref(constraints), std::ref(parameters)));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count()/1000000.0 << " seconds" << std::endl;
    std::cout << scores << std::endl;
}

// *******************************************
// *** gradients with respect to arguments ***
// *******************************************

class GradientsTestGenFn;
class GradientsTestGenFn : public DMLGenFn<GradientsTestGenFn, std::pair<Tensor,Tensor>, Tensor, EmptyModule> {
public:
    explicit GradientsTestGenFn(Tensor x, Tensor y) : DMLGenFn<M,A,R,P>({x, y}) {}
    template <typename Tracer>
    return_type forward(Tracer& tracer) const {
        const auto& [x, y] = tracer.get_args();
        auto z = tracer.call({"z1"}, Normal(x + y, tensor(1.0)));
        return z + x + (y * 2.0);
    }
};

TEST_CASE("gradients with no parameters", "[gradients, dml]") {
    c10::InferenceMode guard{true};
    EmptyModule parameters;
    GradientAccumulator accum {parameters};
    std::random_device rd{};
    std::mt19937 gen{rd()};
    auto x = tensor(1.0);
    auto y = tensor(1.0);
    auto z1 = tensor(1.0);
    auto retval_grad = tensor(1.123);
    auto model = GradientsTestGenFn(x, y);
    ChoiceTrie constraints {};
    constraints.set_value({"z1"}, z1);
    auto [trace, log_weight] = model.generate(gen, parameters, constraints, GenerateOptions().precompute_gradient(true));
    auto arg_grads = any_cast<std::pair<Tensor,Tensor>>(trace->parameter_gradient(accum, retval_grad, 1.0));
    NormalDist dist {x + y, tensor(1.0)};
    auto logpdf_grad = dist.log_density_gradient(z1);
    Tensor expected_x_grad = 1.123 + std::get<1>(logpdf_grad);
    Tensor expected_y_grad = 1.123 * 2 + std::get<1>(logpdf_grad);
    REQUIRE(arg_grads.first.allclose(expected_x_grad));
    REQUIRE(arg_grads.second.allclose(expected_y_grad));
}

// *********************************************************
// *** parameter gradients and generative function calls ***
// *********************************************************

namespace gentorch::tests::dml {

struct ParametersTestCalleeModule : public gentorch::Parameters {
    Tensor theta1;
    ParametersTestCalleeModule() {
        c10::InferenceMode guard{false};
        theta1 = register_parameter("theta1", tensor(1.0));
    }
};

class ParametersTestCallee : public DMLGenFn<ParametersTestCallee, Tensor, Tensor, ParametersTestCalleeModule> {
public:
    explicit ParametersTestCallee(Tensor x) : DMLGenFn<M,A,R,P>{x} {}
    template <typename T>
    return_type forward(T& tracer) const {
        auto& parameters = tracer.get_parameters();
        const auto& x = tracer.get_args();
        auto mu = x + parameters.theta1;
        auto z = tracer.call({"z1"}, Normal(mu, tensor(2.0)));
        return z + mu;
    }
};

struct ParametersTestCallerModule : public gentorch::Parameters {
    Tensor theta2;
    shared_ptr<ParametersTestCalleeModule> callee_params {nullptr};
    ParametersTestCallerModule() {
        c10::InferenceMode guard{false};
        theta2 = register_parameter("theta2", tensor(3.0));
        callee_params = register_gen_module("callee_params", std::make_shared<ParametersTestCalleeModule>());
    }
};

class ParametersTestCaller : public DMLGenFn<ParametersTestCaller, Tensor, Tensor, ParametersTestCallerModule> {
public:
    explicit ParametersTestCaller(Tensor x) : DMLGenFn<M,A,R,P>{x} {}
    template <typename T>
    return_type forward(T& tracer) const {
        auto& parameters = tracer.get_parameters();
        const auto& x = tracer.get_args();
        auto z = tracer.call({"callee_addr"}, ParametersTestCallee(x), *parameters.callee_params);
        return z + x + parameters.theta2;
    }
};

}

TEST_CASE("parameter gradients and generative function calls", "[dml]") {
    c10::InferenceMode guard{true};

    gentorch::tests::dml::ParametersTestCallerModule parameters;
    auto x = tensor(4.0);
    auto z1 = tensor(5.0);
    auto retval_grad = tensor(6.0);
    double scaler = 7.0;

    // compute expected gradients with respect to theta1 and theta2
    NormalDist dist {x + parameters.callee_params->theta1, tensor(2.0)};
    auto log_density_grad = dist.log_density_gradient(z1);
    Tensor expected_theta1_grad = ((std::get<1>(log_density_grad) + retval_grad) * scaler).negative();
    Tensor expected_theta2_grad = (retval_grad * scaler).negative();

    // compute actual gradients with respect to theta1 and theta2
    GradientAccumulator accum {parameters};
    gentorch::tests::dml::ParametersTestCaller model {x};
    std::random_device rd{};
    std::mt19937 rng{rd()};
    ChoiceTrie constraints;
    constraints.set_value({"callee_addr", "z1"}, z1);
    auto [trace, log_weight] = model.generate(rng, parameters, constraints, GenerateOptions().precompute_gradient(true));
    auto arg_grads = trace->parameter_gradient(accum, retval_grad, scaler);
    accum.update_module_gradients();

    REQUIRE(parameters.callee_params->theta1.grad().allclose(expected_theta1_grad));
    REQUIRE(parameters.theta2.grad().allclose(expected_theta2_grad));
}


// *********************************************
// *** parameter gradients and torch modules ***
// *********************************************


namespace gentorch::tests::dml {

    struct ParametersTestTorchCallerModule : public gentorch::Parameters {
        torch::nn::Linear linear {nullptr};
        ParametersTestTorchCallerModule() {
            c10::InferenceMode guard{false};
            linear = register_torch_module("linear", torch::nn::Linear(3, 2));
        }
    };

    class ParametersTestTorchCaller : public DMLGenFn<ParametersTestTorchCaller, Tensor, Tensor, ParametersTestTorchCallerModule> {
    public:
        explicit ParametersTestTorchCaller(Tensor x) : DMLGenFn<M,A,R,P>{x} {}
        template <typename T>
        return_type forward(T& tracer) const {
            auto& parameters = tracer.get_parameters();
            const auto& x = tracer.get_args();
            auto y = parameters.linear->forward(x);
            return y;
        }
    };

}

TEST_CASE("parameter gradients and torch modules", "[dml]") {
    c10::InferenceMode guard{true};

    gentorch::tests::dml::ParametersTestTorchCallerModule parameters;
    auto x = tensor({1.0, 2.0, 3.0});
    auto retval_grad = tensor({4.0, 5.0});
    double scaler = 6.0;

    // compute expected gradients with respect to theta1 and theta2
    Tensor expected_bias_grad = (tensor({1.0, 1.0}) * retval_grad * scaler).negative();

    // compute actual gradients with respect to theta1 and theta2
    GradientAccumulator accum {parameters};
    gentorch::tests::dml::ParametersTestTorchCaller model {x};
    std::random_device rd{};
    std::mt19937 rng{rd()};
    auto trace = model.simulate(rng, parameters, SimulateOptions().precompute_gradient(true));
    auto arg_grads = trace->parameter_gradient(accum, retval_grad, scaler);
    accum.update_module_gradients();

    REQUIRE(parameters.linear->bias.grad().allclose(expected_bias_grad));
}

// *****************************************
// *** multithreaded parameter gradients ***
// *****************************************

void simulate_and_gradients(int n, gentorch::tests::dml::ParametersTestCallerModule& parameters, GradientAccumulator& accum) {
    std::random_device rd{};
    std::mt19937 rng{rd()};
    Tensor x = tensor(1.0);
    gentorch::tests::dml::ParametersTestCaller model {x};
    for (int i = 0; i < n; i++) {
        auto trace = model.simulate(rng, parameters, SimulateOptions().precompute_gradient(true));
        trace->parameter_gradient(accum, tensor(1.0), 1.0);
    }
}

TEST_CASE("multithreaded simulate and gradients", "[dml, parameters, multithreaded]") {
    c10::InferenceMode guard{true};

    gentorch::tests::dml::ParametersTestCallerModule parameters;
    torch::optim::SGD sgd {parameters.all_parameters(), torch::optim::SGDOptions(0.1)};

    size_t num_threads = 20;

    std::vector<GradientAccumulator> accums;
    for (int i = 0; i < num_threads; i++) {
        accums.emplace_back(GradientAccumulator{parameters});
    };

    // NOTE: in a real application we would reuse threads across iterations (e.g. thread pool)
    for (int iter = 0; iter < 10; iter++) {
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; i++) {
            threads.emplace_back(std::thread(simulate_and_gradients, 10, std::ref(parameters), std::ref(accums[i])));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        for (auto& accum : accums) {
            accum.update_module_gradients();
        }
        sgd.step();
    }

}
