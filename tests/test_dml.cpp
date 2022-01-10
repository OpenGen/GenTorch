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
#include <gen/trie.h>
#include <gen/dml.h>
#include <gen/normal.h>

#include <cassert>
#include <iostream>
#include <memory>

#include <torch/torch.h>

using torch::Tensor;
using torch::TensorOptions;
using torch::tensor;
using gen::dml::DMLGenFn;
using gen::distributions::normal::Normal;
using gen::distributions::normal::NormalDist;
using gen::ChoiceTrie;
using gen::EmptyModule;

bool tensor_scalar_bool(Tensor a) {
    if (a.ndimension() != 0) {
        throw std::logic_error("bool error");
    }
    return *(a.data_ptr<bool>());
}

class Foo : public DMLGenFn<Foo, std::pair<Tensor,int>, Tensor, EmptyModule> {
public:
    explicit Foo(Tensor z, int depth) : DMLGenFn<Foo, args_type, return_type, parameters_type>({z, depth}) {}

    template <typename Tracer>
    return_type exec(Tracer& tracer) const {
        const auto& [z, depth] = tracer.get_args();
        auto& parameters = tracer.get_parameters();
        const auto x = tensor(0.0);
        const auto y = tensor(1.0) + z + x;
        const auto a = tensor(0.0);
        const Tensor z1 = tracer.call({"z1"}, Normal(a, tensor(1.0)));
        const Tensor z2 = tracer.call({"z2"}, Normal(z1, tensor(0.1)));
        Tensor result;
        if (tensor_scalar_bool(z1 < 0.0)) {
            const Tensor z3 = tracer.call({"recursive"}, Foo(z2, depth + 1), parameters);
            result = x + z1 + y + z2 + z3;
        } else {
            result = x + z1 + y + z2;
        }
        return result;// + w;
    }
};

TEST_CASE("simulate", "[dml]") {
    EmptyModule parameters;
    Foo model {tensor(1.0), 0};
    std::random_device rd{};
    std::mt19937 gen{rd()};
    auto trace = model.simulate(gen, parameters, true);
    ChoiceTrie choices = trace.get_choice_trie();
    REQUIRE(choices.get_subtrie({"z1"}).has_value());
    REQUIRE(choices.get_subtrie({"z2"}).has_value());
}

TEST_CASE("generate", "[dml]") {
    EmptyModule parameters;
    Foo model {tensor(1.0), 0};
    std::random_device rd{};
    std::mt19937 gen{rd()};
    ChoiceTrie constraints {};
    constraints.set_value({"z1"}, tensor(-1.0));
    constraints.set_value({"z2"}, tensor(2.0));
    constraints.set_value({"recursive", "z1"}, tensor(1.0));
    constraints.set_value({"recursive", "z2"}, tensor(3.0));
    auto [trace, log_weight] = model.generate(gen, parameters, constraints, true);
    ChoiceTrie choices = trace.get_choice_trie();
    REQUIRE(any_cast<Tensor>(choices.get_value({"z1"})).equal(tensor(-1.0)));
    REQUIRE(any_cast<Tensor>(choices.get_value({"z2"})).equal(tensor(2.0)));
    REQUIRE(any_cast<Tensor>(choices.get_value({"recursive", "z1"})).equal(tensor(1.0)));
    REQUIRE(any_cast<Tensor>(choices.get_value({"recursive", "z2"})).equal(tensor(3.0)));
}

void do_simulate(int idx, int n, std::vector<double>& scores, EmptyModule& parameters) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    double score = 0.0;
    for (int j = 0; j < n; j++) {
        Tensor z = tensor(1.0, TensorOptions().dtype(torch::kFloat64));
        auto model = Foo(z, 0);
        const auto trace = model.simulate(gen, parameters, false);
        score += trace.get_score();
    }
    scores[idx] = score;
}

void do_generate(int idx, int n, std::vector<double>& scores, const ChoiceTrie& constraints, EmptyModule& parameters) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    double total_log_weight = 0.0;
    for (int j = 0; j < n; j++) {
        Tensor z = tensor(1.0, TensorOptions().dtype(torch::kFloat64));
        auto model = Foo(z, 0);
        auto trace_and_log_weight = model.generate(gen, parameters, constraints, false);
        auto trace = std::move(trace_and_log_weight.first);
        double log_weight = trace_and_log_weight.second;
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

/* gradients test */

class GradientsTestGenFn;

class GradientsTestGenFn : public DMLGenFn<GradientsTestGenFn, std::vector<Tensor>, Tensor, EmptyModule> {
public:
    explicit GradientsTestGenFn(Tensor x, Tensor y) :
        DMLGenFn<GradientsTestGenFn, args_type, return_type, parameters_type>({x, y}) {}

    template <typename Tracer>
    return_type exec(Tracer& tracer) const {
        const std::vector<Tensor>& args = tracer.get_args();
        const auto& x = args.at(0);
        const auto& y = args.at(1);
        const Tensor z = tracer.call({"z1"}, Normal(x + y, tensor(1.0)));
        return z + x + (y * 2.0);
    }
};

TEST_CASE("gradients with no parameters", "[gradients, dml]") {
    EmptyModule parameters;
    GradientAccumulator accum {parameters};
    std::random_device rd{};
    std::mt19937 gen{rd()};
    Tensor x = tensor(1.0);
    Tensor y = tensor(1.0);
    auto model = GradientsTestGenFn(x, y);
    ChoiceTrie constraints {};
    Tensor z1 = tensor(1.0);
    constraints.set_value({"z1"}, z1);
    auto [trace, log_weight] = model.generate(gen, parameters, constraints, true);
    Tensor retval_grad = tensor(1.123);
    auto arg_grads = any_cast<std::vector<Tensor>>(trace.gradients(retval_grad, 1.0, accum));
    std::cout << std::endl;
    REQUIRE(arg_grads.size() == 2);
    NormalDist dist {x + y, tensor(1.0)};
    auto logpdf_grad = dist.log_density_gradient(z1);
    Tensor expected_x_grad = tensor(1.123) + std::get<1>(logpdf_grad);
    Tensor expected_y_grad = tensor(1.123 * 2) + std::get<1>(logpdf_grad);
    REQUIRE(arg_grads.at(0).allclose(expected_x_grad));
    REQUIRE(arg_grads.at(1).allclose(expected_y_grad));
}


/* invoking a torch::nn::Module test */

using std::pair;

struct BarModule : public gen::Module {
    BarModule() {
        // register torch modules that we call directly
        // Construct and register three Linear submodules.
        fc1 = register_torch_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_torch_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_torch_module("fc3", torch::nn::Linear(32, 10));
    }
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

class Bar : public DMLGenFn<Bar, pair<Tensor,Tensor>, Tensor, BarModule> {
public:
    explicit Bar(Tensor x, Tensor y) : DMLGenFn<Bar, args_type, return_type, parameters_type>({x, y}) {}

    template <typename Tracer>
    return_type exec(Tracer& tracer) {
        auto& parameters = tracer.get_parameters();
        auto& [x, y] = tracer.get_args();
        Tensor h1 = parameters.fc1->forward(torch::relu(x));
        Tensor h2 = parameters.fc2->forward(torch::relu(h1));
        Tensor h3 = parameters.fc3->forward(torch::relu(h2));
        Tensor mu = torch::sum(h3);
        const Tensor z = tracer.call({"z1"}, Normal(mu, tensor(1.0)));
        return z + mu;
    }
};

struct BazModule : public gen::Module {
    BazModule() {
        fc1 = register_torch_module("fc1", torch::nn::Linear(784, 64));
        bar = register_gen_module("bar", make_shared<BarModule>());
    }
    torch::nn::Linear fc1 {nullptr};
    shared_ptr<BarModule> bar {nullptr};
};

class Baz : public DMLGenFn<Baz, pair<Tensor,Tensor>, Tensor, BazModule> {
public:
    explicit Baz(Tensor x, Tensor y) : DMLGenFn<Baz, args_type, return_type, parameters_type>({x, y}) {}

    template <typename Tracer>
    return_type exec(Tracer& tracer) {
        auto& parameters = tracer.get_parameters();
        auto& [x, y] = tracer.get_args();
        Tensor h1 = parameters.fc1->forward(torch::relu(x));
        Tensor mu = torch::sum(h1);
        const Tensor z = tracer.call({"bar_addr"}, Bar(x, y), *parameters.bar);
        return z + mu;
    }
};

void simulate_and_gradients(int n, BazModule& parameters, GradientAccumulator& accum) {
    std::random_device rd{};
    std::mt19937 rng{rd()};
    Tensor x = torch::rand({784});
    Tensor y = torch::rand({4}); // unused
    Baz model{x, y};
    for (int i = 0; i < n; i++) {
        auto trace = model.simulate(rng, parameters, true);
        trace.gradients(tensor(1.0), 1.0, accum);
    }
}

TEST_CASE("multithreaded simulate and gradients", "[dml, parameters, multithreaded]") {

    BazModule parameters;
    torch::optim::SGD sgd {parameters.all_parameters(), torch::optim::SGDOptions(0.1)};

    GradientAccumulator accum1 {parameters};
    GradientAccumulator accum2 {parameters};

    // NOTE: in a real application we would reuse threads across iterations (e.g. thread pool)
    for (int iter = 0; iter < 10; iter++) {
        std::thread thread1 {simulate_and_gradients, 10, std::ref(parameters), std::ref(accum1)};
        std::thread thread2 {simulate_and_gradients, 10, std::ref(parameters), std::ref(accum2)};
        thread1.join();
        thread2.join();
        accum1.update_module_gradients();
        accum2.update_module_gradients();
        sgd.step();
    }


}