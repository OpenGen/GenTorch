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
#include <gen/address.h>
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
using distributions::normal::Normal;
using distributions::normal::NormalDist;


bool tensor_scalar_bool(Tensor a) {
    if (a.ndimension() != 0) {
        throw std::logic_error("bool error");
    }
    return *(a.data_ptr<bool>());
}

class Foo;

class Foo : public DMLGenFn<Foo, std::pair<Tensor,int>, Tensor> {
public:
    explicit Foo(Tensor z, int depth) : DMLGenFn<Foo, args_type, return_type>({z, depth}) {}

    template <typename Tracer>
    return_type exec(Tracer& tracer) const {
        const auto& [z, depth] = tracer.get_args();
        const auto x = tensor(0.0);
        const auto y = tensor(1.0) + z + x;
        const auto a = tensor(0.0);
        const Tensor z1 = tracer.call(Address{"z1"}, Normal(a, tensor(1.0)));
        const Tensor z2 = tracer.call(Address{"z2"}, Normal(z1, tensor(0.1)));
        Tensor result;
        if (tensor_scalar_bool(z1 < 0.0)) {
            const Tensor z3 = tracer.call(Address{"recursive"}, Foo(z2, depth + 1));
            result = x + z1 + y + z2 + z3;
        } else {
            result = x + z1 + y + z2;
        }
        return result;// + w;
    }
};

void do_simulate(int idx, int n, std::vector<double>& scores) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    double score = 0.0;
    for (int j = 0; j < n; j++) {
        Tensor z = tensor(1.0, TensorOptions().dtype(torch::kFloat64));
        const auto model = Foo(z, 0);
        const auto trace = model.simulate(gen, false);
        score += trace.get_score();
    }
    scores[idx] = score;
}

void do_generate(int idx, int n, std::vector<double>& scores, const Trie& constraints) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    double total_log_weight = 0.0;
    for (int j = 0; j < n; j++) {
        Tensor z = tensor(1.0, TensorOptions().dtype(torch::kFloat64));
        const auto model = Foo(z, 0);
        auto trace_and_log_weight = model.generate(gen, constraints, false);
        auto trace = std::move(trace_and_log_weight.first);
        double log_weight = trace_and_log_weight.second;
        total_log_weight += log_weight;
    }
    scores[idx] = total_log_weight;
}


TEST_CASE("dummy2", "[dummy]") {
    std::cout << "inside test_dml..." << std::endl;
    dummy();
}

TEST_CASE("multithreaded_simulate", "[multithreading, dml]") {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    int n = 1000;
    int n_threads = 1;
    std::vector<double> scores(n_threads, 0.0);
    std::vector<std::thread> threads;
    for (int i = 0; i < n_threads; i++) {
        threads.push_back(std::thread(do_simulate, i, n, std::ref(scores)));
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
    Trie constraints {};
    constraints.set_value(Address{"z2"}, tensor(1.0));
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    int n = 1000;
    int n_threads = 1;
    std::vector<double> scores(n_threads, 0.0);
    std::vector<std::thread> threads;
    for (int i = 0; i < n_threads; i++) {
        do_generate(i, n, scores, constraints);
        threads.push_back(
                std::thread(do_generate, i, n, std::ref(scores), std::ref(constraints)));
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

class GradientsTestGenFn : public DMLGenFn<GradientsTestGenFn, std::vector<Tensor>, Tensor> {
public:
    explicit GradientsTestGenFn(Tensor x, Tensor y) : DMLGenFn<GradientsTestGenFn, args_type, return_type>({x, y}) {}

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
    std::random_device rd{};
    std::mt19937 gen{rd()};
    Tensor x = tensor(1.0);
    Tensor y = tensor(1.0);
    const auto model = GradientsTestGenFn(x, y);
    Trie constraints {};
    Tensor z1 = tensor(1.0);
    constraints.set_value({"z1"}, z1);
    auto [trace, log_weight] = model.generate(gen, constraints, true);
    Tensor retval_grad = tensor(1.123);
    auto arg_grads = any_cast<std::vector<Tensor>>(trace.gradients(retval_grad, 1.0));
    std::cout << std::endl;
    REQUIRE(arg_grads.size() == 2);
    NormalDist dist {x + y, tensor(1.0)};
    auto logpdf_grad = dist.log_density_gradient(z1);
    Tensor expected_x_grad = tensor(1.123) + std::get<1>(logpdf_grad);
    Tensor expected_y_grad = tensor(1.123 * 2) + std::get<1>(logpdf_grad);
    REQUIRE(arg_grads.at(0).allclose(expected_x_grad));
    REQUIRE(arg_grads.at(1).allclose(expected_y_grad));
}
