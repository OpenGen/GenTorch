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
#include <gen/parameters.h>

#include <cassert>
#include <memory>

#include <torch/torch.h>


class BarModule;

struct FooModule : public gen::Module {

    FooModule(int64_t n, int64_t m) {
        a = register_parameter("a", torch::randn({n, m}));
        layer1 = register_torch_module("layer1", torch::nn::Linear(3 * n, 3 * m));
    }

    // parameters
    Tensor a;

    // torch modules
    torch::nn::Linear layer1 {nullptr};

    // gen modules
    std::shared_ptr<BarModule> bar {nullptr};
};

struct BarModule : public gen::Module {

    BarModule(int64_t n, int64_t m) {
        b = register_parameter("b", torch::randn({n, m}));
        layer2 = register_torch_module("layer2", torch::nn::Linear(2 * n, 2 * m));
    }

    // parameters
    Tensor b;

    // torch modules
    torch::nn::Linear layer2 {nullptr};

    // gen modules
    std::shared_ptr<FooModule> foo {nullptr};
};

bool same_data(Tensor a, Tensor b) {
    return a.data_ptr<float>() == b.data_ptr<float>();
}

TEST_CASE("parameters", "[Module]") {
    
    std::shared_ptr<FooModule> foo = std::make_shared<FooModule>(10, 3);
    std::shared_ptr<BarModule> bar = std::make_shared<BarModule>(5, 4);
    
    // necessary to stitch together afterwards because of recursion
    foo->bar = foo->register_gen_module("bar", bar);
    bar->foo = bar->register_gen_module("foo", foo);
    
    std::vector<Tensor> foo_locals = foo->local_parameters();
    REQUIRE(foo_locals.size() == 3);
    REQUIRE(same_data(foo_locals[0], foo->a));
    REQUIRE(same_data(foo_locals[1], foo->layer1->weight));
    REQUIRE(same_data(foo_locals[2], foo->layer1->bias));
    
    std::vector<Tensor> bar_locals = bar->local_parameters();
    REQUIRE(bar_locals.size() == 3);
    REQUIRE(same_data(bar_locals[0], bar->b));
    REQUIRE(same_data(bar_locals[1], bar->layer2->weight));
    REQUIRE(same_data(bar_locals[2], bar->layer2->bias));
    
    std::vector<Tensor> foo_all = foo->all_parameters();
    REQUIRE(foo_all.size() == 6);
    REQUIRE(same_data(foo_all[0], foo->a));
    REQUIRE(same_data(foo_all[1], foo->layer1->weight));
    REQUIRE(same_data(foo_all[2], foo->layer1->bias));
    REQUIRE(same_data(foo_all[3], bar->b));
    REQUIRE(same_data(foo_all[4], bar->layer2->weight));
    REQUIRE(same_data(foo_all[5], bar->layer2->bias));
    
    std::vector<Tensor> bar_all = bar->all_parameters();
    REQUIRE(bar_all.size() == 6);
    REQUIRE(same_data(bar_all[0], bar->b));
    REQUIRE(same_data(bar_all[1], bar->layer2->weight));
    REQUIRE(same_data(bar_all[2], bar->layer2->bias));
    REQUIRE(same_data(bar_all[3], foo->a));
    REQUIRE(same_data(bar_all[4], foo->layer1->weight));
    REQUIRE(same_data(bar_all[5], foo->layer1->bias));
    
    torch::optim::SGD sgd {foo->all_parameters(), torch::optim::SGDOptions{0.01}};
    gen::GradientAccumulator accum1 {foo};
    gen::GradientAccumulator accum2 {foo};
    accum1.update_module_gradients();
    accum2.update_module_gradients();
    sgd.step();
    
    auto it = accum1.begin(*foo);
    REQUIRE((*(it++)).sizes().equals(foo->a.sizes()));
    REQUIRE((*(it++)).sizes().equals(foo->layer1->weight.sizes()));
    REQUIRE((*(it++)).sizes().equals(foo->layer1->bias.sizes()));
    REQUIRE((*(it++)).sizes().equals(bar->b.sizes()));
    REQUIRE((*(it++)).sizes().equals(bar->layer2->weight.sizes()));
    REQUIRE((*(it++)).sizes().equals(bar->layer2->bias.sizes()));
    REQUIRE((*accum1.end(*foo)).sizes().equals(bar->b.sizes()));

}
