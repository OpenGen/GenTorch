#include <memory>
#include <torch/torch.h>
#include <gen/module.h>
#include <cassert>

using torch::Tensor;

class BarModule;

struct FooModule : public gen::GenModule {

    FooModule(int64_t n, int64_t m) {
        a = register_parameter("a", torch::randn({n, m}));
        layer1 = register_torch_module("layer1", torch::nn::Linear(n, m));
    }

    // parameters
    Tensor a;

    // torch modules
    torch::nn::Linear layer1 {nullptr};

    // gen modules
    std::shared_ptr<BarModule> bar {nullptr};
};

struct BarModule : public gen::GenModule {

    BarModule(int64_t n, int64_t m) {
        b = register_parameter("b", torch::randn({n, m}));
        layer2 = register_torch_module("layer2", torch::nn::Linear(n, m));
    }

    // parameters
    Tensor b;

    // torch modules
    torch::nn::Linear layer2 {nullptr};

    // gen modules
    std::shared_ptr<FooModule> foo {nullptr};
};

int main() {

    shared_ptr<FooModule> foo = make_shared<FooModule>(10, 3);
    shared_ptr<BarModule> bar = make_shared<BarModule>(5, 4);
    
    // necessary to stitch together afterwards because of recursion
    foo->bar = foo->register_gen_module("bar", bar);
    bar->foo = bar->register_gen_module("foo", foo);

    std::vector<Tensor> foo_locals = foo->local_parameters();
    assert(foo_locals.size() == 3);
    assert(foo_locals[0].data_ptr<float>() == foo->a.data_ptr<float>());
    assert(foo_locals[1].data_ptr<float>() == foo->layer1->weight.data_ptr<float>());
    assert(foo_locals[2].data_ptr<float>() == foo->layer1->bias.data_ptr<float>());

    std::vector<Tensor> bar_locals = bar->local_parameters();
    assert(bar_locals.size() == 3);
    assert(bar_locals[0].data_ptr<float>() == bar->b.data_ptr<float>());
    assert(bar_locals[1].data_ptr<float>() == bar->layer2->weight.data_ptr<float>());
    assert(bar_locals[2].data_ptr<float>() == bar->layer2->bias.data_ptr<float>());

    std::vector<Tensor> foo_all = foo->all_parameters();
    assert(foo_all.size() == 6);
    assert(foo_all[0].data_ptr<float>() == foo->a.data_ptr<float>());
    assert(foo_all[1].data_ptr<float>() == foo->layer1->weight.data_ptr<float>());
    assert(foo_all[2].data_ptr<float>() == foo->layer1->bias.data_ptr<float>());
    assert(foo_all[3].data_ptr<float>() == bar->b.data_ptr<float>());
    assert(foo_all[4].data_ptr<float>() == bar->layer2->weight.data_ptr<float>());
    assert(foo_all[5].data_ptr<float>() == bar->layer2->bias.data_ptr<float>());

    std::vector<Tensor> bar_all = bar->all_parameters();
    assert(bar_all.size() == 6);
    assert(bar_all[0].data_ptr<float>() == bar->b.data_ptr<float>());
    assert(bar_all[1].data_ptr<float>() == bar->layer2->weight.data_ptr<float>());
    assert(bar_all[2].data_ptr<float>() == bar->layer2->bias.data_ptr<float>());
    assert(bar_all[3].data_ptr<float>() == foo->a.data_ptr<float>());
    assert(bar_all[4].data_ptr<float>() == foo->layer1->weight.data_ptr<float>());
    assert(bar_all[5].data_ptr<float>() == foo->layer1->bias.data_ptr<float>());

    torch::optim::SGD sgd {foo->all_parameters(), torch::optim::SGDOptions{0.01}};
    gen::GradientAccumulator accum1 {*foo};
    gen::GradientAccumulator accum2 {*foo};
    foo->incorporate(accum1);
    foo->incorporate(accum2);
    sgd.step();
}