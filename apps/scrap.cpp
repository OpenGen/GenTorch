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


class GradientsTestGenFn;

class GradientsTestGenFn : public DMLGenFn<GradientsTestGenFn, std::vector<Tensor>, Tensor> {
public:
explicit GradientsTestGenFn(Tensor x, Tensor y) : DMLGenFn<GradientsTestGenFn, args_type, return_type>({x, y}) {}

template <typename Tracer>
return_type exec(Tracer& tracer) const {
    const std::vector<Tensor>& args = tracer.get_args();
    const auto& x = args.at(0);
    const auto& y = args.at(1);
    return x + (y + 1.0);
}
};


int main () {

    std::random_device rd{};
    std::mt19937 gen{rd()};
    Tensor x = tensor(1.0, TensorOptions().dtype(torch::kFloat64));
    Tensor y = tensor(1.0, TensorOptions().dtype(torch::kFloat64));
    const auto model = GradientsTestGenFn(x, y);
    auto [trace, log_weight] = model.generate(gen, Trie{}, true);
    Tensor retval_grad = tensor(1.123, TensorOptions().dtype(torch::kFloat64));
    std::vector<Tensor> arg_grads = trace.gradients(retval_grad, 1.0);
    std::cout << arg_grads << std::endl;

}
