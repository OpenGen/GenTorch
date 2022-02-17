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

#ifndef GENTORCH_AUTODIFF_H
#define GENTORCH_AUTODIFF_H

#include <vector>
#include <utility>
#include <memory>

#include <torch/torch.h>

#include <gentorch/conversions.h>
#include <gentorch/parameters.h>

using std::vector;
using std::pair;
using std::unique_ptr;
using std::make_unique;
using torch::Tensor;
using torch::tensor;
using torch::autograd::variable_list;
using torch::autograd::edge_list;
using torch::autograd::Node;

using gentorch::GradientAccumulator;

namespace gentorch::dml {

template<typename T>
T detach_clone_and_track(const T &args) {
    c10::InferenceMode guard{false};
    vector<Tensor> args_unrolled = unroll(args);
    vector<Tensor> args_unrolled_copy;
    for (auto &t: args_unrolled) {
        args_unrolled_copy.emplace_back(t.detach().clone().set_requires_grad(true));
    }
    return roll(args_unrolled_copy, args); // copy elision
}

template<typename args_type>
std::unique_ptr<pair<const args_type&, unique_ptr<const args_type>>> maybe_track_args(const args_type &args,
                                                                      bool prepare_for_gradients) {
    if (prepare_for_gradients) {
        auto tracked_args_ptr = make_unique<const args_type>(detach_clone_and_track(args));
        return std::make_unique<pair<const args_type&, unique_ptr<const args_type>>>(*tracked_args_ptr, std::move(tracked_args_ptr));
    } else {
        return std::make_unique<pair<const args_type&, unique_ptr<const args_type>>>(args, nullptr);
    }
}



// http://blog.ezyang.com/2019/05/pytorch-internals/
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/autograd/function.h#L50
// https://discuss.pytorch.org/t/extending-autograd-from-c/76240/5

struct GradientHelper {
    double *scaler_ptr_{nullptr};
    GradientAccumulator *accumulator_ptr_{nullptr};
};

template<typename args_type, typename return_value_type, typename subtrace_type>
struct MyGradNode : public Node {
    explicit MyGradNode(subtrace_type &subtrace, const GradientHelper &helper, edge_list &&next_edges)
            : Node(std::move(next_edges)), subtrace_{subtrace},
              helper_{helper} {}

    ~MyGradNode() override = default;

    variable_list apply(variable_list &&output_grad) override {
        const return_value_type &return_value = subtrace_.return_value();

        // read off the return value gradient from the output_grad
        auto[num_read, return_value_grad] = roll(output_grad, 0, return_value);
        // check that we read off all but the last element, which is the dummy element
        assert(num_read == output_grad.size() - 1);

        args_type args_grad = subtrace_.parameter_gradient(
                *helper_.accumulator_ptr_, return_value_grad, *helper_.scaler_ptr_);
        std::vector<Tensor> input_grad = unroll(args_grad);
        input_grad.emplace_back(torch::tensor(0.0));
        return input_grad;
    }

private:
    subtrace_type &subtrace_;
    const GradientHelper &helper_;
};

template<typename args_type, typename return_type, typename subtrace_type>
struct MyNode : public Node {
    explicit MyNode(subtrace_type &subtrace, const GradientHelper &helper)
            : subtrace_{subtrace}, helper_{helper} {}

    ~MyNode() override = default;

    variable_list apply(variable_list &&inputs) override {
        const return_type &value = subtrace_.return_value();
        vector<Tensor> output = unroll(value);

        // value that we use to ensure the corresponding MyGradNode is visited in the backward pass
        Tensor dummy = torch::tensor(0.0); // TODO optimize me, maybe make static?
        output.emplace_back(dummy);

        return torch::autograd::wrap_outputs(inputs, std::move(output), [&](edge_list &&next_edges) {
            return std::make_shared<MyGradNode<args_type, return_type, subtrace_type>>(subtrace_, helper_,
                                                                                       std::move(next_edges));
        });
    }

private:
    subtrace_type &subtrace_;
    const GradientHelper &helper_;
};

}

#endif //GENTORCH_AUTODIFF_H
