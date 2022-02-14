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

#ifndef GENTORCH_PARAMETER_GRADIENT_H
#define GENTORCH_PARAMETER_GRADIENT_H

#include <gen/address.h>
#include <gen/trie.h>
#include <gen/conversions.h>
#include <gen/trace.h>
#include <gen/parameters.h>

#include <gentl/concepts.h>
#include <gentl/types.h>

#include <torch/torch.h>
#include <torch/csrc/autograd/functions/utils.h>

#include <any>
#include <memory>
#include <optional>

using gentl::SimulateOptions;
using gentl::GenerateOptions;
using gentl::UpdateOptions;

using std::any, std::any_cast;
using std::vector, std::pair;
using std::cout, std::endl;
using std::shared_ptr, std::unique_ptr, std::make_shared;
using std::optional, std::make_optional;

using torch::Tensor;
using torch::optim::OptimizerParamGroup;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
using torch::autograd::Node;
using torch::autograd::VariableInfo;
using torch::autograd::edge_list;

using gen::GradientAccumulator;

namespace gen::dml {

template <typename Model>
template <typename callee_args_type, typename callee_return_type, typename subtrace_type>
callee_return_type DMLTrace<Model>::make_tracked_return_value(subtrace_type &subtrace, const callee_args_type &tracked_args, const callee_return_type &value) {
    auto node = MyNode<callee_args_type, callee_return_type, subtrace_type>(subtrace, *helper_);
    // NOTE: gen_fn_with_args.get_args() returns args that are tracked as part of the autograd graph
    std::vector<Tensor> inputs = unroll(tracked_args);
    assert(dummy_input_.requires_grad());
    inputs.emplace_back(dummy_input_);
    auto outputs = node(std::move(inputs));
    auto[num_read, tracked_value] = roll(outputs, 0, value);
    assert(num_read == outputs.size() - 1);
    Tensor dummy = outputs[num_read];
    assert(dummy.requires_grad());
    dummy_output_ += dummy;
    assert(dummy_output_.requires_grad());
    return tracked_value;
}


template<typename Model>
typename DMLTrace<Model>::args_type DMLTrace<Model>::parameter_gradient(GradientAccumulator &accumulator, double scaler) {
    return parameter_gradient(accumulator, zero_gradient(return_value()), scaler);
}

template<typename Model>
typename DMLTrace<Model>::args_type DMLTrace<Model>::parameter_gradient(GradientAccumulator &accumulator, return_type retval_grad, double scaler) {

    // the computation graph is already set up, there should be no Tensors created anyways
    c10::InferenceMode guard{true};

    if (!prepared_for_gradients_) {
        throw std::logic_error("not ready for gradients");
    }

    // add the arguments to the inputs
    vector<Tensor> inputs = unroll(get_args());
    size_t num_args = inputs.size();

    // add the local parameters to the inputs
    vector<Tensor> local_parameter_tensors = parameters_.local_parameters();
    size_t num_local_parameters = local_parameter_tensors.size();
    inputs.insert(inputs.end(), local_parameter_tensors.begin(), local_parameter_tensors.end());

    // add the dummy input
    assert(dummy_input_.requires_grad());
    inputs.emplace_back(dummy_input_);

    // add the return value to the outputs
    const return_type &retval = return_value();
    vector<Tensor> outputs = unroll(retval);
    vector<Tensor> output_grads = unroll(retval_grad);

    // add the dummy output
    assert(dummy_output_.requires_grad());
    outputs.emplace_back(dummy_output_);
    output_grads.emplace_back(torch::tensor(1.0));

    // pass information back to the calls to parameter_gradient() that happen within the backward pass will have
    // NOTE: not thread safe, but this is okay, traces aren't intended to be thread safe
    helper_->scaler_ptr_ = &scaler;
    helper_->accumulator_ptr_ = &accumulator;

    // do the backward pass. this recursively invokes parameter_gradient() for each subtrace
    // NOTE: the returned input grads do not include parameters for callee generative functions
    // TODO check why we are setting allow_unused to true
    vector<Tensor> input_grads = torch::autograd::grad(outputs, inputs, output_grads, {}, false, true);

    // read off argument gradients
    vector<Tensor> args_grad_unrolled;
    size_t i;
    for (i = 0; i < num_args; ++i) {
        args_grad_unrolled.emplace_back(input_grads[i]);
    }
    args_type args_grad = roll(args_grad_unrolled, get_args());

    // read off local parameter gradients and increment the accumulators
    assert(i == num_args);
    for (auto it = accumulator.begin(parameters_); it != accumulator.end(parameters_); ++it) {
        assert((*it).sizes().equals(input_grads[i].sizes()));
        (*it).add_(input_grads[i++], scaler);
    }

    // the last element should be the dummy
    assert(i == input_grads.size() - 1);

    return std::move(args_grad);
}


}

#endif //GENTORCH_PARAMETER_GRADIENT_H
