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

#pragma once

#include <gen/address.h>
#include <gen/trie.h>
#include <gen/conversions.h>
#include <gen/trace.h>

#include <any>
#include <memory>
#include <optional>

#include <torch/torch.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/ordered_dict.h>

using std::any, std::any_cast;
using std::vector, std::pair;
using std::cout, std::endl;
using std::shared_ptr, std::unique_ptr, std::make_shared;
using std::optional, std::make_optional;

using torch::Tensor;
using torch::nn::Module;
using torch::optim::OptimizerParamGroup;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
using torch::autograd::Node;
using torch::autograd::VariableInfo;
using torch::autograd::edge_list;

namespace gen {

class GenModule;

class GradientAccumulator {
    friend class GenModule;
public:
    explicit GradientAccumulator(const GenModule& module);

private:
    std::vector<Tensor> gradients_;
};


class GenModule {

public:

    virtual ~GenModule() = default;

    Tensor& register_parameter(std::string name, Tensor tensor, bool requires_grad = true) {
    // TODO add checks similar to register_parameter in
    // https://github.com/pytorch/pytorch/blob/9267fd8d7395074001ad7cf2a8f28082dbff6b0b/torch/csrc/api/src/nn/module.cpp
        tensor.set_requires_grad(requires_grad);
        return local_parameters_.insert(std::move(name), std::move(tensor));
    }

    template <typename ModuleType>
    std::shared_ptr<ModuleType> register_torch_module(std::string name, std::shared_ptr<ModuleType> module) {
        auto& base_module = torch_submodules_.insert(std::move(name), std::move(module));
        return std::dynamic_pointer_cast<ModuleType>(base_module);
    }

    template <typename ModuleType>
    std::shared_ptr<ModuleType> register_torch_module(std::string name,
                                                      torch::nn::ModuleHolder<ModuleType> module_holder) {
        return register_torch_module(std::move(name), module_holder.ptr());
    }

    template <typename ModuleType>
    std::shared_ptr<ModuleType> register_gen_module(std::string name, std::shared_ptr<ModuleType> module) {
        auto& base_module = gen_submodules_.insert(std::move(name), std::move(module));
        return std::dynamic_pointer_cast<ModuleType>(base_module);
    }

    void local_parameters(std::vector<Tensor>& parameters) const;

    std::vector<Tensor> local_parameters() const;

    void all_parameters(std::vector<Tensor>& parameters, std::unordered_set<const GenModule*>& visited) const;

    std::vector<Tensor> all_parameters() const;

    void incorporate(const GradientAccumulator& accum) const;

private:
    torch::OrderedDict<std::string, Tensor> local_parameters_;
    torch::OrderedDict<std::string, std::shared_ptr<Module>> torch_submodules_;
    torch::OrderedDict<std::string, std::shared_ptr<GenModule>> gen_submodules_;
};

GradientAccumulator::GradientAccumulator(const GenModule& module) {
    for (const auto& t : module.all_parameters()) {
        gradients_.emplace_back(torch::zeros_like(t).detach());
    }
}

void GenModule::local_parameters(std::vector<Tensor>& parameters) const {
    for (const auto& parameter : local_parameters_) {
        parameters.emplace_back(parameter.value());
    }
    for (const auto& submodule : torch_submodules_) {
        // TODO consider optimizing this to avoid using torch::nn::Module::parameters() but instead
        // writing directly into our parameters vector
        for (const auto& parameter : submodule.value()->parameters()) {
            parameters.emplace_back(parameter);
        }
    }
}

std::vector<Tensor> GenModule::local_parameters() const {
    std::vector<Tensor> result;
    local_parameters(result);
    return result;
}

void GenModule::all_parameters(std::vector<Tensor>& parameters, std::unordered_set<const GenModule*>& visited) const {
    local_parameters(parameters);
    visited.emplace(this);
    for (const auto& submodule : gen_submodules_) {
        if (visited.find(submodule.value().get()) == visited.end()) {
            submodule.value()->all_parameters(parameters, visited);
        }
    }
}

std::vector<Tensor> GenModule::all_parameters() const {
    std::vector<Tensor> result;
    std::unordered_set<const GenModule*> visited;
    all_parameters(result, visited);
    return result;
}

void GenModule::incorporate(const GradientAccumulator& accum) const {
    size_t i = 0;
    // TODO depending on profiling, consider inlining parameters
    std::vector<Tensor> parameters = all_parameters();
    for (const auto& t : accum.gradients_) {
        Tensor& grad = parameters[i++].mutable_grad();
        if (grad.defined()) {
            grad.add_(t);
        } else {
            grad = t.clone();
        }
        t.zero_();
    }

}

}