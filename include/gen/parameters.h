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

#include <memory>

#include <torch/torch.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/ordered_dict.h>

using torch::Tensor;

namespace gen {

class Parameters;

/**
 * Stores a copy of the gradients of a generative function.
 *
 * For use in accumulating gradients with Trace::gradients.
 * Typically there will be multiple GradientAccumulators for a given Module.
 *
 * Also see Module.
 *
 * Not thread-safe. Each thread should have its own GradientAccumulator.
 */
class GradientAccumulator {
    friend class Parameters;
public:
    explicit GradientAccumulator(const Parameters& module);
    explicit GradientAccumulator(std::shared_ptr<Parameters> module_ptr);
    std::vector<Tensor>::const_iterator begin(const Parameters& submodule);
    std::vector<Tensor>::const_iterator end(const Parameters& submodule);
    void update_module_gradients(bool reset = true);
private:
    std::vector<Tensor> all_parameters_;
    std::vector<Tensor> gradients_;
    std::unordered_map<const Parameters*, std::pair<size_t,size_t>> begin_end_idx_;
};

/**
 * Stores the parameters of a generative function all its callee torch modules and callee generative functions.
 *
 * Every parameter is a torch::Tensor.
 * Parameters are either 'local' or 'non-local'.
 * Local parameters are registered with `register_parameter`.
 * Non-local parameters are those owned by callee torch modules or callee generative functions.
 * Not thread-safe.
 *
 * See `::GradientAccumulator`.
 *
 */
class Parameters {
    friend class GradientAccumulator;
private:
    torch::OrderedDict<std::string, Tensor> local_parameters_;
    torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> torch_submodules_;
    torch::OrderedDict<std::string, std::shared_ptr<Parameters>> gen_submodules_;
public:
    typedef GradientAccumulator accumulator_t; // part of the concept
    virtual ~Parameters() = default;

    std::vector<Tensor> local_parameters() const;

    std::vector<Tensor> all_parameters() const;

    Tensor& register_parameter(std::string name, Tensor tensor, bool requires_grad = true);

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

private:
    void local_parameters(std::vector<Tensor>& parameters) const;
    void all_parameters(std::vector<Tensor>& parameters,
                        std::unordered_map<const Parameters*, std::pair<size_t,size_t>>& begin_end_idx) const;
};

class EmptyModule : public Parameters {};

// NOTE: cannot currently be used with DMLGenFn's that use EmptyModule, only distributions
const EmptyModule empty_module_singleton {};

}