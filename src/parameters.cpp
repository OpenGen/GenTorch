
#include <gen/parameters.h>

#include <memory>

#include <torch/torch.h>

using torch::Tensor;

namespace gen {

// gen::GradientAccumulator implementation

std::vector<Tensor>::const_iterator GradientAccumulator::begin(const Parameters& submodule) {
    auto [begin_idx, end_idx] = begin_end_idx_.at(&submodule); // TODO throw nice error if key is not found
    return gradients_.begin() + begin_idx;
}

std::vector<Tensor>::const_iterator GradientAccumulator::end(const Parameters& submodule) {
    auto [begin_idx, end_idx] = begin_end_idx_.at(&submodule); // TODO throw nice error if key is not found
    return gradients_.begin() + end_idx;
}

void GradientAccumulator::update_module_gradients(bool reset) {
    c10::InferenceMode guard {true};
    size_t i = 0;
    for (const auto& source : gradients_) {
        Tensor& destination = all_parameters_[i++].mutable_grad();
        if (destination.defined()) {
            destination.sub_(source); // because torch optimizers always minimize
        } else {
            destination = source.negative();
        }
        if (reset) {
            source.zero_();
        }
    }
}

GradientAccumulator::GradientAccumulator(const Parameters& module) {
    c10::InferenceMode guard {true};
    module.all_parameters(all_parameters_, begin_end_idx_);
    for (const auto& t : all_parameters_) {
        gradients_.emplace_back(torch::zeros_like(t));
    }
}

GradientAccumulator::GradientAccumulator(std::shared_ptr<Parameters> module_ptr) : GradientAccumulator{*module_ptr} {}

// gen::Module implementation

Tensor& Parameters::register_parameter(std::string name, Tensor tensor, bool requires_grad) {
    // TODO add checks similar to register_parameter in
    // https://github.com/pytorch/pytorch/blob/9267fd8d7395074001ad7cf2a8f28082dbff6b0b/torch/csrc/api/src/nn/module.cpp
    tensor.set_requires_grad(requires_grad);
    return local_parameters_.insert(std::move(name), std::move(tensor));
}

void Parameters::local_parameters(std::vector<Tensor>& parameters) const {
    c10::InferenceMode guard {true};
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

std::vector<Tensor> Parameters::local_parameters() const {
    std::vector<Tensor> result;
    local_parameters(result);
    return result;
}

void Parameters::all_parameters(std::vector<Tensor>& parameters,
                                std::unordered_map<const Parameters*, std::pair<size_t,size_t>>& begin_end_idx) const {
    size_t begin_idx = parameters.size();
    local_parameters(parameters);
    size_t end_idx = parameters.size();
    begin_end_idx[this] = {begin_idx, end_idx};
    for (const auto& submodule : gen_submodules_) {
        if (begin_end_idx.find(submodule.value().get()) == begin_end_idx.end()) {
            submodule.value()->all_parameters(parameters, begin_end_idx);
        }
    }
}

std::vector<Tensor> Parameters::all_parameters() const {
    std::vector<Tensor> parameters;
    std::unordered_map<const Parameters*, std::pair<size_t,size_t>> begin_end_idx;
    all_parameters(parameters, begin_end_idx);
    return parameters;
}

}