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

#include <gen/address.h>
#include <gen/trie.h>
#include <gen/trace.h>
#include <gen/dml.h>

#include <any>
#include <memory>
#include <optional>

#include <torch/torch.h>

using std::any_cast;
using std::shared_ptr;
using std::optional;
using torch::Tensor;

template <typename Model>
std::any DMLTrace<Model>::get_return_value() const {
    if (!maybe_value_.has_value()) {
        throw std::runtime_error("set_value was never called");
    }
    return maybe_value_.value();
}

template <typename Model>
template <typename SubtraceType>
void DMLTrace<Model>::add_subtrace(const Address& address, SubtraceType subtrace) {
    score_ += subtrace.get_score();
    try {
        subtraces_->set_value(address, subtrace, false);
    } catch (const TrieOverwriteError&) {
        throw DMLAlreadyVisitedError(address);
    }
}

template <typename Model>
bool DMLTrace<Model>::has_subtrace(const Address& address) const {
    return subtraces_->get_subtrie(address).has_value();
}

template <typename Model>
const Trace& DMLTrace<Model>::get_subtrace(const Address& address) const {
    return *any_cast<Trace>(&subtraces_->get_value(address));
}

template <typename Model>
template <typename Generator>
typename DMLTrace<Model>::update_return_type DMLTrace<Model>::update(Generator& gen, const Model& model_with_args,
                                                           const Trie& constraints) const {
    auto tracer = DMLUpdateTracer<Generator, Model>(gen, constraints);
    auto value = model_with_args.exec(tracer);
    return tracer.finish(value);
}

template <typename Model>
Trie DMLTrace<Model>::get_choice_trie(const Trie& subtraces) {
    Trie trie {};
    // TODO handle calls at the empty address properly
    for (const auto& [key, subtrie] : subtraces.subtries()) {
        if (subtrie.has_value()) {
            const auto* subtrace = any_cast<Trace>(&subtrie.get_value());
            trie.set_subtrie(Address{key}, subtrace->get_choice_trie());
        } else if (subtrie.empty()) {
        } else {
            trie.set_subtrie(Address{key}, get_choice_trie(subtrie));
        }
    }
    return trie; // copy elision
}

// *******************
// * Simulate tracer *
// *******************

template <typename Generator, typename Model>
class DMLSimulateTracer {
public:
    explicit DMLSimulateTracer(Generator& gen)
            : finished_(false), gen_{gen}, trace_{} { }

    template <typename CalleeType>
    typename CalleeType::return_type
    call(const Address& address, const CalleeType& gen_fn_with_args) {
        assert(!finished_); // if this assertion fails, it is a bug in DML not user code
        typename CalleeType::trace_type subtrace = gen_fn_with_args.simulate(gen_);
        const auto& value = any_cast<typename CalleeType::return_type>(subtrace.get_return_value());
        trace_.add_subtrace(address, std::move(subtrace));
        return value;
    }

    DMLTrace<Model> finish(typename Model::return_type value) {
        finished_ = true;
        trace_.set_value(value);
        return std::move(trace_);
    }

private:
    bool finished_;
    Generator& gen_;
    DMLTrace<Model> trace_;
};

// *******************
// * Generate tracer *
// *******************

template <typename Generator, typename Model>
class DMLGenerateTracer {
public:
    explicit DMLGenerateTracer(Generator& gen, const Trie& constraints)
            : finished_(false), gen_{gen}, trace_{}, log_weight_(0.0), constraints_(constraints) { }

    template <typename CalleeType>
    typename CalleeType::return_type
    call(const Address& address, const CalleeType& gen_fn_with_args) {
        assert(!finished_); // if this assertion fails, it is a bug in DML not user code
        Trie sub_constraints { constraints_.get_subtrie(address, false) };
        auto subtrace_and_log_weight = gen_fn_with_args.generate(gen_, sub_constraints);
        typename CalleeType::trace_type subtrace = std::move(subtrace_and_log_weight.first);
        double log_weight_increment = subtrace_and_log_weight.second;
        log_weight_ += log_weight_increment;
        const auto value = any_cast<typename CalleeType::return_type>(subtrace.get_return_value());
        trace_.add_subtrace(address, std::move(subtrace));
        return value;
    }

    std::pair<DMLTrace<Model>,double> finish(typename Model::return_type value) {
        finished_ = true;
        trace_.set_value(value);
        return std::pair(std::move(trace_), log_weight_);
    }

private:
    double log_weight_;
    bool finished_;
    Generator& gen_;
    DMLTrace<Model> trace_;
    const Trie& constraints_;
};

// *******************
// * Update tracer *
// *******************

template <typename Generator, typename Model>
class DMLUpdateTracer {
public:
    explicit DMLUpdateTracer(Generator& gen, const Trie& constraints)
            : finished_(false), gen_{gen}, trace_{},
              log_weight_(0.0), constraints_(constraints) { }

    template <typename CalleeType>
    typename CalleeType::return_type
    call(const Address& address, const CalleeType& gen_fn_with_args) {
        assert(!finished_); // if this assertion fails, it is a bug in DML not user code
        Trie sub_constraints { constraints_.get_subtrie(address, false) };
        typename CalleeType::trace_type subtrace;
        if (prev_trace_.has_subtrace(address)) {
            auto& prev_subtrace = any_cast<typename CalleeType::trace_type&>(prev_trace_.get_subtrace(address));
            auto update_result = prev_subtrace.update(gen_, gen_fn_with_args, sub_constraints);
            subtrace = std::get<0>(update_result);
            log_weight_ += std::get<1>(update_result);
            auto discard = std::get<2>(update_result);
            discard_.set_subtrie(address, discard);
        } else {
            auto generate_result = gen_fn_with_args.generate(gen_, sub_constraints);
            subtrace = generate_result.first;
            double log_weight_increment = generate_result.second;
            log_weight_ += log_weight_increment;
        }
        const auto value = any_cast<typename CalleeType::return_type>(subtrace.get_return_value());
        trace_.add_subtrace(address, std::move(subtrace));
        return value;
    }

    std::tuple<DMLTrace<Model>,double,Trie> finish(typename Model::return_type value) {
        log_weight_ += 0; // TODO decrement for all visited
        // TODO discard all that were not visied (using update method of Trie, which still needs to be implemented)
        finished_ = true;
        trace_.set_value(value);
        return std::tuple(std::move(trace_), log_weight_, discard_);
    }

private:
    double log_weight_;
    bool finished_;
    Generator& gen_;
    const DMLTrace<Model>& prev_trace_;
    DMLTrace<Model> trace_;
    const Trie& constraints_;
    Trie discard_;
};


template <typename Model, typename ArgsType, typename ReturnType>
template <typename Generator>
DMLTrace<Model> DMLGenFn<Model,ArgsType,ReturnType>::simulate(Generator& gen) const {
    auto tracer = DMLSimulateTracer<Generator,Model>(gen);
    auto value = static_cast<const Model*>(this)->exec(tracer);
    return tracer.finish(value);
}

template <typename Model, typename ArgsType, typename ReturnType>
template <typename Generator>
std::pair<DMLTrace<Model>,double> DMLGenFn<Model,ArgsType,ReturnType>::generate(
        Generator& gen, const Trie& constraints) const {
    auto tracer = DMLGenerateTracer<Generator,Model>(gen, constraints);
    auto value = static_cast<const Model*>(this)->exec(tracer);
    return tracer.finish(value);
}