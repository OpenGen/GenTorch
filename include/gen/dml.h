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
#include <gen/trace.h>

#include <any>
#include <memory>
#include <optional>

#include <torch/torch.h>

using std::any_cast;
using std::shared_ptr;
using std::optional;
using torch::Tensor;


// *************
// * DML Trace *
// *************

class DMLAlreadyVisitedError : public std::exception {
public:
    explicit  DMLAlreadyVisitedError(const Address& address) : msg_ {
            [&](){
                std::stringstream ss;
                ss << "Address already visited: " << address;
                return ss.str(); }()
    } { }
    [[nodiscard]] const char* what() const noexcept override {
        return msg_.c_str();
    }
private:
    const std::string msg_;
};

template <typename Generator, typename Model>
class DMLUpdateTracer;

template <typename Model>
class DMLTrace : Trace {
public:
    typedef typename Model::return_type return_type;
    typedef typename Model::args_type args_type;
    typedef std::pair<DMLTrace<Model>, double> update_return_type;

    DMLTrace() : subtraces_{make_shared<Trie>()}, score_{0.0} { }
    DMLTrace(const DMLTrace& other) = default;
    DMLTrace(DMLTrace&& other) noexcept = default;

    DMLTrace& operator= (const DMLTrace& other) = default;
    DMLTrace& operator= (DMLTrace&& other) noexcept = default;

    ~DMLTrace() = default;

    [[nodiscard]] double get_score() const override { return score_; }

    [[nodiscard]] std::any get_return_value() const {
        if (!maybe_value_.has_value()) {
            throw std::runtime_error("set_value was never called");
        }
        return maybe_value_.value();
    }

    template <typename SubtraceType>
    void add_subtrace(const Address& address, SubtraceType subtrace) {
        score_ += subtrace.get_score();
        try {
            subtraces_->set_value(address, subtrace, false);
        } catch (const TrieOverwriteError&) {
            throw DMLAlreadyVisitedError(address);
        }
    }

    [[nodiscard]]bool has_subtrace(const Address& address) const {
        return subtraces_->get_subtrie(address).has_value();
    }

    [[nodiscard]]const Trace& get_subtrace(const Address& address) const {
        return *any_cast<Trace>(&subtraces_->get_value(address));
    }

    template <typename Generator>
    update_return_type update(Generator& gen, const Model& model_with_args,
                                                                         const Trie& constraints) const {
        auto tracer = DMLUpdateTracer<Generator, Model>(gen, constraints);
        auto value = model_with_args.exec(tracer);
        return tracer.finish(value);
    }

    static Trie get_choice_trie(const Trie& subtraces) {
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

    void set_value(return_type value) { maybe_value_ = value; }

    [[nodiscard]] Trie get_choice_trie() const override { return get_choice_trie(*subtraces_); }

    void set_ready_for_gradients(bool ready_for_gradients) { ready_for_gradients_ = ready_for_gradients; }

    // NOTE: assumes that the return value is a Tensor and the arguments is a vector of Tensors
    std::vector<Tensor> gradients(Tensor retval_grad, double scaler) {

        std::vector<Tensor> inputs;
        for (const Tensor& input : get_tensors(args_copy_)) {
            inputs.template emplace_back(input);
        }
        // TODO add all parameters as inputs
        Tensor retval = any_cast<Tensor>(get_return_value());
        // TODO add contributions to logpdf for each call to a generative function!?
        // TODO use scaler on parameter gradients only
        std::vector<Tensor> input_grads = torch::autograd::grad({retval}, inputs, {retval_grad});
        return std::move(input_grads);
    }

private:
    args_type args_copy_; // TODO new; implement get_tensors
    shared_ptr<Trie> subtraces_;
    double score_;
    optional<return_type> maybe_value_;
    bool ready_for_gradients_ = false;
};



// *******************
// * Simulate tracer *
// *******************

template <typename Generator, typename Model>
class DMLSimulateTracer {
public:
    explicit DMLSimulateTracer(Generator& gen, bool prepare_for_gradients)
            : finished_(false), gen_{gen}, trace_{}, prepare_for_gradients_{prepare_for_gradients} { }

    template <typename CalleeType>
    typename CalleeType::return_type
    call(const Address& address, const CalleeType& gen_fn_with_args) {
        assert(!finished_); // if this assertion fails, it is a bug in DML not user code
        typename CalleeType::trace_type subtrace = gen_fn_with_args.simulate(gen_, prepare_for_gradients_);
        const auto& value = any_cast<typename CalleeType::return_type>(subtrace.get_return_value());
        trace_.add_subtrace(address, std::move(subtrace));
        return value;
    }

    DMLTrace<Model> finish(typename Model::return_type value, bool ready_for_gradients) {
        finished_ = true;
        trace_.set_value(value);
        trace_.set_ready_for_gradients(ready_for_gradients);
        return std::move(trace_);
    }

private:
    bool finished_;
    Generator& gen_;
    DMLTrace<Model> trace_;
    bool prepare_for_gradients_;
};

// *******************
// * Generate tracer *
// *******************

template <typename Generator, typename Model>
class DMLGenerateTracer {
public:
    explicit DMLGenerateTracer(Generator& gen, const Trie& constraints, bool prepare_for_gradients)
            : finished_(false), gen_{gen}, trace_{}, log_weight_(0.0), constraints_(constraints),
              prepare_for_gradients_{prepare_for_gradients} { }

    template <typename CalleeType>
    typename CalleeType::return_type
    call(const Address& address, const CalleeType& gen_fn_with_args) {
        assert(!finished_); // if this assertion fails, it is a bug in DML not user code
        Trie sub_constraints { constraints_.get_subtrie(address, false) };
        auto subtrace_and_log_weight = gen_fn_with_args.generate(gen_, sub_constraints, prepare_for_gradients_);
        typename CalleeType::trace_type subtrace = std::move(subtrace_and_log_weight.first);
        double log_weight_increment = subtrace_and_log_weight.second;
        log_weight_ += log_weight_increment;
        const auto value = any_cast<typename CalleeType::return_type>(subtrace.get_return_value());
        trace_.add_subtrace(address, std::move(subtrace));
        return value;
    }

    std::pair<DMLTrace<Model>,double> finish(typename Model::return_type value, bool ready_for_gradients) {
        finished_ = true;
        trace_.set_value(value);
        trace_.set_ready_for_gradients(ready_for_gradients);
        return std::pair(std::move(trace_), log_weight_);
    }

private:
    double log_weight_;
    bool finished_;
    Generator& gen_;
    DMLTrace<Model> trace_;
    const Trie& constraints_;
    bool prepare_for_gradients_;
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
        trace_.set_ready_for_gradients(false); // TODO add support for preparing for gradients in update
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


// ***************************
// * DML generative function *
// ***************************

template <typename Model, typename ArgsType, typename ReturnType>
class DMLGenFn {
private:
    const ArgsType args_;
public:
    typedef ArgsType args_type;
    typedef ReturnType return_type;
    typedef DMLTrace<Model> trace_type;

    explicit DMLGenFn(ArgsType args) : args_(args) {}

    args_type get_args() const { return args_; }

    template <typename Generator>
    DMLTrace<Model> simulate(Generator& gen, bool prepare_for_gradients) const {
        c10::InferenceMode guard{!prepare_for_gradients}; // inference mode is on if we are not preparing for gradients
        auto tracer = DMLSimulateTracer<Generator,Model>{gen, prepare_for_gradients};
        auto value = static_cast<const Model*>(this)->exec(tracer);
        return tracer.finish(value, prepare_for_gradients);
    }

    template <typename Generator>
    std::pair<DMLTrace<Model>,double> generate(
            Generator& gen, const Trie& constraints, bool prepare_for_gradients) const {
        c10::InferenceMode guard{!prepare_for_gradients}; // inference mode is on if we are not preparing for gradients
        auto tracer = DMLGenerateTracer<Generator,Model>{gen, constraints, prepare_for_gradients};
        auto value = static_cast<const Model*>(this)->exec(tracer);
        return tracer.finish(value, prepare_for_gradients);
    }
};

