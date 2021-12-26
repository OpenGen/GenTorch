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
#include <torch/csrc/autograd/functions/utils.h>

using std::any_cast;
using std::vector;
using std::cout, std::endl;
using std::shared_ptr, std::unique_ptr;
using std::optional, std::pair;
using torch::Tensor;

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
using torch::autograd::Node;
using torch::autograd::VariableInfo;
using torch::autograd::edge_list;


// ****************************************
// * unrolling arguments into Tensor list *
// ****************************************


std::vector<Tensor> unroll(const Tensor& args) {
    return {args};
}

std::vector<Tensor> unroll(const std::vector<Tensor>& args) {
    return args;
}

std::vector<Tensor> unroll(const std::pair<Tensor, int>& args) {
    std::vector<Tensor> tensors {args.first};
    return std::move(tensors);
}

std::vector<Tensor> unroll(const std::pair<Tensor, Tensor>& args) {
    std::vector<Tensor> tensors {args.first, args.second};
    return std::move(tensors);
}

Tensor roll(const std::vector<Tensor>& rolled, const Tensor& value) {
    assert(rolled.size() == 1);
    return rolled[0];
}

std::vector<Tensor> roll(const std::vector<Tensor>& rolled, const std::vector<Tensor>& value) {
    return rolled;
}

std::pair<Tensor, int> roll(const std::vector<Tensor>& rolled, const std::pair<Tensor, int>& value) {
    assert(rolled.size() == 1);
    return {rolled[0], value.second};
}
std::pair<Tensor, Tensor> roll(const std::vector<Tensor>& rolled, const std::pair<Tensor, Tensor>& value) {
    assert(rolled.size() == 2);
    return {rolled[0], rolled[1]};
}

std::vector<Tensor> detach_clone_and_track(const std::vector<Tensor>& args) {
    std::vector<Tensor> args_copy;
    for (auto& t : args) {
        args_copy.emplace_back(t.detach().clone().set_requires_grad(true));
    }
    return args_copy; // copy elision
}

std::pair<Tensor, int> detach_clone_and_track(const std::pair<Tensor, int>& args) {
    return {args.first.detach().clone().set_requires_grad(true), args.second};
}

template <typename args_type>
pair<const args_type&, optional<shared_ptr<const args_type>>> maybe_track_args(const args_type& args,
                                                                               bool prepare_for_gradients) {
    if (prepare_for_gradients) {
        auto tracked_args_ptr = std::make_unique<const args_type>(detach_clone_and_track(args));
        return {*tracked_args_ptr, std::make_optional<shared_ptr<const args_type>>(std::move(tracked_args_ptr))};
    } else {
        return {args, std::make_optional<shared_ptr<const args_type>>()};
    }
}

// TODO add other overloaded versions of these functions for other compound data types

// *******************************************************************
// * autograd function associated with each generative function call *
// *******************************************************************

// http://blog.ezyang.com/2019/05/pytorch-internals/
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/autograd/function.h#L50
// https://discuss.pytorch.org/t/extending-autograd-from-c/76240/5

template <typename args_type, typename return_value_type>
struct MyGradNode : public Node {
    explicit MyGradNode(Trace& subtrace, const double& scaler_reference, edge_list&& next_edges)
            : Node(std::move(next_edges)), subtrace_{subtrace}, scaler_reference_{scaler_reference} { };
    ~MyGradNode() override = default;
    variable_list apply(variable_list&& unrolled_return_value_grad) override {
        variable_list unrolled_args_grad;
        auto return_value = any_cast<return_value_type>(subtrace_.get_return_value());
        return_value_type return_value_grad = roll(unrolled_return_value_grad, return_value);
        auto args_grad = any_cast<args_type>(subtrace_.gradients(return_value_grad, scaler_reference_));
        return unroll(args_grad);
    }
private:
    Trace& subtrace_;
    const double& scaler_reference_;
};

template <typename args_type, typename return_type>
struct MyNode : public Node {
    explicit MyNode(Trace& subtrace, const double& scaler_reference)
        : subtrace_{subtrace}, scaler_reference_{scaler_reference} {}
    ~MyNode() override = default;
    variable_list apply(variable_list&& inputs) override {
        std::any value_any = subtrace_.get_return_value();
        auto value = any_cast<return_type>(value_any);
        vector<Tensor> unrolled = unroll(value);
        return torch::autograd::wrap_outputs(inputs, std::move(unrolled), [&](edge_list&& next_edges) {
            return std::make_shared<MyGradNode<args_type, return_type>>(subtrace_, scaler_reference_, std::move(next_edges));
        });
    }
private:
    Trace& subtrace_;
    const double& scaler_reference_;
};

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
class DMLTrace : public Trace {
public:
    typedef typename Model::return_type return_type;
    typedef typename Model::args_type args_type;
    typedef std::pair<DMLTrace<Model>, double> update_return_type;

    explicit DMLTrace(const args_type& args, bool prepare_for_gradients, bool assert_retval_grad) :
        subtraces_{make_shared<Trie>()},
        score_{0.0},
        assert_retval_grad_{assert_retval_grad},
        args_{maybe_track_args(args, prepare_for_gradients)},
        prepared_for_gradients_{prepare_for_gradients},
        scaler_{1} { }
    DMLTrace(const DMLTrace& other) = default;
    DMLTrace(DMLTrace&& other) noexcept = default;

    DMLTrace& operator= (const DMLTrace& other) = default;
    DMLTrace& operator= (DMLTrace&& other) noexcept = default;

    ~DMLTrace() override = default;

    [[nodiscard]] double get_score() const override { return score_; }

    [[nodiscard]] std::any get_return_value() const override {
        if (!maybe_value_.has_value()) {
            throw std::runtime_error("set_value was never called");
        }
        return maybe_value_.value();
    }

    template <typename SubtraceType>
    SubtraceType& add_subtrace(const Address& address, SubtraceType subtrace) {
        score_ += subtrace.get_score();
        try {
            // TODO: fix copy
            return subtraces_->set_value(address, subtrace, false);
        } catch (const TrieOverwriteError&) {
            throw DMLAlreadyVisitedError(address);
        }
    }

    [[nodiscard]] bool has_subtrace(const Address& address) const {
        return subtraces_->get_subtrie(address).has_value();
    }

    [[nodiscard]] const Trace& get_subtrace(const Address& address) const {
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

    const args_type& get_args() const { return args_.first; }

    double& get_scaler_reference() { return scaler_; }

    std::any gradients(std::any retval_grad_any, double scaler) override {
        scaler_ = scaler; // NOTE: not threadsafe
        // TODO add all parameters as inputs; we can require users to register their torch modules for now..
        if (!prepared_for_gradients_) {
            throw std::logic_error("not ready for gradients");
        }
        return_type retval = any_cast<return_type>(get_return_value());
        return_type retval_grad = std::any_cast<return_type>(retval_grad_any);
        vector<Tensor> retval_unrolled = unroll(retval);
        vector<Tensor> retval_grad_unrolled = unroll(retval_grad);
        vector<Tensor> args_unrolled = unroll(get_args());
        vector<Tensor> args_grad_unrolled = torch::autograd::grad(retval_unrolled, args_unrolled, retval_grad_unrolled);
        args_type args_grad = roll(args_grad_unrolled, get_args());
        return args_grad; // TODO detach?
    }

private:
    pair<const args_type&, optional<shared_ptr<const args_type>>> args_;
    shared_ptr<Trie> subtraces_;
    double score_;
    optional<return_type> maybe_value_;
    bool prepared_for_gradients_;
    bool assert_retval_grad_;
    double scaler_; // for parameter gradients
};

// *******************
// * Simulate tracer *
// *******************

template <typename Generator, typename Model>
class DMLSimulateTracer {
public:
    typedef typename Model::args_type args_type;

    explicit DMLSimulateTracer(Generator& gen, const args_type& args,
                               bool prepare_for_gradients, bool assert_retval_grad) :
            finished_(false),
            gen_{gen},
            trace_{args, prepare_for_gradients, assert_retval_grad},
            prepare_for_gradients_{prepare_for_gradients},
            scaler_reference_{trace_.get_scaler_reference()} {}

    const args_type& get_args() const { return trace_.get_args(); }

    template <typename CalleeType>
    typename CalleeType::return_type
    call(Address&& address, CalleeType&& gen_fn_with_args) {
        assert(!finished_); // if this assertion fails, it is a bug in DML not user code
        Trace& subtrace = trace_.add_subtrace(address, gen_fn_with_args.simulate(gen_, prepare_for_gradients_));
        const auto& value = any_cast<typename CalleeType::return_type>(subtrace.get_return_value());
        auto node = MyNode<typename CalleeType::args_type, typename CalleeType::return_type>(subtrace, scaler_reference_);
        // NOTE: gen_fn_with_args.get_args() returns args that are tracked as part of the autograd graph
        auto tracked_value_unrolled = node(unroll(gen_fn_with_args.get_args()));
        typename CalleeType::return_type tracked_value = roll(tracked_value_unrolled, value);
        return tracked_value;
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
    bool prepare_for_gradients_;
    double& scaler_reference_;
};

// *******************
// * Generate tracer *
// *******************

template <typename Generator, typename Model>
class DMLGenerateTracer {
public:
    typedef typename Model::args_type args_type;

    explicit DMLGenerateTracer(Generator& gen, const args_type& args, const Trie& constraints,
                               bool prepare_for_gradients, bool assert_retval_grad) :
            finished_(false),
            gen_{gen},
            trace_{args, prepare_for_gradients, assert_retval_grad},
            log_weight_(0.0),
            constraints_(constraints),
            prepare_for_gradients_{prepare_for_gradients},
            scaler_reference_{trace_.get_scaler_reference()} {}

    const args_type& get_args() const { return trace_.get_args(); }

    template <typename CalleeType>
    typename CalleeType::return_type
    call(Address&& address, CalleeType&& gen_fn_with_args) {
        assert(!finished_); // if this assertion fails, it is a bug in DML not user code
        Trie sub_constraints { constraints_.get_subtrie(address, false) };
        auto subtrace_and_log_weight = gen_fn_with_args.generate(gen_, sub_constraints, prepare_for_gradients_);
        Trace& subtrace = trace_.add_subtrace(address, std::move(subtrace_and_log_weight.first));
        const auto& value = any_cast<typename CalleeType::return_type>(subtrace.get_return_value());
        auto node = MyNode<typename CalleeType::args_type, typename CalleeType::return_type>(subtrace, scaler_reference_);
        // NOTE: gen_fn_with_args.get_args() returns args that are tracked as part of the autograd graph
        auto tracked_value_unrolled = node(unroll(gen_fn_with_args.get_args()));
        typename CalleeType::return_type tracked_value = roll(tracked_value_unrolled, value);
        return tracked_value;
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
    bool prepare_for_gradients_;
    double& scaler_reference_;
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
    call(Address&& address, CalleeType&& gen_fn_with_args) {
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


// ***************************
// * DML generative function *
// ***************************

template <typename Model, typename ArgsType, typename ReturnType>
class DMLGenFn {
private:
    const ArgsType args_;
    const bool assert_retval_grad_;
public:
    typedef ArgsType args_type;
    typedef ReturnType return_type;
    typedef DMLTrace<Model> trace_type;

    explicit DMLGenFn(ArgsType args, bool assert_retval_grad=false) : args_(args), assert_retval_grad_(assert_retval_grad) {}

    const ArgsType& get_args() const {
        return args_;
    }

    template <typename Generator>
    DMLTrace<Model> simulate(Generator& gen, bool prepare_for_gradients) const {
        c10::InferenceMode guard{!prepare_for_gradients}; // inference mode is on if we are not preparing for gradients
        auto tracer = DMLSimulateTracer<Generator,Model>{gen, args_, prepare_for_gradients, assert_retval_grad_};
        auto value = static_cast<const Model*>(this)->exec(tracer);
        return tracer.finish(value);
    }

    template <typename Generator>
    std::pair<DMLTrace<Model>,double> generate(
            Generator& gen, const Trie& constraints, bool prepare_for_gradients) const {
        c10::InferenceMode guard{!prepare_for_gradients}; // inference mode is on if we are not preparing for gradients
        auto tracer = DMLGenerateTracer<Generator,Model>{gen, args_, constraints, prepare_for_gradients, assert_retval_grad_};
        auto value = static_cast<const Model*>(this)->exec(tracer);
        return tracer.finish(value);
    }
};

