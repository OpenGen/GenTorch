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
#include <gen/parameters.h>

#include <any>
#include <memory>
#include <optional>

#include <torch/torch.h>
#include <torch/csrc/autograd/functions/utils.h>

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

// *******************************************************************
// * autograd function associated with each generative function call *
// *******************************************************************

// http://blog.ezyang.com/2019/05/pytorch-internals/
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/autograd/function.h#L50
// https://discuss.pytorch.org/t/extending-autograd-from-c/76240/5

struct GradientHelper {
    double* scaler_ptr_ {nullptr};
    GradientAccumulator* accumulator_ptr_ {nullptr};
};

template<typename args_type, typename return_value_type, typename subtrace_type>
struct MyGradNode : public Node {
    explicit MyGradNode(subtrace_type& subtrace, const GradientHelper& helper, edge_list &&next_edges)
            : Node(std::move(next_edges)), subtrace_{subtrace},
              helper_{helper} {};

    ~MyGradNode() override = default;

    variable_list apply(variable_list &&output_grad) override {
        const return_value_type& return_value = subtrace_.get_return_value();

        // read off the return value gradient from the output_grad
        auto [num_read, return_value_grad] = roll(output_grad, 0, return_value);
        // check that we read off all but the last element, which is the dummy element
        assert(num_read == output_grad.size() - 1);

        auto args_grad = any_cast<args_type>(subtrace_.gradients(
                return_value_grad, *helper_.scaler_ptr_, *helper_.accumulator_ptr_));
        std::vector<Tensor> input_grad = unroll(args_grad);
        input_grad.emplace_back(torch::tensor(0.0));
        return input_grad;
    }

private:
    subtrace_type& subtrace_;
    const GradientHelper& helper_;
};

template<typename args_type, typename return_type, typename subtrace_type>
struct MyNode : public Node {
    explicit MyNode(subtrace_type& subtrace, const GradientHelper& helper)
            : subtrace_{subtrace}, helper_{helper} {}

    ~MyNode() override = default;

    variable_list apply(variable_list &&inputs) override {
        const return_type& value = subtrace_.get_return_value();
        vector<Tensor> output = unroll(value);

        // value that we use to ensure the corresponding MyGradNode is visited in the backward pass
        Tensor dummy = torch::tensor(0.0); // TODO optimize me, maybe make static?
        output.emplace_back(dummy);

        return torch::autograd::wrap_outputs(inputs, std::move(output), [&](edge_list &&next_edges) {
            return std::make_shared<MyGradNode<args_type, return_type, subtrace_type>>(subtrace_, helper_, std::move(next_edges));
        });
    }

private:
    subtrace_type& subtrace_;
    const GradientHelper& helper_;
};


// *************
// * DML Trace *
// *************

class DMLAlreadyVisitedError : public std::exception {
public:
    explicit DMLAlreadyVisitedError(const Address &address) : msg_{
            [&]() {
                std::stringstream ss;
                ss << "Address already visited: " << address;
                return ss.str();
            }()
    } {}

    [[nodiscard]] const char *what() const noexcept override {
        return msg_.c_str();
    }

private:
    const std::string msg_;
};

template<typename T>
T detach_clone_and_track(const T &args) {
    assert(!c10::InferenceMode::is_enabled());
    vector<Tensor> args_unrolled = unroll(args);
    vector<Tensor> args_unrolled_copy;
    for (auto &t: args_unrolled) {
        args_unrolled_copy.emplace_back(t.detach().clone().set_requires_grad(true));
    }
    return roll(args_unrolled_copy, args); // copy elision
}

template<typename args_type>
pair<const args_type &, unique_ptr<const args_type>> maybe_track_args(const args_type &args,
                                                                                bool prepare_for_gradients) {
    if (prepare_for_gradients) {
        auto tracked_args_ptr = make_unique<const args_type>(detach_clone_and_track(args));
        return {*tracked_args_ptr, std::move(tracked_args_ptr)};
    } else {
        return {args, nullptr};
    }
}


/**
 * NOTE: not copy-constructible!
 *
 * @tparam Model
 */
template<typename Model>
class DMLTrace : public Trace {
public:
    typedef typename Model::return_type return_type;
    typedef typename Model::args_type args_type;
    typedef typename Model::parameters_type parameters_type;
    typedef std::pair<DMLTrace<Model>, double> update_return_type;

    explicit DMLTrace(const args_type &args, bool prepare_for_gradients, bool assert_retval_grad,
                      parameters_type& parameters) :
            subtraces_{std::make_unique<Trie<unique_ptr<Trace>>>()},
            score_{0.0},
            assert_retval_grad_{assert_retval_grad},
            args_{maybe_track_args(args, prepare_for_gradients)},
            prepared_for_gradients_{prepare_for_gradients},
            parameters_{parameters},
            helper_{std::make_unique<GradientHelper>()},
            dummy_input_{torch::tensor(0.0, torch::TensorOptions().requires_grad(prepare_for_gradients))},
            dummy_output_{torch::tensor(0.0)} {
        if (prepare_for_gradients) {
            dummy_output_ += dummy_input_;
        }
    }

    DMLTrace(const DMLTrace &other) = default;

    DMLTrace(DMLTrace &&other) noexcept = default;

    DMLTrace &operator=(const DMLTrace &other) = default;

    DMLTrace &operator=(DMLTrace &&other) noexcept = default;

    ~DMLTrace() override = default;

    [[nodiscard]] double get_score() const override { return score_; }

    [[nodiscard]] const return_type& get_return_value() const {
        if (!maybe_value_.has_value()) {
            throw std::runtime_error("set_value was never called");
        }
        return maybe_value_.value();
    }

    template<typename SubtraceType>
    SubtraceType& add_subtrace(const Address &address, SubtraceType&& subtrace) {
        score_ += subtrace.get_score();
        try {
            std::unique_ptr<SubtraceType> subtrace_ptr = std::make_unique<SubtraceType>(std::move(subtrace));
            SubtraceType* subtrace_observer_ptr = subtrace_ptr.get();
            std::unique_ptr<Trace> subtrace_base_ptr = std::move(subtrace_ptr);
            subtraces_->set_value(address, std::move(subtrace_base_ptr), false);
            return *subtrace_observer_ptr;
        } catch (const TrieOverwriteError &) {
            throw DMLAlreadyVisitedError(address);
        }
    }

    [[nodiscard]] bool has_subtrace(const Address &address) const {
        return subtraces_->get_subtrie(address).has_value();
    }

    [[nodiscard]] const Trace& get_subtrace(const Address &address) const {
        return *subtraces_->get_value(address);
    }

    static ChoiceTrie get_choice_trie(const Trie<unique_ptr<Trace>> &subtraces) {
        c10::InferenceMode guard {true};
        ChoiceTrie trie{};
        // TODO handle calls at the empty address properly
        for (const auto&[key, subtrie]: subtraces.subtries()) {
            if (subtrie.has_value()) {
                trie.set_subtrie(Address{key}, subtrie.get_value()->get_choice_trie());
            } else if (subtrie.empty()) {
            } else {
                trie.set_subtrie(Address{key}, get_choice_trie(subtrie));
            }
        }
        return trie; // copy elision
    }

    void set_value(return_type value) { maybe_value_ = value; }

    [[nodiscard]] ChoiceTrie get_choice_trie() const override { return get_choice_trie(*subtraces_); }

    const args_type &get_args() const { return args_.first; }

    // TODO: change retval_grad to use retval_type, and similarly for gradients_args

    any gradients(any retval_grad_any, double scaler, GradientAccumulator& accumulator) override {

        // the computation graph is already set up, there should be no Tensors created anyways
        c10::InferenceMode guard {true};

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
        const return_type& retval = get_return_value();
        return_type retval_grad = std::any_cast<return_type>(retval_grad_any);
        vector<Tensor> outputs = unroll(retval);
        vector<Tensor> output_grads = unroll(retval_grad);

        // add the dummy output
        assert(dummy_output_.requires_grad());
        outputs.emplace_back(dummy_output_);
        output_grads.emplace_back(torch::tensor(1.0));

        // pass information back to the calls to gradients() that happen within the backward pass will have
        // NOTE: not thread safe, but this is okay, traces aren't intended to be thread safe
        helper_->scaler_ptr_ = &scaler;
        helper_->accumulator_ptr_ = &accumulator;

        // do the backward pass. this recursively invokes gradients() for each subtrace
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

        return args_grad;
    }

    template <typename args_type, typename return_type, typename subtrace_type>
    return_type make_tracked_return_value(subtrace_type& subtrace, const args_type& tracked_args, const return_type& value) {
        auto node = MyNode<args_type, return_type, subtrace_type>(subtrace, *helper_);
        // NOTE: gen_fn_with_args.get_args() returns args that are tracked as part of the autograd graph
        std::vector<Tensor> inputs = unroll(tracked_args);
        assert(dummy_input_.requires_grad());
        inputs.emplace_back(dummy_input_);
        auto outputs = node(std::move(inputs));
        auto [num_read, tracked_value] = roll(outputs, 0, value);
        assert(num_read == outputs.size() - 1);
        Tensor dummy = outputs[num_read];
        assert(dummy.requires_grad());
        dummy_output_ += dummy;
        assert(dummy_output_.requires_grad());
        return tracked_value;
    }

private:
    pair<const args_type&, std::unique_ptr<const args_type>> args_;
    std::unique_ptr<Trie<std::unique_ptr<Trace>>> subtraces_;
    double score_;
    optional<return_type> maybe_value_;
    bool prepared_for_gradients_;
    bool assert_retval_grad_;
    std::unique_ptr<GradientHelper> helper_;
    parameters_type& parameters_;
    Tensor dummy_input_;
    Tensor dummy_output_;
};





// *******************
// * Simulate tracer *
// *******************

template<typename Generator, typename Model>
class DMLSimulateTracer {
public:
    typedef typename Model::args_type args_type;
    typedef typename Model::parameters_type parameters_type;

    explicit DMLSimulateTracer(Generator &gen, const args_type &args,
                               parameters_type& parameters,
                               bool prepare_for_gradients, bool assert_retval_grad) :
            finished_(false),
            gen_{gen},
            trace_{args, prepare_for_gradients, assert_retval_grad, parameters},
            prepare_for_gradients_{prepare_for_gradients},
            parameters_{parameters} {
        assert(!(prepare_for_gradients && c10::InferenceMode::is_enabled()));
    }

    const args_type &get_args() const { return trace_.get_args(); }

    template<typename CalleeType, typename CalleeParametersType>
    typename CalleeType::return_type
    call(Address &&address, CalleeType &&gen_fn_with_args, CalleeParametersType& parameters) {
        typedef typename CalleeType::args_type callee_args_type;
        typedef typename CalleeType::trace_type callee_trace_type;
        typedef typename CalleeType::return_type callee_return_type;
        assert(!finished_); // if this assertion fails, it is a bug in DML not user code
        callee_trace_type& subtrace = trace_.add_subtrace(address,
                                              gen_fn_with_args.simulate(gen_, parameters, prepare_for_gradients_));
        const auto& value = subtrace.get_return_value();
        if (prepare_for_gradients_) {
            return trace_.make_tracked_return_value(subtrace, gen_fn_with_args.get_args(), value);
        } else {
            return value; // copy the value
        }
    }

    template<typename CalleeType>
    typename CalleeType::return_type
    call(Address&& address, CalleeType&& gen_fn_with_args) {
        return call(std::move(address), std::move(gen_fn_with_args), gen::empty_module_singleton);
    }

    DMLTrace<Model> finish(typename Model::return_type value) {
        assert(!(prepare_for_gradients_ && c10::InferenceMode::is_enabled()));
        finished_ = true;
        trace_.set_value(value);
        return std::move(trace_);
    }

    // for use in the body of the exec() function
    parameters_type& get_parameters() { return parameters_; }

    bool prepare_for_gradients() const { return prepare_for_gradients_; }

private:
    bool finished_;
    Generator &gen_;
    DMLTrace<Model> trace_;
    bool prepare_for_gradients_;
    parameters_type& parameters_;
};

// *******************
// * Generate tracer *
// *******************

template<typename Generator, typename Model>
class DMLGenerateTracer {
public:
    typedef typename Model::args_type args_type;
    typedef typename Model::parameters_type parameters_type;

    explicit DMLGenerateTracer(Generator &gen, const args_type &args, parameters_type& parameters,
                               const ChoiceTrie &constraints, bool prepare_for_gradients, bool assert_retval_grad) :
            finished_(false),
            gen_{gen},
            trace_{args, prepare_for_gradients, assert_retval_grad, parameters},
            log_weight_(0.0),
            constraints_(constraints),
            prepare_for_gradients_{prepare_for_gradients},
            parameters_{parameters} {
        assert(!(prepare_for_gradients && c10::InferenceMode::is_enabled()));
    }

    const args_type &get_args() const { return trace_.get_args(); }

    template<typename CalleeType, typename CalleeParametersType>
    typename CalleeType::return_type
    call(Address &&address, CalleeType &&gen_fn_with_args, CalleeParametersType& parameters) {
        typedef typename CalleeType::args_type callee_args_type;
        typedef typename CalleeType::trace_type callee_trace_type;
        typedef typename CalleeType::return_type callee_return_type;
        assert(!finished_); // if this assertion fails, it is a bug in DML not user code
        ChoiceTrie sub_constraints{constraints_.get_subtrie(address, false)};
        auto subtrace_and_log_weight = gen_fn_with_args.generate(gen_, parameters, sub_constraints, prepare_for_gradients_);
        callee_trace_type& subtrace = trace_.add_subtrace(address, std::move(subtrace_and_log_weight.first));
        log_weight_ += subtrace_and_log_weight.second;
        const callee_return_type& value = subtrace.get_return_value();
        if (prepare_for_gradients_) {
            return trace_.make_tracked_return_value(subtrace, gen_fn_with_args.get_args(), value);
        } else {
            return value; // copy
        }
    }

    template<typename CalleeType>
    typename CalleeType::return_type
    call(Address&& address, CalleeType&& gen_fn_with_args) {
        return call(std::move(address), std::move(gen_fn_with_args), gen::empty_module_singleton);
    }

    std::pair<DMLTrace<Model>, double> finish(typename Model::return_type value) {
        assert(!(prepare_for_gradients_ && c10::InferenceMode::is_enabled()));
        finished_ = true;
        trace_.set_value(value);
        return std::pair(std::move(trace_), log_weight_);
    }

    parameters_type& get_parameters() { return parameters_; }

    bool prepare_for_gradients() const { return prepare_for_gradients_; }

private:
    double log_weight_;
    bool finished_;
    Generator &gen_;
    DMLTrace<Model> trace_;
    const ChoiceTrie &constraints_;
    bool prepare_for_gradients_;
    parameters_type& parameters_;
};

// ***************************
// * DML generative function *
// ***************************

/**
 * Abstract type for a generative function constructed with the Dynamic Modeling Language (DML), which is embedded in C++.
 *
 * Each DML generative function is a concrete type that inherits from `DMLGenFn` and has an `exec` member function.
 *
 * @tparam Model Type of the generative function, which must inherit from `DMLGenFn` via the CRTP.
 * @tparam ArgsType Type of the input to the generative function.
 * @tparam ReturnType Type of the return value of hte generative function.
 */
template<typename Model, typename ArgsType, typename ReturnType, typename ParametersType>
class DMLGenFn {
private:
    const ArgsType args_;
    const bool assert_retval_grad_;
public:

    // part of the generative function interface
    typedef ArgsType args_type;
    typedef ReturnType return_type;
    typedef ParametersType parameters_type;
    typedef DMLTrace<Model> trace_type;

    // for use in call to base class constructor DMLGenFn<M,A,R,P>(..)
    typedef Model M;
    typedef args_type A;
    typedef return_type R;
    typedef parameters_type P;

    explicit DMLGenFn(ArgsType args, bool assert_retval_grad = false)
        : args_(args), assert_retval_grad_(assert_retval_grad) {
        // NOTE: if the user is constructing us within the body of a DML generative function, then
        // we need inference mode to be disabled here. If there user is constructing us in their inference program
        // then we would typically want inference mode to be enabled. For now, we trust that users remember
        // to set InferenceMode in their inference (and learning) program.

        // NOTE: the inference library can include checks that you are in inference mode
    }

    const ArgsType &get_args() const {
        return args_;
    }

    template<typename Generator>
    DMLTrace<Model> simulate(Generator &gen, parameters_type& parameters, bool prepare_for_gradients) {
        c10::InferenceMode guard{
                !prepare_for_gradients}; // inference mode is on if we are not preparing for gradients
        auto tracer = DMLSimulateTracer<Generator, Model>{gen, args_, parameters, prepare_for_gradients,
                                                          assert_retval_grad_};
        auto value = static_cast<Model*>(this)->forward(tracer);
        return tracer.finish(value);
    }

    template<typename Generator>
    std::pair<DMLTrace<Model>, double> generate(
            Generator &gen, parameters_type& parameters,
            const ChoiceTrie &constraints, bool prepare_for_gradients) {
        c10::InferenceMode guard{
                !prepare_for_gradients}; // inference mode is on if we are not preparing for gradients
        auto tracer = DMLGenerateTracer<Generator, Model>{gen, args_, parameters, constraints, prepare_for_gradients,
                                                          assert_retval_grad_};
        auto value = static_cast<Model*>(this)->forward(tracer);
        return tracer.finish(value);
    }
};

}
