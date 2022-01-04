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

namespace gen::dml {

    // *******************************************************************
    // * autograd function associated with each generative function call *
    // *******************************************************************

    // http://blog.ezyang.com/2019/05/pytorch-internals/
    // https://github.com/pytorch/pytorch/blob/master/torch/csrc/autograd/function.h#L50
    // https://discuss.pytorch.org/t/extending-autograd-from-c/76240/5

    template<typename args_type, typename return_value_type>
    struct MyGradNode : public Node {
        explicit MyGradNode(Trace &subtrace, const double &scaler_reference, edge_list &&next_edges)
                : Node(std::move(next_edges)), subtrace_{subtrace}, scaler_reference_{scaler_reference} {};

        ~MyGradNode() override = default;

        variable_list apply(variable_list &&unrolled_return_value_grad) override {
            variable_list unrolled_args_grad;
            auto return_value = any_cast<return_value_type>(subtrace_.get_return_value());
            return_value_type return_value_grad = roll(unrolled_return_value_grad, return_value);
            auto args_grad = any_cast<args_type>(subtrace_.gradients(return_value_grad, scaler_reference_));
            std::cout << "inside MyGradNode" << std::endl;
            return unroll(args_grad);
        }

    private:
        Trace &subtrace_;
        const double &scaler_reference_;
    };

    template<typename args_type, typename return_type>
    struct MyNode : public Node {
        explicit MyNode(Trace &subtrace, const double &scaler_reference)
                : subtrace_{subtrace}, scaler_reference_{scaler_reference} {}

        ~MyNode() override = default;

        variable_list apply(variable_list &&inputs) override {
            std::any value_any = subtrace_.get_return_value();
            auto value = any_cast<return_type>(value_any);
            vector<Tensor> unrolled = unroll(value);
            return torch::autograd::wrap_outputs(inputs, std::move(unrolled), [&](edge_list &&next_edges) {
                return std::make_shared<MyGradNode<args_type, return_type>>(subtrace_, scaler_reference_,
                                                                            std::move(next_edges));
            });
        }

    private:
        Trace &subtrace_;
        const double &scaler_reference_;
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
        vector<Tensor> args_unrolled = unroll(args);
        vector<Tensor> args_unrolled_copy;
        for (auto &t: args_unrolled) {
            args_unrolled_copy.emplace_back(t.detach().clone().set_requires_grad(true));
        }
        return roll(args_unrolled_copy, args); // copy elision
    }

    template<typename args_type>
    pair<const args_type &, optional<shared_ptr<const args_type>>> maybe_track_args(const args_type &args,
                                                                                    bool prepare_for_gradients) {
        if (prepare_for_gradients) {
            auto tracked_args_ptr = make_unique<const args_type>(detach_clone_and_track(args));
            return {*tracked_args_ptr, make_optional<shared_ptr<const args_type>>(move(tracked_args_ptr))};
        } else {
            return {args, make_optional<shared_ptr<const args_type>>()};
        }
    }

    template<typename Generator, typename Model>
    class DMLUpdateTracer;

    template<typename Model>
    class DMLTrace : public Trace {
    public:
        typedef typename Model::return_type return_type;
        typedef typename Model::args_type args_type;
        typedef typename Model::parameters_type parameters_type;
        typedef std::pair<DMLTrace<Model>, double> update_return_type;

        explicit DMLTrace(const args_type &args, bool prepare_for_gradients, bool assert_retval_grad,
                          const parameters_type& parameters) :
                subtraces_{make_shared<Trie<shared_ptr<Trace>>>()},
                score_{0.0},
                assert_retval_grad_{assert_retval_grad},
                args_{maybe_track_args(args, prepare_for_gradients)},
                prepared_for_gradients_{prepare_for_gradients},
                scaler_{1},
                parameters_{parameters} {}

        DMLTrace(const DMLTrace &other) = default;

        DMLTrace(DMLTrace &&other) noexcept = default;

        DMLTrace &operator=(const DMLTrace &other) = default;

        DMLTrace &operator=(DMLTrace &&other) noexcept = default;

        ~DMLTrace() override = default;

        [[nodiscard]] double get_score() const override { return score_; }

        [[nodiscard]] std::any get_return_value() const override {
            if (!maybe_value_.has_value()) {
                throw std::runtime_error("set_value was never called");
            }
            return maybe_value_.value();
        }

        template<typename SubtraceType>
        SubtraceType &add_subtrace(const Address &address, SubtraceType subtrace) {
            score_ += subtrace.get_score();
            try {
                shared_ptr<SubtraceType> subtrace_ptr = make_shared<SubtraceType>(std::move(subtrace));
                shared_ptr<Trace> subtrace_base_ptr = subtrace_ptr;
                subtraces_->set_value(address, subtrace_base_ptr, false);
                return *subtrace_ptr;
            } catch (const TrieOverwriteError &) {
                throw DMLAlreadyVisitedError(address);
            }
        }

        [[nodiscard]] bool has_subtrace(const Address &address) const {
            return subtraces_->get_subtrie(address).has_value();
        }

        [[nodiscard]] const Trace &get_subtrace(const Address &address) const {
            return *any_cast<shared_ptr<Trace>>(subtraces_->get_value(address));
        }

        template<typename Generator>
        update_return_type update(Generator &gen, const Model &model_with_args,
                                  const Trie<shared_ptr<Trace>> &constraints) const {
            auto tracer = DMLUpdateTracer<Generator, Model>(gen, constraints);
            auto value = model_with_args.exec(tracer);
            return tracer.finish(value);
        }

        static ChoiceTrie get_choice_trie(const Trie<shared_ptr<Trace>> &subtraces) {
            ChoiceTrie trie{};
            // TODO handle calls at the empty address properly
            for (const auto&[key, subtrie]: subtraces.subtries()) {
                if (subtrie.has_value()) {
                    const auto subtrace = subtrie.get_value();
                    trie.set_subtrie(Address{key}, subtrace->get_choice_trie());
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

        double &get_scaler_reference() { return scaler_; }

        any gradients(any retval_grad_any, double scaler) override {

            if (!prepared_for_gradients_) {
                throw std::logic_error("not ready for gradients");
            }

            // set the scaler_ so that recursive calls to gradients() will pass along this value
            // NOTE: not thread safe, but this is okay, traces aren't intended to be thread safe
            scaler_ = scaler;

            return_type retval = any_cast<return_type>(get_return_value());
            return_type retval_grad = std::any_cast<return_type>(retval_grad_any);
            vector<Tensor> retval_unrolled = unroll(retval);
            vector<Tensor> retval_grad_unrolled = unroll(retval_grad);

            vector<Tensor> inputs = unroll(get_args());
            size_t num_args = inputs.size();

            // this means all recursive parameters of children modules, but not any children generative functions..
            vector<Tensor> local_parameter_tensors = parameters_.parameters();
            size_t num_local_parameters = local_parameter_tensors.size();

            // combine arguments and gradients
            inputs.insert(inputs.end(), local_parameter_tensors.begin(), local_parameter_tensors.end());

            // do the backward pass. this recursively invokes gradients() for each subtrace
            vector<Tensor> input_grads = torch::autograd::grad(retval_unrolled, inputs, retval_grad_unrolled);

            // split arguments and gradients
            // TODO pass start and end pointers recursively to roll calls instead of allocating a new vector here
            vector<Tensor> args_grad_unrolled;
            for (size_t i = 0; i < num_args; ++i) {
                args_grad_unrolled.emplace_back(input_grads[i]);
            }
            args_type args_grad = roll(args_grad_unrolled, get_args());

            // accumulate parameter gradients
            for (size_t i = num_args; i < num_args + num_local_parameters; ++i) {
                local_parameter_tensors[i].add_(input_grads[i], scaler);
            }

            return args_grad; // TODO detach?
        }

    private:
        // TODO why are we using shared_pointers here? we don't necessarily need Traces to be
        // TODO copy-constructible...
        pair<const args_type &, optional<shared_ptr<const args_type>>> args_;
        shared_ptr<Trie<shared_ptr<Trace>>> subtraces_;
        double score_;
        optional<return_type> maybe_value_;
        bool prepared_for_gradients_;
        bool assert_retval_grad_;
        double scaler_; // for parameter gradients
        const parameters_type& parameters_;
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
                                   const parameters_type& parameters,
                                   bool prepare_for_gradients, bool assert_retval_grad) :
                finished_(false),
                gen_{gen},
                trace_{args, prepare_for_gradients, assert_retval_grad, parameters},
                prepare_for_gradients_{prepare_for_gradients},
                scaler_reference_{trace_.get_scaler_reference()},
                parameters_{parameters} {}

        const args_type &get_args() const { return trace_.get_args(); }

        template<typename CalleeType, typename CalleeParametersType = std::nullptr_t>
        typename CalleeType::return_type
        call(Address &&address, CalleeType &&gen_fn_with_args, CalleeParametersType& parameters = nullptr) {
            assert(!finished_); // if this assertion fails, it is a bug in DML not user code
            Trace &subtrace = trace_.add_subtrace(address,
                                                  gen_fn_with_args.simulate(gen_, parameters, prepare_for_gradients_));
            const auto &value = any_cast<typename CalleeType::return_type>(subtrace.get_return_value());

            // TODO we also want to record something in the this MyNode object that will let us, in gradients(), set the
            // TODO appropriate sub-parameters object for use by the corresponding MyGradNode...
            // instead of passing the parameter object here, the user would need to pass a function that extracts
            // it from the parent object, and this function would need to work with the accumulator object as well..

            // we could use a second Trie object, that stores mutable references to parameters objects,
            //
            auto node = MyNode<typename CalleeType::args_type, typename CalleeType::return_type>(
                    subtrace, scaler_reference_);

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

        // for use in the body of the exec() function
        const parameters_type& get_parameters() { return parameters_; }

    private:
        bool finished_;
        Generator &gen_;
        DMLTrace<Model> trace_;
        bool prepare_for_gradients_;
        double &scaler_reference_;
        const parameters_type& parameters_;
    };

    // *******************
    // * Generate tracer *
    // *******************

    template<typename Generator, typename Model>
    class DMLGenerateTracer {
    public:
        typedef typename Model::args_type args_type;
        typedef typename Model::parameters_type parameters_type;

        explicit DMLGenerateTracer(Generator &gen, const args_type &args, const ChoiceTrie &constraints,
                                   const parameters_type& parameters,
                                   bool prepare_for_gradients, bool assert_retval_grad) :
                finished_(false),
                gen_{gen},
                trace_{args, prepare_for_gradients, assert_retval_grad, parameters},
                log_weight_(0.0),
                constraints_(constraints),
                prepare_for_gradients_{prepare_for_gradients},
                scaler_reference_{trace_.get_scaler_reference()},
                parameters_{parameters} {}

        const args_type &get_args() const { return trace_.get_args(); }

        template<typename CalleeType, typename CalleeParametersType = std::nullptr_t>
        typename CalleeType::return_type
        call(Address &&address, CalleeType &&gen_fn_with_args, CalleeParametersType& parameters = nullptr) {
            assert(!finished_); // if this assertion fails, it is a bug in DML not user code
            ChoiceTrie sub_constraints{constraints_.get_subtrie(address, false)};
            auto subtrace_and_log_weight = gen_fn_with_args.generate(gen_, parameters, sub_constraints, prepare_for_gradients_);
            Trace &subtrace = trace_.add_subtrace(address, std::move(subtrace_and_log_weight.first));
            const auto &value = any_cast<typename CalleeType::return_type>(subtrace.get_return_value());
            auto node = MyNode<typename CalleeType::args_type, typename CalleeType::return_type>(subtrace,
                                                                                                 scaler_reference_);
            // NOTE: gen_fn_with_args.get_args() returns args that are tracked as part of the autograd graph
            auto tracked_value_unrolled = node(unroll(gen_fn_with_args.get_args()));
            typename CalleeType::return_type tracked_value = roll(tracked_value_unrolled, value);
            return tracked_value;
        }

        std::pair<DMLTrace<Model>, double> finish(typename Model::return_type value) {
            finished_ = true;
            trace_.set_value(value);
            return std::pair(std::move(trace_), log_weight_);
        }

        const parameters_type& get_parameters() { return parameters_; }

    private:
        double log_weight_;
        bool finished_;
        Generator &gen_;
        DMLTrace<Model> trace_;
        const ChoiceTrie &constraints_;
        bool prepare_for_gradients_;
        double &scaler_reference_;
        const parameters_type& parameters_;
    };

    // *******************
    // * Update tracer *
    // *******************

    // TODO add integration with parameter store
    template<typename Generator, typename Model>
    class DMLUpdateTracer {
    public:
        explicit DMLUpdateTracer(Generator &gen, const ChoiceTrie &constraints)
                : finished_(false), gen_{gen}, trace_{},
                  log_weight_(0.0), constraints_(constraints) {}

        // TODO add third argument for parameter store for calllee (see simulate above)
        template<typename CalleeType>
        typename CalleeType::return_type
        call(Address &&address, CalleeType &&gen_fn_with_args) {
            assert(!finished_); // if this assertion fails, it is a bug in DML not user code
            ChoiceTrie sub_constraints{constraints_.get_subtrie(address, false)};
            typename CalleeType::trace_type subtrace;
            if (prev_trace_.has_subtrace(address)) {
                auto &prev_subtrace = *any_cast<unique_ptr<Trace>>(prev_trace_.get_subtrace(address));
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

        std::tuple<DMLTrace<Model>, double, ChoiceTrie> finish(typename Model::return_type value) {
            log_weight_ += 0; // TODO decrement for all visited
            // TODO discard all that were not visied (using update method of Trie, which still needs to be implemented)
            finished_ = true;
            trace_.set_value(value);
            return std::tuple(std::move(trace_), log_weight_, discard_);
        }

    private:
        double log_weight_;
        bool finished_;
        Generator &gen_;
        const DMLTrace<Model> &prev_trace_;
        DMLTrace<Model> trace_;
        const ChoiceTrie &constraints_;
        ChoiceTrie discard_;
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
    template<typename Model, typename ArgsType, typename ReturnType, typename ModuleType>
    class DMLGenFn {
    private:
        const ArgsType args_;
        const bool assert_retval_grad_;
    public:
        typedef ArgsType args_type;
        typedef ReturnType return_type;
        typedef ModuleType module_type;
        typedef Parameters<ModuleType> parameters_type;
        typedef DMLTrace<Model> trace_type;

        explicit DMLGenFn(ArgsType args, bool assert_retval_grad = false)
            : args_(args), assert_retval_grad_(assert_retval_grad) {}

        const ArgsType &get_args() const {
            return args_;
        }

        template<typename Generator>
        DMLTrace<Model> simulate(Generator &gen, const parameters_type& parameters, bool prepare_for_gradients) const {
            c10::InferenceMode guard{
                    !prepare_for_gradients}; // inference mode is on if we are not preparing for gradients
            auto tracer = DMLSimulateTracer<Generator, Model>{gen, args_, parameters, prepare_for_gradients,
                                                              assert_retval_grad_};
            auto value = static_cast<const Model *>(this)->exec(tracer);
            return tracer.finish(value);
        }

        template<typename Generator>
        std::pair<DMLTrace<Model>, double> generate(
                Generator &gen, const parameters_type& parameters,
                const ChoiceTrie &constraints, bool prepare_for_gradients) const {
            c10::InferenceMode guard{
                    !prepare_for_gradients}; // inference mode is on if we are not preparing for gradients
            auto tracer = DMLGenerateTracer<Generator, Model>{gen, args_, parameters, constraints, prepare_for_gradients,
                                                              assert_retval_grad_};
            auto value = static_cast<const Model *>(this)->exec(tracer);
            return tracer.finish(value);
        }
    };

}
