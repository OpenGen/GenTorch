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

#ifndef GENTORCH_TRACE_H
#define GENTORCH_TRACE_H

#include <gen/address.h>
#include <gen/trie.h>
#include <gen/conversions.h>
#include <gen/trace.h>
#include <gen/parameters.h>

#include <gentl/concepts.h>
#include <gentl/types.h>

#include <torch/torch.h>

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

struct SubtraceRecord {
    bool was_active;
    bool is_active;
    bool save;
    bool revert_on_restore;
    std::unique_ptr<Trace> subtrace;
    SubtraceRecord(bool active_, bool save_, std::unique_ptr<Trace>&& subtrace_) :
            was_active{false}, is_active{active_}, save{save_}, revert_on_restore{false}, subtrace{std::move(subtrace_)} {}
    SubtraceRecord(SubtraceRecord&& other) noexcept = default;
};


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

private:
    Trie<SubtraceRecord> subtraces_;

    std::unique_ptr<Model> gen_fn_with_args_;
    std::unique_ptr<Model> gen_fn_with_args_alternate_ = nullptr;
    std::unique_ptr<pair<const args_type&, std::unique_ptr<const args_type>>> args_;


    double score_ = 0.0;
    double score_alternate_;

    std::optional<return_type> maybe_value_;
    std::optional<return_type> maybe_value_alternate_;

    bool prepared_for_gradients_;
    bool can_be_reverted_{false};


    // for gradients
    std::unique_ptr<GradientHelper> helper_;
    parameters_type &parameters_;
    std::unique_ptr<Tensor> dummy_input_ = nullptr;
    std::unique_ptr<Tensor> dummy_output_ = nullptr;
    std::unique_ptr<ChoiceTrie> backward_constraints_ = nullptr;

public:

    // TODO make this private, possibly by making the tracers into nested classes
    explicit DMLTrace(const Model& gen_fn_with_args,
                      bool prepare_for_gradients, bool assert_retval_grad,
                      parameters_type &parameters) :
            gen_fn_with_args_{std::make_unique<Model>(gen_fn_with_args)},
            args_{maybe_track_args(gen_fn_with_args_->get_args(), prepare_for_gradients)},
            prepared_for_gradients_{prepare_for_gradients},
            parameters_{parameters},
            helper_{std::make_unique<GradientHelper>()} {
        assert(c10::InferenceMode::is_enabled());
        {
            // these values will only be used by gradient operations
            c10::InferenceMode guard{false};
            dummy_input_ = std::make_unique<Tensor>(torch::tensor(0.0, torch::TensorOptions().requires_grad(true)));
            dummy_output_ = std::make_unique<Tensor>(torch::tensor(0.0));
            *dummy_output_ += *dummy_input_;
        }
    }

    // TODO remove this public function after making tracers into nested classes
    Trie<SubtraceRecord>& get_subtraces() {
        return subtraces_;
    }

    DMLTrace(const DMLTrace& other) = default;
    DMLTrace(DMLTrace&& other) noexcept = default;
    DMLTrace &operator=(const DMLTrace& other) = default;
    DMLTrace &operator=(DMLTrace&& other) noexcept = default;
    ~DMLTrace() override = default;

    [[nodiscard]] double score() const override { return score_; }

    [[nodiscard]] const return_type& return_value() const {
        if (!maybe_value_.has_value()) {
            throw std::runtime_error("set_value was never called");
        }
        return maybe_value_.value();
    }

    template<typename SubtraceType>
    std::pair<SubtraceType&, double> add_subtrace(Trie<SubtraceRecord>& subtraces,
                                                  const Address& address,
                                                  std::unique_ptr<SubtraceType>&& subtrace_ptr) {
        try {
            SubtraceType* subtrace_observer_ptr = subtrace_ptr.get();
            std::unique_ptr<Trace> subtrace_base_ptr = std::move(subtrace_ptr);
            bool is_active_ = true;
            bool save_ = false;
            SubtraceRecord record{is_active_, save_, std::move(subtrace_base_ptr)};
            subtraces.set_value(address, std::move(record), false);
            return {*subtrace_observer_ptr, subtrace_observer_ptr->score()};
        } catch (const TrieOverwriteError &) {
            throw DMLAlreadyVisitedError(address);
        }
    }

    template<typename SubtraceType>
    SubtraceType& add_subtrace(const Address& address, std::unique_ptr<SubtraceType>&& subtrace_ptr) {
        auto [subtrace_ref, score_increment] = add_subtrace(
                subtraces_, address, std::forward<std::unique_ptr<SubtraceType>>(subtrace_ptr));
        score_ += score_increment;
        return subtrace_ref;
    }

    [[nodiscard]] bool has_subtrace(const Address& address) {
        return subtraces_.get_subtrie(address).has_value();
    }
    [[nodiscard]] SubtraceRecord& get_subtrace_record(const Address& address) {
        return subtraces_.get_subtrie(address).get_value();
    }


    static ChoiceTrie get_choice_trie(const Trie<SubtraceRecord>& subtraces) {
        c10::InferenceMode guard{true};
        ChoiceTrie trie{};
        // TODO handle calls at the empty address properly
        for (const auto&[key, subtrie]: subtraces.subtries()) {
            if (subtrie.has_value()) {
                const auto& record = subtrie.get_value();
                if (record.is_active)
                    trie.set_subtrie(Address{key}, record.subtrace->choices());
            } else if (subtrie.empty()) {
            } else {
                ChoiceTrie choice_subtrie{get_choice_trie(subtrie)};
                if (!choice_subtrie.empty())
                    trie.set_subtrie(Address{key}, std::move(choice_subtrie));
            }
        }
        return trie; // copy elision
    }

    void set_value(return_type value) { maybe_value_ = value; }

    [[nodiscard]] ChoiceTrie choices() const override {
        return get_choice_trie(subtraces_);
    }

    // TODO
    [[nodiscard]] ChoiceTrie choices(const gentl::selection::All &) const {
        return choices();
    }

    const args_type& get_args() const {
        return args_->first;
    }


    template <typename callee_args_type, typename callee_return_type, typename subtrace_type>
    callee_return_type make_tracked_return_value(subtrace_type &subtrace, const callee_args_type &tracked_args, const callee_return_type &value) {
        auto node = MyNode<callee_args_type, callee_return_type, subtrace_type>(subtrace, *helper_);
        c10::InferenceMode guard{false};
        // NOTE: gen_fn_with_args.get_args() returns args that are tracked as part of the autograd graph
        std::vector<Tensor> inputs = unroll(tracked_args);
        assert(dummy_input_->requires_grad());
        inputs.emplace_back(*dummy_input_);
        auto outputs = node(std::move(inputs));
        auto[num_read, tracked_value] = roll(outputs, 0, value);
        assert(num_read == outputs.size() - 1);
        Tensor dummy = outputs[num_read];
        assert(dummy.requires_grad());
        *dummy_output_ += dummy;
        assert(dummy_output_->requires_grad());
        return tracked_value;
    }

    // parameter_gradient.h
    args_type parameter_gradient(GradientAccumulator &accumulator, return_type retval_grad, double scaler);
    args_type parameter_gradient(GradientAccumulator &accumulator, double scaler);


    // update.h
    template<typename RNG>
    double update(RNG &rng, const gentl::change::UnknownChange<Model> &change,
                  const ChoiceTrie &constraints, const UpdateOptions &options);
    const ChoiceTrie &backward_constraints() const;
    void revert() override;
    void set_backward_constraints(std::unique_ptr<ChoiceTrie> &&backward_constraints);
};

}


#endif //GENTORCH_TRACE_H
