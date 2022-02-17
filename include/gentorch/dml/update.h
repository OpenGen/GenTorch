/* Copyright 2021-2022 Massachusetts Institute of Technology

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

#ifndef GENTORCH_UPDATE_H
#define GENTORCH_UPDATE_H

namespace gentorch::dml {

void update_pre_visit(Trie<SubtraceRecord>& trie);
[[nodiscard]] bool has_subtrace(const Trie<SubtraceRecord>& trie, const Address& address);
[[nodiscard]] SubtraceRecord& get_subtrace_record(const Trie<SubtraceRecord>& trie, const Address& address);
double update_post_visit(Trie<SubtraceRecord>& trie, bool save, ChoiceTrie& backward_constraints);
void revert_visit(Trie<SubtraceRecord>& trie);

template<typename RNG, typename Model>
class DMLUpdateTracer {
    typedef typename Model::args_type args_type;
    typedef DMLTrace<Model> trace_type;
    typedef typename Model::parameters_type parameters_type;

    double log_weight_;
    bool finished_;
    RNG& rng_;
    trace_type& trace_;
    const ChoiceTrie& constraints_;
    bool prepare_for_gradients_;
    parameters_type& parameters_;
    std::unique_ptr<ChoiceTrie> backward_constraints_;
    bool save_;

public:

    explicit DMLUpdateTracer(RNG& rng,
                             parameters_type& parameters,
                             const ChoiceTrie& constraints,
                             trace_type& trace,
                             bool prepare_for_gradients, bool save) :
            finished_(false),
            rng_{rng},
            trace_{trace},
            log_weight_(0.0),
            constraints_(constraints),
            prepare_for_gradients_{prepare_for_gradients},
            parameters_{parameters},
            save_{save},
            backward_constraints_{std::move(std::make_unique<ChoiceTrie>())} {
        assert(!(prepare_for_gradients && c10::InferenceMode::is_enabled()));
    }

    const args_type& get_args() const { return trace_.get_args(); }

    template<typename CalleeType, typename CalleeParametersType>
    typename CalleeType::return_type call(Address&& address, CalleeType&& gen_fn_with_args,
                                          CalleeParametersType& parameters) {
        typedef typename CalleeType::args_type callee_args_type;
        typedef typename CalleeType::trace_type callee_trace_type;
        typedef typename CalleeType::return_type callee_return_type;
        assert(!finished_); // if this assertion fails, it is a bug in DML not user code
        ChoiceTrie sub_constraints{constraints_.get_subtrie(address, false)};

        callee_trace_type* subtrace = nullptr;
        if (has_subtrace(trace_.get_subtraces(), address)) {
            auto& record = get_subtrace_record(trace_.get_subtraces(), address);
            subtrace = static_cast<callee_trace_type*>(record.subtrace.get());
            record.is_active = true;
            switch (record.state) {
                case RecordState::Active:
                    // ACTIVE ----- U -----> ACTIVE
                    // ACTIVE ----- US ----> ACTIVE_SAVED
                    record.state = (save_ ? RecordState::ActiveSaved : RecordState::Active);
                    break;
                case RecordState::ActiveSaved:
                    // ACTIVE_SAVED ----- U ----> ACTIVE_SAVED
                    // ACTIVE_SAVED ----- US ---> ACTIVE_SAVED
                    break;
                case RecordState::InactiveSaved:
                    // INACTIVE_SAVED --- UR ---> ACTIVE_SAVED
                    // INACTIVE_SAVED --- URS --> ACTIVE
                    record.state = (save_ ? RecordState::Active : RecordState::ActiveSaved);
                    break;
                case RecordState::InactiveSavedRevert:
                    // INACTIVE_SAVED_REVERT --- UR ---> ACTIVE_SAVED
                    // INACTIVE_SAVED_REVERT --- URS --> ACTIVE
                    record.state = (save_ ? RecordState::Active : RecordState::ActiveSaved);
                    break;
            }
            log_weight_ += subtrace->update(
                    rng_, gentl::change::UnknownChange(gen_fn_with_args), sub_constraints,
                    UpdateOptions().precompute_gradient(prepare_for_gradients_)
                                   .save((record.was_active && save_) || !record.was_active)
                                   .ignore_previous_choices(!record.was_active));
            // TODO avoid copy of subtrace_backward by modifying set_subtrie interface to accept unique_ptr
            const ChoiceTrie& subtrace_backward = subtrace->backward_constraints();
            if (!subtrace_backward.empty())
                backward_constraints_->set_subtrie(address, subtrace_backward);
        } else {
            auto [subtrace_ptr, log_weight_increment] = gen_fn_with_args.generate(
                    rng_, parameters, sub_constraints, GenerateOptions().precompute_gradient(prepare_for_gradients_));
            log_weight_ += log_weight_increment;
            // This sets RecordState::Active, not RecordState::ActiveSaved, even if save=true
            subtrace = &trace_.add_subtrace(address, std::move(subtrace_ptr));
        }


        const callee_return_type &value = subtrace->return_value();
        if (prepare_for_gradients_) {
            return trace_.make_tracked_return_value(*subtrace, gen_fn_with_args.get_args(), value);
        } else {
            return value; // copy
        }
    }

    template<typename CalleeType>
    typename CalleeType::return_type call(Address &&address, CalleeType &&gen_fn_with_args) {
        return call(std::move(address), std::forward<CalleeType>(gen_fn_with_args), gentorch::empty_module_singleton);
    }

    double finish(typename Model::return_type value) {
        c10::InferenceMode guard{true};
        log_weight_ -= update_post_visit(trace_.get_subtraces(), save_, *backward_constraints_);
        finished_ = true;
        trace_.set_value(value);
        trace_.set_backward_constraints(std::move(backward_constraints_));
        return log_weight_;
    }

    parameters_type &get_parameters() { return parameters_; }

    bool prepare_for_gradients() const { return prepare_for_gradients_; }
};

template <typename Model>
template <typename RNG>
double DMLTrace<Model>::update(
        RNG& rng,
        const gentl::change::UnknownChange<Model>& change,
        const ChoiceTrie& constraints,
        const UpdateOptions& options) {
    c10::InferenceMode guard{true};

    // the code that runs during forward() always modifies the non-alternate values
    if (options.save()) {
        std::swap(gen_fn_with_args_, gen_fn_with_args_alternate_);
        std::swap(score_, score_alternate_);
        std::swap(maybe_value_, maybe_value_alternate_);
        can_be_reverted_ = true;
    }

    // calls copy constructor of arguments type (TODO optimize away this copy?)
    gen_fn_with_args_ = std::make_unique<Model>(change.new_value());
    args_ = maybe_track_args(gen_fn_with_args_->get_args(), options.precompute_gradient());

    DMLUpdateTracer<RNG, Model> tracer{
            rng,
            parameters_,
            constraints,
            *this,
            options.precompute_gradient(),
            options.save()};
    update_pre_visit(subtraces_);
    {
        c10::InferenceMode inner_guard{!options.precompute_gradient()};
        auto value = gen_fn_with_args_->forward(tracer);
        return tracer.finish(value);
    }
}

template <typename Model>
template <typename RNG>
double DMLTrace<Model>::update(
        RNG& rng,
        const gentl::change::NoChange& change,
        const ChoiceTrie& constraints,
        const UpdateOptions& options) {
    // TODO this copies the gen_fn_with_args unecessarily
    return update(rng, gentl::change::UnknownChange<Model>(*gen_fn_with_args_), constraints, options);
}

template <typename Model>
void DMLTrace<Model>::revert() {
    if (!can_be_reverted_)
        throw std::logic_error("trace can not be reverted");
    std::swap(gen_fn_with_args_, gen_fn_with_args_alternate_);
    std::swap(score_, score_alternate_);
    std::swap(maybe_value_, maybe_value_alternate_);
    revert_visit(subtraces_);
    can_be_reverted_ = false;
    prepared_for_gradients_ = false; // TODO support caching precomputed autograd graph
}

template <typename Model>
void DMLTrace<Model>::set_backward_constraints(std::unique_ptr<ChoiceTrie> &&backward_constraints) {
    backward_constraints_ = std::move(backward_constraints);
}

template <typename Model>
const ChoiceTrie& DMLTrace<Model>::backward_constraints() const {
    return *backward_constraints_;
}

template <typename Model>
std::unique_ptr<DMLTrace<Model>> DMLTrace<Model>::fork() {
    // calls the private copy constructor
    return std::unique_ptr<DMLTrace<Model>>(new DMLTrace<Model>(*this));
}

}


#endif //GENTORCH_UPDATE_H
