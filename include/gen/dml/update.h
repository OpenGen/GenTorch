#ifndef GENTORCH_UPDATE_H
#define GENTORCH_UPDATE_H

namespace gen::dml {

void update_pre_visit(Trie<SubtraceRecord>& trie, bool save) {
    // recursively walk the trie and mark all existing records
    for (auto& [key, subtrie] : trie.subtries()) {
        if (subtrie.has_value()) {
            auto record = subtrie.get_value();
            // if save, then all inactive records will be un-saved and all active records will be saved
            // otherwise, we do not change the save status of records
            if (save)
                record.save = record.is_active;
            record.was_active = record.is_active;
            record.is_active = false;
        } else {
            assert(!subtrie.empty());
            update_pre_visit(subtrie, save);
        }
    }
}

[[nodiscard]] bool has_subtrace(const Trie<SubtraceRecord>& trie, const Address& address) {
    return trie.get_subtrie(address).has_value();
}

[[nodiscard]] SubtraceRecord& get_subtrace_record(const Trie<SubtraceRecord>& trie, const Address& address) {
    return trie.get_subtrie(address).get_value();
}

double update_post_visit(Trie<SubtraceRecord>& trie, bool save, ChoiceTrie& backward_constraints) {
    double unvisited_score = 0.0;
    // recursively walk the trie
    auto& subtries_map = trie.subtries();
    for (auto it = subtries_map.begin(); it != subtries_map.end();) {
        auto& [address, subtrie] = *it;
        assert(!subtrie.empty());
        if (subtrie.has_value()) {
            auto record = subtrie.get_value();

            // if the subtrace was_active, but is not now active, then it contributes to backward_constraints
            const ChoiceTrie& backward_constraints_subtrie = record.subtrace->choices();
            if (!backward_constraints_subtrie.empty())
                backward_constraints.set_subtrie(address, backward_constraints_subtrie);
            unvisited_score += record.subtrace->score();

            if (!record.is_active && !record.save) {

                // destruct records that are inactive and not marked as saved
                it = subtries_map.erase(it);
            } else {

                // records that were already inactive and saved, and are not re-saved, should be reverted on restore
                if (!save && record.was_active && !record.is_active) {
                    assert(record.save);
                    record.revert_on_restore = true;
                }
                it++;
            }
        } else {
            ChoiceTrie backward_constraints_subtrie;
            unvisited_score += update_post_visit(subtrie, save, backward_constraints_subtrie);
            backward_constraints.set_subtrie(address, std::move(backward_subtrie));
        }
    }
    return unvisited_score;
}

void revert_visit(Trie<SubtraceRecord>& trie) {
    auto& subtries = trie.subtries();
    for (auto it = subtries.begin(); it != subtries.end();) {
        auto& [address, subtrie] = *it;
        if (subtrie.has_value()) {
            auto record = subtrie.get_value();
            if (record.save) {
                // if it was already active it should be reverted
                // if it was not already active, then it may need to be reverted, depending on revert_on_restore
                if (record.is_active || record.revert_on_restore)
                    record.subtrace->revert();

                // anything that is marked as save should be marked as active
                record.is_active = true;
            }

            if (!record.save) {

                // anything that is not marked as saved should be destructed
                it = subtries.erase(it);
            } else {

                // anything that was marked as save will be unmarked
                record.save = false;

                // reset
                record.revert_on_restore = false;

                // continue
                it++;
            }
        } else {
            assert(!subtrie.empty());
            revert_visit(subtrie);
        }
    }
}


template<typename RNG, typename Model>
class DMLUpdateTracer {
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
    typedef typename Model::args_type args_type;
    typedef typename Model::parameters_type parameters_type;
    typedef DMLTrace<Model> trace_type;

    explicit DMLUpdateTracer(RNG& rng,
                             parameters_type& parameters,
                             const ChoiceTrie& constraints,
                             trace_type& trace,
                             bool prepare_for_gradients, bool assert_retval_grad, bool save) :
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

    const args_type& get_args() const { return trace_->get_args(); }

    template<typename CalleeType, typename CalleeParametersType>
    typename CalleeType::return_type
    call(Address&& address, CalleeType&& gen_fn_with_args, CalleeParametersType& parameters) {
        typedef typename CalleeType::args_type callee_args_type;
        typedef typename CalleeType::trace_type callee_trace_type;
        typedef typename CalleeType::return_type callee_return_type;
        assert(!finished_); // if this assertion fails, it is a bug in DML not user code
        ChoiceTrie sub_constraints{constraints_.get_subtrie(address, false)};

        callee_trace_type* subtrace = nullptr;
        if (has_subtrace(trace_->subtraces_, address)) {
            auto& record = get_subtrace_record(trace_->subtraces_, address);
                // do an in-place generate update call
            log_weight += record.subtrace->update(
                    rng_, gentl::change::UnknownChange(gen_fn_with_args), sub_constraints,
                    UpdateOptions().precompute_gradient(prepare_for_gradients_)
                                   .save(save_)
                                   .ignore_previous_choices(!record.was_active));
            record.is_active = true;
            // TODO avoid copy of subtrace_backward by modifying set_subtrie interface to accept unique_ptr
            const ChoiceTrie& subtrace_backward = subtrace->backward_constraints();
            backward_constraints_->set_subtrie(address, subtrace_backward);
        } else {
            auto [subtrace_ptr, log_weight_increment] = gen_fn_with_args.generate(
                    rng_, parameters, sub_constraints, GenerateOptions().precompute_gradient(prepare_for_gradients_));
            log_weight_ += log_weight_increment;
            subtrace = &add_subtrace(address, subrace_ptr);
        }


        const callee_return_type &value = subtrace->return_value();
        if (prepare_for_gradients_) {
            return trace_->make_tracked_return_value(*subtrace, gen_fn_with_args.get_args(), value);
        } else {
            return value; // copy
        }
    }

    template<typename CalleeType>
    typename CalleeType::return_type
    call(Address &&address, CalleeType &&gen_fn_with_args) {
        return call(std::move(address), std::forward<CalleeType>(gen_fn_with_args), gen::empty_module_singleton);
    }

    std::pair <std::unique_ptr<DMLTrace < Model>>, double>

    finish(typename Model::return_type value) {
        assert(!(prepare_for_gradients_ && c10::InferenceMode::is_enabled()));
        log_weight_ -= update_post_visit(trace_.subtraces_, _save, *backward_constraints_);
        finished_ = true;
        trace_->set_value(value);
        trace_->set_backward_constraints(std::move(backward_constraints_));
        return log_weight_;
    }

    parameters_type &get_parameters() { return parameters_; }

    bool prepare_for_gradients() const { return prepare_for_gradients_; }
};

template <typename Model>
template <typename RNG, typename NewModel>
double DMLTrace<Model>::update(
        RNG& rng,
        const gentl::change::UnknownChange<Model>& change,
        const ChoiceTrie& constraints,
        const UpdateOptions& options) {

    c10::InferenceMode guard{
            !options.precompute_gradient()}; // inference mode is on if we are not preparing for gradients

    // the code that runs during forward() always modifies the non-alternate values
    if (options.save()) {
        std::swap(gen_fn_with_args_, gen_fn_with_args_alternate_);
        std::swap(score_, score_alternate_);
        std::swap(maybe_value_, maybe_value_alternate_);
        can_be_reverted_ = true;
    }

    // calls copy constructor (TODO optimize away this copy?)
    gen_fn_with_args_ = std::make_unique<Model>(change.new_value());

    DMLUpdateTracer<RNG, Model> tracer{
            rng,
            parameters_,
            constraints,
            *this,
            options.precompute_gradient(),
            false, options.save()};
    auto value = gen_fn_with_args_.forward(tracer);
    return tracer.finish(value);
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

}


#endif //GENTORCH_UPDATE_H
