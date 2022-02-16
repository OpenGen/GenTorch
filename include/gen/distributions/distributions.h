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

#include <gen/trace.h>
#include <random>

#include <torch/torch.h>

#include <gentl/concepts.h>
#include <gentl/types.h>

using gentl::SimulateOptions;
using gentl::GenerateOptions;
using gentl::UpdateOptions;

namespace gen::distributions {

template<typename GenFnType>
class PrimitiveTrace : public Trace {
public:
    typedef typename GenFnType::return_type return_type;
    typedef typename GenFnType::args_type args_type;
    typedef typename GenFnType::dist_type dist_type;

    PrimitiveTrace(return_type &&value, const dist_type &dist) :
            value_{std::make_unique<return_type>(value)},
            value_alternate_{std::make_unique<return_type>(value)},
            dist_{std::make_unique<dist_type>(dist)},
            dist_alternate_{std::make_unique<dist_type>(dist)},
            score_{0.0} {}

    [[nodiscard]] const return_type& return_value() const {
        return *value_;
    }

    [[nodiscard]] args_type parameter_gradient(
            GradientAccumulator& accumulator, const return_type&, double scaler) {
        return parameter_gradient(accumulator, scaler);
    }

    [[nodiscard]] args_type parameter_gradient(GradientAccumulator& accumulator, double scaler) {
        auto grads = dist_->log_density_gradient(*value_);
        return GenFnType::extract_argument_gradient(grads);
    }

    [[nodiscard]] double score() const override {
        return score_;
    }

    [[nodiscard]] ChoiceTrie choices() const override {
        ChoiceTrie trie{};
        trie.set_value(*value_);
        return trie; // copy elision
    }

    template<typename RNG>
    double update(RNG& rng, const gentl::change::UnknownChange<GenFnType>& change,
                  const ChoiceTrie& constraints, const UpdateOptions& options) {
        const GenFnType& new_gen_fn = change.new_value();
        double log_weight;
        const return_type& prev_value = *value_.get();
        double prev_score = score_;
        if (options.save()) {
            std::swap(value_, value_alternate_);
            std::swap(dist_, dist_alternate_);
            std::swap(score_, score_alternate_);
            can_be_reverted_ = true;
        }
        if (constraints.has_value()) {
            backward_constraints_.set_value(prev_value);
            *value_ = std::any_cast<return_type>(constraints.get_value());
            *dist_ = new_gen_fn.dist_;
            score_ = dist_->log_density(*value_);
            log_weight = score_ - prev_score;
        } else if (constraints.empty()) {
            log_weight = 0.0;
            backward_constraints_.clear();
        } else {
            throw std::domain_error("expected primitive or empty choice dict");
        }
        has_backward_constraints_ = true;
        return log_weight;
    }

    [[nodiscard]] const ChoiceTrie& backward_constraints() const {
        if (!has_backward_constraints_)
            throw std::logic_error("there are no backward constraints; call update");
        return backward_constraints_;
    }

    void revert() override {
        if (!can_be_reverted_)
            throw std::logic_error("cannot be reverted; call update with save=true");
        std::swap(value_, value_alternate_);
        std::swap(dist_, dist_alternate_);
        std::swap(score_, score_alternate_);
        can_be_reverted_ = false;
        has_backward_constraints_ = false;
    }


private:
    std::unique_ptr<return_type> value_;
    std::unique_ptr<return_type> value_alternate_;
    std::unique_ptr<dist_type> dist_;
    std::unique_ptr<dist_type> dist_alternate_;
    ChoiceTrie backward_constraints_;
    double score_;
    double score_alternate_;
    bool can_be_reverted_{false};
    bool has_backward_constraints_{false};
};

template <typename Derived, typename ArgsType, typename ReturnType, typename DistType>
class PrimitiveGenFn {
    friend class PrimitiveTrace<Derived>;
public:

    // for generative function interface
    typedef ArgsType args_type;
    typedef ReturnType return_type;
    typedef DistType dist_type;
    typedef PrimitiveTrace<Derived> trace_type;

    // for more concise user constructor
    typedef Derived M;
    typedef args_type A;
    typedef return_type R;
    typedef dist_type D;

    PrimitiveGenFn(args_type args, dist_type dist) : args_tracked_{args}, dist_{dist} {
        // NOTE: if the user is constructing us within the body of a DML generative function, then
        // we need inference mode to be disabled here. If there user is constructing us in their inference program
        // then we would typically want inference mode to be enabled. For now, we trust that users remember
        // to set InferenceMode in their inference (and learning) program.
    }

    // TODO where is this needed?
    [[nodiscard]] args_type get_args() const {
        return args_tracked_;
    }

    template<class RNGType>
    std::unique_ptr<trace_type> simulate(RNGType &rng, const EmptyModule& parameters,
                                         const SimulateOptions&) const {
        return_type value = dist_.sample(rng);
        return std::make_unique<trace_type>(std::move(value), dist_);
    }

    template<class RNGType>
    std::pair<std::unique_ptr<trace_type>, double>
    generate(RNGType &rng, const EmptyModule& parameters, const ChoiceTrie& constraints,
             const GenerateOptions&) const {
        return_type value;
        double log_weight;
        if (constraints.has_value()) {
            value = std::any_cast<return_type>(constraints.get_value());
            log_weight = dist_.log_density(value);
        } else if (constraints.empty()) {
            value = dist_.sample(rng);
            log_weight = 0.0;
        } else {
            throw std::domain_error("expected primitive or empty choice dict");
        }
        return {std::make_unique<trace_type>(std::move(value), dist_), log_weight};
    }


private:
    args_type args_tracked_;
    const dist_type dist_;

};

}