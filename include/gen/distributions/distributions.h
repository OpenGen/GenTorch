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

namespace gen::distributions {

template<typename GenFnType>
class PrimitiveTrace : public Trace {
public:
    typedef typename GenFnType::return_type return_type;
    typedef typename GenFnType::args_type args_type;
    typedef typename GenFnType::dist_type dist_type;

    PrimitiveTrace(return_type &&value, const dist_type &dist) : value_{value}, dist_{dist}, score_{0.0} {}

    [[nodiscard]] const return_type& get_return_value() const {
        return value_;
    }

    // TODO make gradients not use std::any
    [[nodiscard]] std::any gradients(std::any ret_grad, double scaler, GradientAccumulator& accumulator) override {
        auto grads = dist_.log_density_gradient(value_);
        return GenFnType::extract_argument_gradient(grads);
    }

    [[nodiscard]] double get_score() const override {
        return score_;
    }

    [[nodiscard]] ChoiceTrie get_choice_trie() const override {
        ChoiceTrie trie{};
        trie.set_value(value_);
        return trie; // copy elision
    }

private:
    const return_type value_;
    const dist_type dist_;
    double score_;
};

template <typename Derived, typename ArgsType, typename ReturnType, typename DistType>
class PrimitiveGenFn {
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

    PrimitiveGenFn(args_type args, dist_type dist) : args_tracked_{args}, dist_{dist} {}

    [[nodiscard]] args_type get_args() const {
        return args_tracked_;
    }

    template<class RNGType>
    trace_type simulate(RNGType &rng, const EmptyModule& parameters,
                         bool prepare_for_gradients=false) const {
        return_type value = dist_.sample(rng);
        return {std::move(value), dist_};
    }

    template<class RNGType>
    std::pair<trace_type, double>
    generate(RNGType &rng, const EmptyModule& parameters, const ChoiceTrie& constraints,
             bool prepare_for_gradients=false) const {
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
        return {trace_type(std::move(value), dist_), log_weight};
    }


private:
    args_type args_tracked_;
    const dist_type dist_;

};

}