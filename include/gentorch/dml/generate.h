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

#ifndef GENTORCH_GENERATE_H
#define GENTORCH_GENERATE_H

namespace gentorch::dml {


template<typename RNG, typename Model>
class DMLGenerateTracer {
public:
    typedef typename Model::args_type args_type;
    typedef typename Model::parameters_type parameters_type;
    using trace_type = DMLTrace<Model>;

    explicit DMLGenerateTracer(RNG& rng,
                               const Model& gen_fn_with_args,
                               parameters_type& parameters,
                               const ChoiceTrie& constraints,
                               bool prepare_for_gradients) :
            finished_(false),
            rng_{rng},
            trace_{std::make_unique<trace_type>(gen_fn_with_args, prepare_for_gradients, parameters)},
            log_weight_(0.0),
            constraints_(constraints),
            prepare_for_gradients_{prepare_for_gradients},
            parameters_{parameters} {
    }

    const args_type& get_args() const {
        return trace_->get_args();
    }

    template<typename CalleeType, typename CalleeParametersType>
    typename CalleeType::return_type
    call(Address&& address, CalleeType&& gen_fn_with_args, CalleeParametersType &parameters) {
        c10::InferenceMode guard{true};
        typedef typename CalleeType::trace_type callee_trace_type;
        typedef typename CalleeType::return_type callee_return_type;
        assert(!finished_);
        ChoiceTrie sub_constraints{constraints_.get_subtrie(address, false)};
        auto [subtrace_ptr, log_weight_increment] = gen_fn_with_args.generate(
                rng_, parameters, sub_constraints, GenerateOptions().precompute_gradient(prepare_for_gradients_));
        callee_trace_type& subtrace = trace_->add_subtrace(address, std::move(subtrace_ptr));
        log_weight_ += log_weight_increment;
        const callee_return_type &value = subtrace.return_value();
        if (prepare_for_gradients_) {
            return trace_->make_tracked_return_value(subtrace, gen_fn_with_args.get_args(), value);
        } else {
            return value; // copy
        }
    }

    template<typename CalleeType>
    typename CalleeType::return_type
    call(Address&& address, CalleeType&& gen_fn_with_args) {
        return call(std::move(address), std::forward<CalleeType>(gen_fn_with_args), gentorch::empty_module_singleton);
    }

    std::pair<std::unique_ptr<DMLTrace < Model>>, double>

    finish(typename Model::return_type value) {
        c10::InferenceMode guard{true};
        finished_ = true;
        trace_->set_value(value);
        return std::pair(std::move(trace_), log_weight_);
    }

    parameters_type& get_parameters() { return parameters_; }

    bool prepare_for_gradients() const { return prepare_for_gradients_; }

private:
    double log_weight_;
    bool finished_;
    RNG& rng_;
    std::unique_ptr<DMLTrace<Model>> trace_;
    const ChoiceTrie& constraints_;
    bool prepare_for_gradients_;
    parameters_type& parameters_;
};


template<typename Model, typename Args, typename Return, typename Parameters>
template<typename RNG>
std::pair<std::unique_ptr<DMLTrace<Model>>,double> DMLGenFn<Model, Args, Return, Parameters>::generate(
        RNG& rng,
        Parameters& parameters,
        const ChoiceTrie& constraints,
        const GenerateOptions& options) const {
    c10::InferenceMode guard{true};
    const auto& derived = *static_cast<const Model*>(this);
    DMLGenerateTracer<RNG, Model> tracer{rng, derived, parameters, constraints, options.precompute_gradient()};
    {
        c10::InferenceMode inner_guard{!options.precompute_gradient()};
        auto value = static_cast<const Model*>(this)->forward(tracer);
        return tracer.finish(value);
    }
}

}


#endif //GENTORCH_GENERATE_H
