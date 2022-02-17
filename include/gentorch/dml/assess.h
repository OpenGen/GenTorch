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


#ifndef GENTORCH_ASSESS_H
#define GENTORCH_ASSESS_H

namespace gentorch::dml {

template<typename RNG, typename Model>
class AssessTracer {
public:
    typedef typename Model::args_type args_type;
    typedef typename Model::parameters_type parameters_type;
    typedef typename Model::return_type return_type;

    explicit AssessTracer(RNG& rng,
                          const args_type& args,
                          parameters_type& parameters,
                          const ChoiceTrie& constraints) :
            args_{args},
            finished_{false},
            rng_{rng},
            log_weight_(0.0),
            constraints_(constraints),
            parameters_{parameters} {
    }

    const args_type& get_args() const {
        return args_;
    }

    template<typename CalleeType, typename CalleeParametersType>
    typename CalleeType::return_type
    call(Address&& address, CalleeType&& gen_fn_with_args, CalleeParametersType& parameters) {
        c10::InferenceMode guard{true};
        typedef typename CalleeType::trace_type callee_trace_type;
        typedef typename CalleeType::return_type callee_return_type;
        assert(!finished_);
        ChoiceTrie sub_constraints{constraints_.get_subtrie(address, false)};
        auto [return_value, log_weight_increment] = gen_fn_with_args.assess(rng_, parameters, sub_constraints);
        log_weight_ += log_weight_increment;
        return return_value; // copy
    }

    template<typename CalleeType>
    typename CalleeType::return_type
    call(Address&& address, CalleeType&& gen_fn_with_args) {
        c10::InferenceMode guard{true};
        return call(std::move(address), std::forward<CalleeType>(gen_fn_with_args), gentorch::empty_module_singleton);
    }

    std::pair<return_type, double> finish(return_type&& value) {
        c10::InferenceMode guard{true};
        finished_ = true;
        return {std::forward<return_type>(value), log_weight_};
    }

    parameters_type& get_parameters() { return parameters_; }

private:
    const args_type& args_;
    double log_weight_;
    bool finished_;
    RNG& rng_;
    const ChoiceTrie& constraints_;
    parameters_type& parameters_;
};

template<typename Model, typename Args, typename Return, typename Parameters>
template<typename RNG>
std::pair<typename DMLGenFn<Model, Args, Return, Parameters>::return_type, double> DMLGenFn<Model, Args, Return, Parameters>::assess(
        RNG& rng,
        Parameters& parameters,
        const ChoiceTrie& constraints) const {
    c10::InferenceMode guard{true};
    AssessTracer<RNG, Model> tracer{rng, this->get_args(), parameters, constraints};
    auto value = static_cast<const Model*>(this)->forward(tracer);
    return tracer.finish(std::move(value));
}

}


#endif //GENTORCH_ASSESS_H
