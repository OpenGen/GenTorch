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

#ifndef GENTORCH_GEN_FN_H
#define GENTORCH_GEN_FN_H

#include <gentorch/trie.h>
#include <gentl/types.h>

using gentorch::ChoiceTrie;
using gentl::SimulateOptions, gentl::GenerateOptions, gentl::UpdateOptions;

namespace gentorch::dml {

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

    explicit DMLGenFn(ArgsType args) : args_(args) {}
    DMLGenFn(const DMLGenFn&) = default;

    const ArgsType& get_args() const {
        return args_;
    }

    template<typename RNG>
    std::unique_ptr<DMLTrace<Model>> simulate(RNG& gen, parameters_type& parameters, const SimulateOptions& options) const;

    template<typename RNG>
    std::pair<std::unique_ptr<DMLTrace<Model>>, double> generate(
            RNG& gen, parameters_type& parameters,
            const ChoiceTrie &constraints, const GenerateOptions& options) const;

    template<typename RNG>
    std::pair<return_type, double> assess(RNG& rng, parameters_type& parameters, const ChoiceTrie& constraints) const;
};

}


#endif //GENTORCH_GEN_FN_H
