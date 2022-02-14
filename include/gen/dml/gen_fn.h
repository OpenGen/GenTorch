//
// Created by marcoct on 2/13/22.
//

#ifndef GENTORCH_GEN_FN_H
#define GENTORCH_GEN_FN_H

#include <gen/trie.h>
#include <gentl/concepts.h>

using gen::ChoiceTrie;
using gentl::SimulateOptions, gentl::GenerateOptions, gentl::UpdateOptions;

namespace gen::dml {


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

    template<typename RNG>
    std::unique_ptr<DMLTrace<Model>> simulate(RNG& gen, parameters_type& parameters, const SimulateOptions& options);

    template<typename RNG>
    std::pair<std::unique_ptr<DMLTrace<Model>>, double> generate(
            RNG& gen, parameters_type& parameters,
            const ChoiceTrie &constraints, const GenerateOptions& options);
};

}


#endif //GENTORCH_GEN_FN_H
