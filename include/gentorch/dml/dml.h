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

#ifndef GENTORCH_DML_H
#define GENTORCH_DML_H

namespace gentorch::dml {

// TODO rename 'Model' to 'Program'

// forward declarations

template <typename Model, typename Args, typename Return, typename Parameters>
class DMLGenFn;

template <typename Model>
class DMLTrace;

template<typename RNG, typename Model>
class DMLSimulateTracer;

template<typename RNG, typename Model>
class DMLGenerateTracer;

template<typename RNG, typename Model>
class DMLAssessTracer;

template<typename RNG, typename Model>
class DMLUpdateTracer;

}

#include <gentorch/dml/autodiff.h>
#include <gentorch/dml/gen_fn.h>
#include <gentorch/dml/trace.h>
#include <gentorch/dml/simulate.h>
#include <gentorch/dml/generate.h>
#include <gentorch/dml/assess.h>
#include <gentorch/dml/update.h>
#include <gentorch/dml/parameter_gradient.h>
//#include <gentorch/dml/choice_gradient.h>

#endif // GENTORCH_DML_H
