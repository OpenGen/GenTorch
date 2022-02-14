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

#ifndef GENTORCH_DML_H
#define GENTORCH_DML_H

namespace gen::dml {

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

//template<typename RNG, typename Model>
//class DMLUpdateTracer;

}

#include <gen/dml/autodiff.h>
#include <gen/dml/gen_fn.h>
#include <gen/dml/trace.h>
#include <gen/dml/simulate.h>
#include <gen/dml/generate.h>
//#include <gen/dml/update.h>
#include <gen/dml/parameter_gradient.h>
//#include <gen/dml/choice_gradient.h>

#endif // GENTORCH_DML_H