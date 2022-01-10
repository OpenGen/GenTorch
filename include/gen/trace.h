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

#include <gen/trie.h>
#include <gen/parameters.h>

#include <any>

namespace gen {

    /**
     * Immutable execution of a generative function.
     *
     * Constructed with `simulate` and `generate` member functions of generative functions.
     */
    class Trace {
    public:

        virtual ~Trace() = default;

        /**
         * @return the choice trie.
         */
        [[nodiscard]] virtual Trie<std::any> get_choice_trie() const = 0;

        /**
         * @return the log joint density.
         */
        [[nodiscard]] virtual double get_score() const = 0;

        /**
         * @return the return value of the generative function.
         */
        [[nodiscard]] virtual std::any get_return_value() const = 0;

        virtual std::any gradients(std::any retval_grad, double scaler, GradientAccumulator& accumulator) = 0;

    };

}