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

#include <any>

namespace gen {

//    class Parameters : public torch::nn::Module {
//    public:
//        virtual vector<shared_ptr<Parameters>> callee_modules() { return {}; }
//
//        /**
//         *
//         * @param module a torch::nn::Module, that must have a name, and must implement callee_modules()
//         * @param visited
//         */
//        Parameters(shared_ptr<torch::nn::Module> module, std::set<std::string>& visited) : module_{module}, grad_accumulator_{}, children_{} {
//            visited.insert(module->name());
//            for (const auto& t: module->parameters(true)) {
//                grad_accumulator_.emplace_back(torch::zeros_like(t).detach());
//            }
//            for (const auto& callee_module : module->callee_modules()) {
//                if (visited.find(callee_module->name()) == visited.end()) {
//                    children_.emplace(std::make_pair(callee_module->name(), Parameter(callee_module)));
//                }
//            }
//        }
//
//        static Parameters make_parameters(shared_ptr<torch::nn::Module> module) {
//            std::set<std::string> visited {};
//            return Parameters(module, visited);
//        }
//
//        /**
//         *
//         * @return the torch::nn::Module that contains all the modules directly called by the body
//         */
//        torch::nn::Module& module() { return *module_; }
//
//        // TODO improve upon this approach
//        Parameters& get_callee_module(const std::string& name) { return *children_[name]; }
//
//        void incorporate(std::set<std::string>& visited) {
//            visited.insert(module_->name());
//            size_t i = 0;
//            for (auto& t: module_->parameters(true)) {
//                Tensor& accum = grad_accumulator_[i++];
//                t.mutable_grad().add_(accum);
//                accum.zero_();
//            }
//            for (auto& child : children_) {
//                child.second->incorporate(visited);
//            }
//        }
//
//    private:
//        shared_ptr<torch::nn::Module> module_;
//        unordered_map<std::string, unique_ptr<Parameters>> children_;
//        vector<Tensor> grad_accumulator_;
//    };

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

        virtual std::any gradients(std::any retval_grad, double scaler) = 0;

    };

}