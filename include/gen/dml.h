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

#include <gen/address.h>
#include <gen/trie.h>
#include <gen/trace.h>

#include <any>
#include <memory>
#include <optional>

#include <torch/torch.h>

using std::any_cast;
using std::shared_ptr;
using std::optional;
using torch::Tensor;

// *************
// * DML Trace *
// *************

class DMLAlreadyVisitedError : public std::exception {
public:
    explicit  DMLAlreadyVisitedError(const Address& address) : msg_ {
            [&](){
                std::stringstream ss;
                ss << "Address already visited: " << address;
                return ss.str(); }()
    } { }
    [[nodiscard]] const char* what() const noexcept override {
        return msg_.c_str();
    }
private:
    const std::string msg_;
};

template <typename Generator, typename Model>
class DMLUpdateTracer;

template <typename Model>
class DMLTrace : Trace {
public:
    typedef typename Model::return_type return_type;
    typedef typename Model::args_type args_type;
    typedef std::pair<DMLTrace<Model>, double> update_return_type;

    DMLTrace() : subtraces_{make_shared<Trie>()}, score_{0.0} { }
    DMLTrace(const DMLTrace& other) = default;
    DMLTrace(DMLTrace&& other) noexcept = default;

    DMLTrace& operator= (const DMLTrace& other) = default;
    DMLTrace& operator= (DMLTrace&& other) noexcept = default;

    ~DMLTrace() = default;

    [[nodiscard]] double get_score() const override { return score_; }

    [[nodiscard]] std::any get_return_value() const override;

    template <typename SubtraceType>
    void add_subtrace(const Address& address, SubtraceType subtrace);

    [[nodiscard]] bool has_subtrace(const Address& address) const;

    [[nodiscard]] const Trace& get_subtrace(const Address& address) const;

    void set_value(return_type value) { maybe_value_ = value; }

    template <typename Generator>
    update_return_type update(Generator& gen, const Model& model_with_args, const Trie& constraints) const;

    static Trie get_choice_trie(const Trie& subtraces);

    [[nodiscard]] Trie get_choice_trie() const override { return get_choice_trie(*subtraces_); }

//    std::vector<torch::Tensor> gradients(torch::Tensor retval_grad, double scaler) {
//        for (const auto& grad : torch::autograd::grad({loss}, net_parameters)) {
//            parameter_grads_to_accumulate[param_idx++].add_(grad);
//        }
//    }

private:
    shared_ptr<Trie> subtraces_;
    double score_;
    optional<return_type> maybe_value_;
};


template <typename Model, typename ArgsType, typename ReturnType>
class DMLGenFn {
private:
    const ArgsType args_;
public:
    typedef ArgsType args_type;
    typedef ReturnType return_type;
    typedef DMLTrace<Model> trace_type;

    explicit DMLGenFn(ArgsType args) : args_(args) {}

    args_type get_args() const { return args_; }

    // TODO add option to not record computation graph for simulate, generate, and update

    template <typename Generator>
    trace_type simulate(Generator& gen) const;

    template <typename Generator>
    std::pair<trace_type,double> generate(Generator& gen, const Trie& constraints) const;
};