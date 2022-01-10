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

namespace gen::distributions::normal {

using torch::Tensor;
using torch::tensor;
using gen::ChoiceTrie;

// ******************
// *** NormalDist ***
// ******************

constexpr double pi() { return 3.141592653589793238462643383279502884; }

class NormalDist {
public:

    NormalDist(Tensor mean, Tensor std);

    template<class Generator>
    Tensor sample(Generator &gen) const {
        auto *mean_ptr = mean_.data_ptr<float>();
        auto *std_ptr = std_.data_ptr<float>();
        std::normal_distribution<float> dist{*mean_ptr, *std_ptr};
        return tensor(dist(gen));
    }

    [[nodiscard]] double log_density(const Tensor &x) const;
    [[nodiscard]] std::tuple<Tensor, Tensor, Tensor> log_density_gradient(const Tensor &x) const;
    [[nodiscard]] Tensor get_mean() const;
    [[nodiscard]] Tensor get_std() const;

private:
    const Tensor mean_;
    const Tensor std_;
};

// *******************
// *** NormalTrace ***
// *******************

class NormalTrace : public Trace {
public:
    typedef Tensor return_type;
    typedef std::pair<Tensor,Tensor> args_type;
    NormalTrace(Tensor&& value, const NormalDist &dist);
    ~NormalTrace() override = default;
    [[nodiscard]] std::any get_return_value() const override;
    [[nodiscard]] std::any gradients(std::any ret_grad, double scaler, GradientAccumulator& accumulator) override;
    [[nodiscard]] double get_score() const override;
    [[nodiscard]] ChoiceTrie get_choice_trie() const override;

private:
    const NormalDist dist_;
    const Tensor value_;
    double score_;
};

// **************
// *** Normal ***
// **************

/**
 * A generative function representing a univariate normal distribution.
 */
class Normal {
public:
    typedef Tensor return_type;
    typedef NormalTrace trace_type;
    typedef std::pair<Tensor,Tensor> args_type;

    Normal(Tensor mean, Tensor std);

    [[nodiscard]] args_type get_args() const;

    template<class Generator>
    NormalTrace simulate(Generator &gen, const EmptyModule& parameters,
                         bool prepare_for_gradients=false) const {
        Tensor value = dist_.sample(gen);
        return {std::move(value), dist_};
    }

    template<class Generator>
    std::pair<NormalTrace, double>
    generate(Generator &gen, const EmptyModule& parameters, const ChoiceTrie& constraints,
             bool prepare_for_gradients=false) const {
        Tensor value;
        double log_weight;
        if (constraints.has_value()) {
            value = std::any_cast<Tensor>(constraints.get_value());
            log_weight = dist_.log_density(value);
        } else if (constraints.empty()) {
            value = dist_.sample(gen);
            log_weight = 0.0;
        } else {
            throw std::domain_error("expected primitive or empty choice dict");
        }
        return {NormalTrace(std::move(value), dist_), log_weight};
    }

private:
    Tensor mean_tracked_;
    Tensor std_tracked_;
    const NormalDist dist_;
};

}