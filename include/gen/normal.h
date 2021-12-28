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

// TODO move implementation to cpp files

namespace distributions::normal {

    using torch::Tensor;
    using torch::tensor;
    using std::make_any;

    constexpr double pi() { return 3.141592653589793238462643383279502884; }

    class NormalDist {
    public:

        NormalDist(Tensor mean, Tensor std) : mean_{mean}, std_{std} {};

        template<class Generator>
        Tensor sample(Generator &gen) const {
            auto *mean_ptr = mean_.data_ptr<float>();
            auto *std_ptr = std_.data_ptr<float>();
            std::normal_distribution<float> dist{*mean_ptr, *std_ptr};
            return tensor(dist(gen));
        }

        [[nodiscard]] double log_density(const Tensor &x) const {
            auto *x_ptr = x.data_ptr<float>();
            auto *mean_ptr = mean_.data_ptr<float>();
            auto *std_ptr = std_.data_ptr<float>();
            std::normal_distribution<float> dist{*mean_ptr, *std_ptr};
            double diff = *x_ptr - *mean_ptr;
            double log_density = -std::log(*std_ptr) - 0.5 * std::sqrt(2.0 * pi()) - 0.5 * (diff * diff / *std_ptr);
            return log_density;
        }

        [[nodiscard]] std::tuple<Tensor, Tensor, Tensor> log_density_gradient(const Tensor &x) const {
            Tensor z = (x - mean_) / std_;
            Tensor x_grad = -z / std_;
            Tensor mu_grad = -x_grad;
            Tensor std_grad = -std_.reciprocal() + (z * z / std_);
            return {x_grad, mu_grad, std_grad};
        }

        [[nodiscard]] Tensor get_mean() const { return mean_; }
        [[nodiscard]] Tensor get_std() const { return std_; }

    private:
        const Tensor mean_;
        const Tensor std_;
    };


    class NormalTrace : public Trace {
    public:
        typedef Tensor return_type;
        typedef pair<Tensor,Tensor> args_type;

        NormalTrace(Tensor&& value, const NormalDist &dist)
                : value_{value}, dist_{dist}, score_(dist.log_density(value)) {}

        ~NormalTrace() override = default;

        [[nodiscard]] std::any get_return_value() const override {
            return make_any<Tensor>(value_); // calls copy constructor for Tensor
        }

        [[nodiscard]] std::any gradients(std::any ret_grad, double scaler) override {
            auto grads = dist_.log_density_gradient(value_);
//            auto x_grad = std::get<0>(grads) + ret_grad;
            auto mean_grad = std::get<1>(grads);
            auto std_grad = std::get<2>(grads);
            return pair<Tensor,Tensor>{mean_grad, std_grad};
        }

        [[nodiscard]] double get_score() const override { return score_; }

        [[nodiscard]] Trie get_choice_trie() const override {
            Trie trie {};
            trie.set_value(value_);
            return trie; // copy elision
        }

    private:
        const NormalDist dist_;
        const Tensor value_;
        double score_;
    };

    /**
     * A generative function representing a univariate normal distribution.
     */
    class Normal {
    public:
        typedef Tensor return_type;
        typedef NormalTrace trace_type;
        typedef pair<Tensor,Tensor> args_type;

        Normal(Tensor mean, Tensor std)
            : mean_tracked_{mean}, std_tracked_{std}, dist_{mean.detach(), std.detach()} {};

        [[nodiscard]] args_type get_args() const {
            return {mean_tracked_, std_tracked_};
        }

        template<class Generator>
        NormalTrace simulate(Generator &gen, bool prepare_for_gradients=false) const {
            Tensor value = dist_.sample(gen);
            return {std::move(value), dist_};
        }

        template<class Generator>
        std::pair<NormalTrace, double>
        generate(Generator &gen, const Trie& constraints, bool prepare_for_gradients=false) const {
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