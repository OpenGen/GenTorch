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

#include <torch/torch.h>
#include <gen/normal.h>
#include <gen/parameters.h>
#include <gen/trie.h>

namespace gen::distributions::normal {

using torch::Tensor;
using torch::tensor;
using gen::ChoiceTrie;
using gen::GradientAccumulator;

// ******************
// *** NormalDist ***
// ******************

NormalDist::NormalDist(Tensor mean, Tensor std) : mean_{mean}, std_{std} {};

[[nodiscard]] double NormalDist::log_density(const Tensor &x) const {
    auto *x_ptr = x.data_ptr<float>();
    auto *mean_ptr = mean_.data_ptr<float>();
    auto *std_ptr = std_.data_ptr<float>();
    std::normal_distribution<float> dist{*mean_ptr, *std_ptr};
    double diff = *x_ptr - *mean_ptr;
    double log_density = -std::log(*std_ptr) - 0.5 * std::sqrt(2.0 * pi()) - 0.5 * (diff * diff / *std_ptr);
    return log_density;
}

[[nodiscard]] std::tuple<Tensor, Tensor, Tensor> NormalDist::log_density_gradient(const Tensor &x) const {
    Tensor z = (x - mean_) / std_;
    Tensor x_grad = -z / std_;
    Tensor mu_grad = -x_grad;
    Tensor std_grad = -std_.reciprocal() + (z * z / std_);
    return {x_grad, mu_grad, std_grad};
}

[[nodiscard]] Tensor NormalDist::get_mean() const { return mean_; }
[[nodiscard]] Tensor NormalDist::get_std() const { return std_; }


// ******************
// *** NormalTrace ***
// ******************

NormalTrace::NormalTrace(Tensor&& value, const NormalDist &dist)
        : value_{value}, dist_{dist}, score_(dist.log_density(value)) {}

[[nodiscard]] std::any NormalTrace::get_return_value() const {
    return std::make_any<Tensor>(value_); // calls copy constructor for Tensor
}

[[nodiscard]] std::any NormalTrace::gradients(std::any ret_grad, double scaler, GradientAccumulator& accumulator) {
    auto grads = dist_.log_density_gradient(value_);
    auto mean_grad = std::get<1>(grads);
    auto std_grad = std::get<2>(grads);
    return std::pair<Tensor,Tensor>{mean_grad, std_grad};
}

[[nodiscard]] double NormalTrace::get_score() const { return score_; }

[[nodiscard]] ChoiceTrie NormalTrace::get_choice_trie() const {
    ChoiceTrie trie {};
    trie.set_value(value_);
    return trie; // copy elision
}

// **************
// *** Normal ***
// **************

Normal::Normal(Tensor mean, Tensor std)
            : mean_tracked_{mean}, std_tracked_{std}, dist_{mean.detach(), std.detach()} {};

[[nodiscard]] Normal::args_type Normal::get_args() const {
    return {mean_tracked_, std_tracked_};
}

}