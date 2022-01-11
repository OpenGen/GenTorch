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

#include <cmath>

#include <gen/distributions/bernoulli.h>

namespace gen::distributions::bernoulli {

    BernoulliDist::BernoulliDist(const Tensor& prob_true) : prob_true_{*prob_true.data_ptr<float>()} {}

    [[nodiscard]] double BernoulliDist::log_density(bool x) const {
        if (x) {
            return std::log(prob_true_);
        } else {
            return std::log(1.0 - prob_true_);
        }
    }

    [[nodiscard]] Tensor BernoulliDist::log_density_gradient(bool x) const {
        double prob_true_grad;
        if (x) {
            prob_true_grad = 1.0 / prob_true_;
        } else {
            prob_true_grad = -1.0 / (1.0 - prob_true_);
        }
        return tensor(prob_true_grad);
    }

}