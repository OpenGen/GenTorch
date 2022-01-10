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
#include <gen/distributions/normal.h>

using torch::Tensor;
using torch::tensor;

namespace gen::distributions::normal {

    // logic

    double log_density(double mean, double std, double x) {
        double diff = x - mean;
        return -std::log(std) - 0.5 * sqrt_2_pi() - 0.5 * (diff * diff / std);
    }

    std::tuple<double, double, double> log_density_gradient(double mean, double std, double x) {
        double z = (x - mean) / std;
        double x_grad = -z / std;
        double mu_grad = -x_grad;
        double std_grad = -(1.0 / std) + (z * z / std);
        return {x_grad, mu_grad, std_grad};
    }

    // boilerplate

    NormalDist::NormalDist(const Tensor& mean, const Tensor& std) :
        mean_{*mean.data_ptr<float>()},
        std_{*std.data_ptr<float>()} {}

    [[nodiscard]] double NormalDist::log_density(const Tensor& x) const {
        return gen::distributions::normal::log_density(mean_, std_, *x.data_ptr<float>());
    }

    [[nodiscard]] std::tuple<Tensor, Tensor, Tensor> NormalDist::log_density_gradient(const Tensor& x) const {
        auto grad = gen::distributions::normal::log_density_gradient(mean_, std_, *x.data_ptr<float>());
        return {tensor(std::get<0>(grad)), tensor(std::get<1>(grad)), tensor(std::get<2>(grad))};
    }

}