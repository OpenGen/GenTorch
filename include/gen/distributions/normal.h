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
#include <gen/distributions/distributions.h>

#include <random>

#include <torch/torch.h>

namespace gen::distributions::normal {

    using torch::Tensor;
    using torch::tensor;
    using gen::distributions::PrimitiveGenFn;

    const double pi = 3.141592653589793238462643383279502884;
    const double sqrt_2_pi = std::sqrt(2.0 * pi);

    double log_density(double mean, double std, double x);
    std::tuple<double, double, double> log_density_gradient(double mean, double std, double x);

    class NormalDist {
    public:

        NormalDist(const Tensor& mean, const Tensor& std);

        template<class RNGType>
        Tensor sample(RNGType &rng) const {
            std::normal_distribution<double> dist {mean_, std_};
            return tensor(dist(rng));
        }

        [[nodiscard]] double log_density(const Tensor &x) const;

        [[nodiscard]] std::tuple<Tensor, Tensor, Tensor> log_density_gradient(const Tensor &x) const;

    private:
        const double mean_;
        const double std_;
    };

    /**
     * A generative function representing a univariate normal distribution.
     */
    class Normal;

    class Normal : public PrimitiveGenFn<Normal, std::pair<Tensor, Tensor>, Tensor, NormalDist> {
    public:
        Normal(Tensor mean, Tensor std) : PrimitiveGenFn<M, A, R, D>{{mean, std},
                                                                     NormalDist{mean.detach(), std.detach()}} {}
        static std::pair<Tensor, Tensor> extract_argument_gradient(const std::tuple<Tensor, Tensor, Tensor>& log_density_grad) {
            return {std::get<1>(log_density_grad), std::get<2>(log_density_grad)};
        }
    };

}