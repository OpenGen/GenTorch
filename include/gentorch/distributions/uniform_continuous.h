/* Copyright 2021-2022 Massachusetts Institute of Technology

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


#ifndef GENTORCH_UNIFORM_CONTINUOUS_H
#define GENTORCH_UNIFORM_CONTINUOUS_H

#include <gentorch/trace.h>
#include <gentorch/distributions/distributions.h>
#include <random>
#include <torch/torch.h>

namespace gentorch::distributions::uniform_continuous {

    using torch::Tensor;
    using torch::tensor;
    using gentorch::ChoiceTrie;
    using gentorch::distributions::PrimitiveGenFn;

    class UniformContinuousDist {
    public:

        UniformContinuousDist(const Tensor& min, const Tensor& max);
        UniformContinuousDist& operator=(const UniformContinuousDist&) = default;

        template<class RNGType>
        Tensor sample(RNGType &rng) const {
            std::uniform_real_distribution<double> dist{min_, max_};
            return tensor(dist(rng));
        }

        [[nodiscard]] double log_density(const Tensor& x) const;
        [[nodiscard]] std::tuple<Tensor,Tensor,Tensor> log_density_gradient(const Tensor& x) const;

    private:
        double min_;
        double max_;
    };

/**
 * A generative function representing a univariate uniform_continuous distribution.
 */
    class UniformContinuous;
    class UniformContinuous : public PrimitiveGenFn<UniformContinuous, std::pair<Tensor,Tensor>, Tensor, UniformContinuousDist> {
    public:
        UniformContinuous(Tensor min, Tensor max) : PrimitiveGenFn<M,A,R,D>{
                {min, max}, UniformContinuousDist{min.detach(), max.detach()}} {}
        static std::pair<Tensor,Tensor> extract_argument_gradient(const std::tuple<Tensor,Tensor,Tensor>& log_density_grad) {
            return {std::get<1>(log_density_grad), std::get<2>(log_density_grad)};
        }
    };

}

#endif // GENTORCH_UNIFORM_CONTINUOUS_H