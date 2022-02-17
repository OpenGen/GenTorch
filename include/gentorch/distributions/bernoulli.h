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

#ifndef GENTORCH_BERNOULLI_H
#define GENTORCH_BERNOULLI_H

#include <gentorch/trace.h>
#include <gentorch/distributions/distributions.h>
#include <random>
#include <torch/torch.h>

namespace gentorch::distributions::bernoulli {

    using torch::Tensor;
    using torch::tensor;
    using gentorch::ChoiceTrie;
    using gentorch::distributions::PrimitiveGenFn;

    class BernoulliDist {
    public:

        BernoulliDist(const Tensor &prob_true);
        BernoulliDist& operator=(const BernoulliDist&) = default;

        template<class RNGType>
        bool sample(RNGType &rng) const {
            std::bernoulli_distribution dist{prob_true_};
            return dist(rng);
        }

        [[nodiscard]] double log_density(bool x) const;

        [[nodiscard]] Tensor log_density_gradient(bool x) const;

    private:
        double prob_true_;
    };

/**
* A generative function representing a bernoulli distribution.
*/
    class Bernoulli;

    class Bernoulli : public PrimitiveGenFn<Bernoulli, Tensor, bool, BernoulliDist> {
    public:
        explicit Bernoulli(Tensor prob_true) : PrimitiveGenFn<M, A, R, D>{prob_true,
                                                                          BernoulliDist{prob_true.detach()}} {}
        static Tensor extract_argument_gradient(Tensor log_density_grad) {
            return log_density_grad;
        }

    };

}

#endif // GENTORCH_BERNOULLI_H