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

#include <gen/distributions/uniform_continuous.h>

namespace gen::distributions::uniform_continuous {

    // logic

    double log_density(double min, double max, double x) {
        if (x >= min && x <= max) {
            return -std::log(max - min);
        } else {
            return -std::numeric_limits<double>::infinity();
        }
    }

    std::tuple<double, double, double> log_density_gradient(double min, double max, double x) {
        double inv_diff = 1.0 / (max - min);
        return {0.0, inv_diff, -inv_diff};
    }

    // boilerplate

    UniformContinuousDist::UniformContinuousDist(const Tensor& min, const Tensor& max) :
        min_{*min.data_ptr<float>()},
        max_{*max.data_ptr<float>()} {}

    [[nodiscard]] double UniformContinuousDist::log_density(const Tensor& x) const {
        float x_float = *x.data_ptr<float>();
        return gen::distributions::uniform_continuous::log_density(min_, max_, x_float);
    }

    [[nodiscard]] std::tuple<Tensor, Tensor, Tensor> UniformContinuousDist::log_density_gradient(const Tensor& x) const {
        float x_float = *x.data_ptr<float>();
        auto grad = gen::distributions::uniform_continuous::log_density_gradient(min_, max_, x_float);
        return {tensor(std::get<0>(grad)), tensor(std::get<1>(grad)), tensor(std::get<2>(grad))};
    }

}