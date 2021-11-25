#ifndef NORMAL_H
#define NORMAL_H

#include <random>

#include <torch/torch.h>

#include "interfaces/trace.h"

namespace distributions::normal {

    using torch::Tensor;
    using torch::tensor;
    using std::make_any;

    constexpr double pi() { return 3.141592653589793238462643383279502884; }

    class NormalDist {
    public:

        // NOTE: we should be able to pass-by-value without copying, since Tensor is liked a shared_ptr
        NormalDist(const Tensor &mean, const Tensor &std) : mean_{mean}, std_{std} {};

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
            return {tensor(0.0), tensor(0.0), tensor(0.0)}; // TODO
        }

    private:
        const Tensor mean_;
        const Tensor std_;
    };


    class NormalTrace : Trace {
    public:
        NormalTrace(const Tensor &value, const NormalDist &dist)
                : value_{value}, dist_{dist}, score_(dist.log_density(value)) {}

        [[nodiscard]] std::any get_return_value() const override { return make_any<Tensor>(value_); }

        [[nodiscard]] std::vector<Tensor> gradients(const Tensor &ret_grad) const {
            auto grads = dist_.log_density_gradient(value_);
            auto x_grad = std::get<0>(grads) + ret_grad;
            auto mean_grad = std::get<1>(grads);
            auto std_grad = std::get<2>(grads);
            return {x_grad, mean_grad, std_grad};
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

    class Normal {
    public:
        typedef Tensor return_type;
        typedef NormalTrace trace_type;

        Normal(const Tensor &mean, const Tensor &std) : dist_{mean, std} {};

        template<class Generator>
        NormalTrace simulate(Generator &gen) const {
            Tensor value = dist_.sample(gen);
            return NormalTrace(NormalTrace(value, dist_));
        }

        template<class Generator>
        std::pair<NormalTrace, double>
        generate(Generator &gen, const Trie& constraints) const {
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
            return {NormalTrace(value, dist_), log_weight};
        }

    private:
        const NormalDist dist_;
    };

}


#endif // NORMAL_H
