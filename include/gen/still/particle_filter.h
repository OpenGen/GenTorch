#ifndef GEN_PARTICLE_FILTER_H
#define GEN_PARTICLE_FILTER_H

#include <cmath>
#include <random>
#include <algorithm>


namespace gen::still::smc {

    // TODO add rejuvenation example (by accepting a callback)
    // TODO add different resampling strategies
    // TODO add multi-threading (the SMC algorithm should use its own thread pool, and invoke user callbaks for rejuvenation)
    // TODO see section C of https://arxiv.org/pdf/1301.4019.pdf for in-place propagation
    // TODO see particle cascade algorithm: https://papers.nips.cc/paper/2014/file/7eb7eabbe9bd03c2fc99881d04da9cbd-Paper.
    // TODO add variant where we use a custom proposal, and other variants based on trace translators

    double logsumexp(const std::vector<double>& values) {
        double max = *std::max_element(values.cbegin(), values.cend());
        static double negative_infinity = -std::numeric_limits<double>::infinity();
        if (max == negative_infinity) {
            return negative_infinity;
        } else {
            double sum_exp = 0.0;
            for (auto value : values)
                sum_exp += std::exp(value - max);
            return max + std::log(sum_exp);
        }
    }

    template <typename Trace, typename RNGType>
    class ParticleSystem {
    private:
        size_t num_particles_;
        std::vector<std::unique_ptr<Trace>> traces_;
        std::vector<std::unique_ptr<Trace>> traces_tmp_; // initially contains all nullptrs
        std::vector<double> log_weights_;
        std::vector<double> log_normalized_weights_;
        std::vector<double> two_times_log_normalized_weights_;
        std::vector<double> normalized_weights_;
        std::vector<size_t> parents_;
        std::vector<Trace*> trace_nonowning_ptrs_;
        RNGType& rng_; // TODO this will need to be replaced with a seed_seq for the multi-threaded version.

        double normalize_weights() {
            double log_total_weight = logsumexp(log_weights_);
            for (size_t i = 0; i < num_particles_; i++) {
                log_normalized_weights_[i] = log_weights_[i] - log_total_weight;
                two_times_log_normalized_weights_[i] = 2.0 * log_normalized_weights_[i];
                normalized_weights_[i] = std::exp(log_normalized_weights_[i]);
            }
            return log_total_weight;
        }

        void multinomial_resampling() {
            std::discrete_distribution<size_t> dist(normalized_weights_.cbegin(), normalized_weights_.cend());
            for (size_t i = 0; i < num_particles_; i++) {
                parents_[i] = dist(rng_);
            }
        }

    public:
        ParticleSystem(size_t num_particles, RNGType& rng) :
                num_particles_(num_particles),
                traces_(num_particles),
                traces_tmp_(num_particles),
                log_weights_(num_particles),
                log_normalized_weights_(num_particles),
                two_times_log_normalized_weights_(num_particles),
                normalized_weights_(num_particles),
                parents_(num_particles),
                trace_nonowning_ptrs_(num_particles),
                rng_(rng) {
        }

        template <typename Model, typename Parameters, typename Constraints>
        void init_step(const Model& model, Parameters& parameters, const Constraints& constraints) {
            // TODO make prepare_for_gradient optional for each generate() and update() call
            for (size_t i = 0; i < num_particles_; i++) {
                auto [trace_ptr, log_weight] = model.generate(constraints, rng_, parameters, false);
                traces_[i] = std::move(trace_ptr);
                trace_nonowning_ptrs_[i] = traces_[i].get();
                log_weights_[i] = log_weight;
            }
        }

        // TODO document requirements (and non-requirements) associated with normalization of the model
        // (the model doesn't need to be normalized, and the normalizing constant can change)
        // technically, this means you could just implement this via a ModelChange without constraints
        // but the distinction is still useful for the common case when the models are normalized
        template <typename ModelChange, typename Constraints>
        void step(const ModelChange& model_change, const Constraints& constraints) {
            for (size_t i = 0; i < traces_.size(); i++) {
                // TODO document specification for 'update' (for the case when it includes random choices)
                // TODO also pass an RNG argument to update...
                const auto& [increment, discard, retdiff] = traces_[i]->update(
                        rng_, model_change, constraints, false, false);
                log_weights_[i] += increment;
            }
        }

        [[nodiscard]] double effective_sample_size() const {
            return std::exp(-logsumexp(two_times_log_normalized_weights_));
        }

        double resample() {
            // writes to normalized_weights_
            double log_total_weight = normalize_weights();

            // reads from normalized_weights_ and writes to parents_
            multinomial_resampling();

            // TODO do this in a way that keeps traces in the same thread if you can to minimize data movement
            // and avoids ruining the cache every time we resample
            for (size_t i = 0; i < num_particles_; i++) {
                traces_tmp_[i] = traces_[parents_[i]]->fork();
            }
            for (size_t i = 0; i < num_particles_; i++) {
                traces_[i] = std::move(traces_tmp_[i]); // move assignment
                trace_nonowning_ptrs_[i] = traces_[i].get();
            }
            std::fill(log_weights_.begin(), log_weights_.end(), 0.0);
            return log_total_weight;
        }

        // so that user can run rejuvenation moves on them
        const std::vector<Trace*>& traces() const {
            return trace_nonowning_ptrs_;
        }

    };


}

#endif //GEN_PARTICLE_FILTER_H
