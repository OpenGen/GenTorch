#ifndef GEN_MH_H
#define GEN_MH_H

#include <cmath>
#include <random>

namespace gen::still::mcmc {

// ***********************************************************************
// *** Metropolis-Hastings using a generative function as the proposal ***
// ***********************************************************************

    double mh_accept_prob(double model_log_weight, double proposal_forward_score, double proposal_backward_score) {
        return std::min(1.0, std::exp(model_log_weight + proposal_backward_score - proposal_forward_score));
    }

// make_proposal must be a callable that takes a model trace as its only argument and returns a generative function
// with arguments instantiated
    template<typename TraceType, typename ProposalTraceType,
            typename MakeProposalType, typename ProposalParametersType,
            typename RNGType>
    bool mh(TraceType &model_trace, const MakeProposalType &make_proposal,
            ProposalParametersType &proposal_params, RNGType &rng,
            ProposalTraceType &proposal_trace, bool prepare_for_gradient=false) {
        auto proposal = make_proposal(model_trace);
        proposal.simulate(proposal_trace, rng, proposal_params, false);
        double proposal_forward_score = proposal_trace.get_score();
        const auto &change = proposal_trace.choices();
        // note that log_weight and reverse_change and retdiff are only available in between
        // a call to update and a call to revert (before calling update, they will error, and after calling revert
        // they will error).
        const auto& [model_log_weight, reverse_change, retdiff] = model_trace.update(
                change, true, prepare_for_gradient);
        auto backward_proposal = make_proposal(model_trace);
        double proposal_backward_score = backward_proposal.assess(reverse_change, rng, proposal_params);
        double prob_accept = mh_accept_prob(model_log_weight, proposal_forward_score, proposal_backward_score);
        std::bernoulli_distribution dist{prob_accept};
        bool accept = dist(rng);
        if (!accept) {
            model_trace.revert();
        }
        return accept;
    }

    template<typename TraceType, typename MakeProposalType, typename ProposalParametersType, typename RNGType>
    size_t mh_chain(TraceType &model_trace, const MakeProposalType &make_proposal,
                    ProposalParametersType &proposal_params, size_t num_steps, RNGType &rng) {
        // allocate initial proposal trace memory
        auto proposal = make_proposal(model_trace);
        auto proposal_trace = proposal.simulate(rng, proposal_params, false);
        // do iterations, reusing proposal trace memory (as well as model trace memory)
        size_t num_accepted = 0;
        for (size_t step = 0; step < num_steps; step++) {
            bool accepted = mh(model_trace, make_proposal, proposal_params, rng, proposal_trace);
            if (accepted)
                num_accepted++;
        }
        return num_accepted;
    }


// *******************************************
// *** Metropolis-Adjusted Langevin (MALA) ***
// *******************************************

    template <typename ChoiceBuffer, typename RNGType>
    double mala_propose_values(ChoiceBuffer& proposal, const ChoiceBuffer& current,
                               const ChoiceBuffer& gradient, double tau, RNGType& rng) {
        double stdev = std::sqrt(2 * tau);
        static std::normal_distribution<double> standard_normal {0.0, 1.0};
        static double logSqrt2Pi = 0.5*std::log(2*M_PI);
        double log_density = (current.cend() - current.cbegin()) * (-std::log(stdev) - logSqrt2Pi);

        // first compute the mean in-place
        proposal = current + (tau * gradient);

        // then sample new values (over-writing the mean), and finish computing the log density
        for (auto proposal_it = proposal.begin(); proposal_it != proposal.end(); proposal_it++) {
            double standard_normal_increment = standard_normal(rng);
            *proposal_it += (standard_normal_increment * stdev);
            log_density += -0.5 * (standard_normal_increment * standard_normal_increment);
        }
        return log_density;
    }

    template <typename ChoiceBuffer>
    double mala_assess(const ChoiceBuffer& proposed, const ChoiceBuffer& current,
                       const ChoiceBuffer& gradient, double tau, ChoiceBuffer& storage) {
        double stdev = std::sqrt(2 * tau);
        static double logSqrt2Pi = 0.5*std::log(2*M_PI);
        double log_density = (current.cend() - current.cbegin()) * (-std::log(stdev) - logSqrt2Pi);

        ChoiceBuffer& proposal_mean = storage; // rename it
        proposal_mean = current + (tau * gradient);

        auto proposal_mean_it = proposal_mean.cbegin();
        for (auto proposed_it = proposed.cbegin(); proposed_it != proposed.cend(); proposed_it++) {
            double standard_normal_increment = (*proposed_it - *(proposal_mean_it++)) / stdev;
            log_density += -0.5 * (standard_normal_increment * standard_normal_increment);
        }
        return log_density;
    }

    template<typename TraceType, typename SelectionType, typename RNGType, typename ChoiceBufferType>
    bool mala(TraceType &trace, const SelectionType &selection, double tau,
              ChoiceBufferType &storage1, ChoiceBufferType& storage2, RNGType &rng) {

        // NOTE: these buffers are only valid up until the next call to update.
        ChoiceBufferType& proposed_values = storage1;
        double forward_log_density = mala_propose_values(proposed_values, trace.choices(selection),
                                                         trace.choice_gradients(selection), tau, rng);
        const auto& [log_weight, previous_values, retdiff] = trace.update(proposed_values, true, true);

        // compute backward log density
        double backward_log_density = mala_assess(previous_values, proposed_values,
                                                  trace.choice_gradients(selection), tau,
                                                  storage2);

        double prob_accept = std::min(1.0, std::exp(log_weight + backward_log_density - forward_log_density));
        std::bernoulli_distribution dist{prob_accept};
        bool accept = dist(rng);
        if (!accept) {
            trace.revert();
        }
        return accept;
    }

// *************************************
// *** Hamiltonian Monte Carlo (HMC) ***
// *************************************


    template <typename ChoiceBufferType, typename RNGType>
    void sample_momenta(ChoiceBufferType& momenta, RNGType& rng) {
        static std::normal_distribution<double> standard_normal{0.0, 1.0};
        for (auto& momentum : momenta)
            momentum = standard_normal(rng);
    }

    template <typename ChoiceBufferType>
    double assess_momenta(const ChoiceBufferType& momenta) {
        static double logSqrt2Pi = 0.5*std::log(2*M_PI);
        double sum = 0.0;
        for (const auto& momentum : momenta)
            sum += -0.5 * momentum * momentum;
        return sum - (momenta.cend() - momenta.cbegin()) * logSqrt2Pi;
    }

    template<typename TraceType, typename SelectionType, typename RNGType, typename ChoiceBufferType>
    bool hmc(TraceType &trace, const SelectionType &selection,
             size_t leapfrog_steps, double eps,
             ChoiceBufferType &momenta_buffer,
             ChoiceBufferType &values_buffer,
             RNGType &rng) {

        // NOTE: this read-only buffer is only valid up until the next call to update.
        const ChoiceBufferType* gradient_buffer = &trace.choice_gradients(selection);

        // this overwrites the memory in the buffer
        sample_momenta(momenta_buffer, rng);
        double prev_momenta_score = assess_momenta(momenta_buffer);

        double log_weight = 0.0;
        for (size_t step = 0; step < leapfrog_steps; step++) {

            // half step on momenta
            momenta_buffer += (eps / 2.0) * (*gradient_buffer);

            // full step on positions
            values_buffer += eps * momenta_buffer;

            // get incremental log weight and new gradient
            bool save_prev_state = (step == 0);
            const auto&[log_weight_increment, backward_choices, retdiff] = trace.update(
                    values_buffer, save_prev_state, true);
            log_weight += log_weight_increment;
            gradient_buffer = &trace.choice_gradients(selection);

            // half step on momenta
            momenta_buffer += (eps / 2.0) * (*gradient_buffer);
        }

        double new_momenta_score = assess_momenta(momenta_buffer);

        double prob_accept = std::min(1.0, std::exp(log_weight + new_momenta_score - prev_momenta_score));
        std::bernoulli_distribution dist{prob_accept};
        bool accept = dist(rng);
        if (!accept) {
            trace.revert();
        }
        return accept;
    }


// TODO elliptical slice sampling
// TODO involutive MCMC...

}
#endif //GEN_MH_H
