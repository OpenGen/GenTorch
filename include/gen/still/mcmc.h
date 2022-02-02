#ifndef GEN_MH_H
#define GEN_MH_H

#include <cmath>
#include <random>

namespace gen::still::mcmc {

// ***********************************************************************
// *** Metropolis-Hastings using a generative function as the proposal ***
// ***********************************************************************

    double mh_accept_prob(double model_log_weight, double proposal_forward_score, double proposal_backward_score) {
        return exp(model_log_weight + proposal_backward_score - proposal_forward_score);
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
        double proposal_backward_score = backward_proposal.assess(rng, proposal_params, reverse_change);
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

    template<typename TraceType, typename SelectionType, typename RNGType, typename ChoiceBufferType>
    bool mala(TraceType &trace, const SelectionType &selection, double tau,
              ChoiceBufferType &choice_buffer, RNGType &rng) {

        // NOTE: these buffers are only valid up until the next call to update.
        const auto &choice_gradients_buffer = trace.choice_gradients(selection);
        const auto &choice_values_buffer = trace.choices(selection);

        // TODO do math and sample proposed values (fixme)
        choice_buffer = choice_values_buffer + (tau * choice_gradients_buffer);
//        choice_buffer = mala_sample_normals(choice_buffer, tau, rng); // TODO

        const auto& [log_weight, backward_constraints, retdiff] = trace.update(choice_buffer, true, true);
        double prob_accept = log_weight; // TODO calculate it, using backward constrinats
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

    template<typename TraceType, typename SelectionType, typename RNGType, typename ChoiceBufferType>
    bool hmc(TraceType &trace, const SelectionType &selection,
             size_t leapfrog_steps, double eps,
             ChoiceBufferType &momenta_buffer,
             ChoiceBufferType &values_buffer,
             RNGType &rng) {

        // NOTE: this read-only buffer is only valid up until the next call to update.
        const ChoiceBufferType* gradient_buffer = &trace.choice_gradients(selection);

        // this overwrites the memory in the buffer
//        momenta_buffer = sample_momenta(gradient_buffer); // TODO implement
//        double prev_momenta_score = assess_momenta(momenta_buffer); // TODO implement
        double prev_momenta_score = 0.0;

        double log_weight = 0.0;
        for (size_t step = 0; step < leapfrog_steps; step++) {

            // half step on momenta
            momenta_buffer += (eps / 2.0) * (*gradient_buffer);

            // full step on positions
            values_buffer += eps * momenta_buffer; // TODO invalid..., need another choice buffer.

            // get incremental log weight and new gradient
            bool save_prev_state = (step == 0);
            const auto&[log_weight_increment, backward_choices, retdiff] = trace.update(
                    values_buffer, save_prev_state, true);
            log_weight += log_weight_increment;
            gradient_buffer = &trace.choice_gradients(selection);

            // half step on momenta
            momenta_buffer += (eps / 2.0) * (*gradient_buffer);
        }

//        double new_momenta_score = assess_momenta_negative(momenta_buffer); // TODO implement
        double new_momenta_score = 0.0;

        double prob_accept = exp(log_weight + new_momenta_score - prev_momenta_score);
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
