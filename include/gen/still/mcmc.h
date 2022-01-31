#ifndef GEN_MH_H
#define GEN_MH_H

#include <cmath>
#include <random>


// ***********************************************************************
// *** Metropolis-Hastings using a generative function as the proposal ***
// ***********************************************************************

double mh_accept_prob(double model_log_weight, double proposal_forward_score, double proposal_backward_score) {
    return exp(model_log_weight + proposal_backward_score - proposal_forward_score);
}
// make_proposal must be a callable that takes a model trace as its only argument and returns a generative function
// with arguments instantiated
template <typename TraceType, typename ProposalTraceType,
          typename MakeProposalType, typename ProposalParametersType,
          typename RNGType>
bool mh(TraceType& model_trace, const MakeProposalType& make_proposal,
        ProposalParametersType& proposal_params, RNGType& rng,
        ProposalTraceType& proposal_trace) {
    auto proposal = make_proposal(model_trace);
    proposal.simulate(proposal_trace, rng, proposal_params, false); // TODO overwrites previous proposal trace
    double proposal_forward_score = proposal_trace.get_score();
    const auto& change = proposal_trace.read_only_choice_buffer();
    // note that log_weight and reverse_change and retdiff are only available in between
    // a call to update and a call to revert (before calling update, they will error, and after calling revert
    // they will error).
    model_trace.update(change);
    double model_log_weight = model_trace.log_weight();
    const auto& reverse_change = model_trace.reverse_change();
    auto backward_proposal = make_proposal(model_trace);
    double proposal_backward_score = backward_proposal.assess(rng, proposal_params, reverse_change);
    double prob_accept = mh_accept_prob(model_log_weight, proposal_forward_score, proposal_backward_score);
    std::bernoulli_distribution dist {prob_accept};
    bool accept = dist(rng);
    if (!accept) {
        model_trace.revert();
    }
    return accept;
}

template <typename TraceType, typename MakeProposalType, typename ProposalParametersType, typename RNGType>
num_accepted mh_chain(TraceType& model_trace, const MakeProposalType& make_proposal,
        ProposalParametersType& proposal_params, size_t num_steps, RNGType& rng) {
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

template <typename TraceType, typename SelectionType, typename RNGType>
bool mala(TraceType& trace, const SelectionType& selection, double tau,
          typename TraceType::choice_buffer_t& choice_buffer, RNGType& rng) {
    typedef typename TraceType::choice_buffer_t ChoiceBufferType;

    // NOTE: these buffers are only valid up until the next call to update.
    const ChoiceBufferType& choice_gradients_buffer = trace.choice_gradients(selection);
    const ChoiceBufferType& choice_values_buffer = trace.get_choices(selection);

    // TODO do math and sample proposed values (fixme)
    choice_buffer = choice_values_buffer + (tau * choice_gradients_buffer);
    choice_buffer = mala_sample_normals(choice_buffer, tau, rng);

    model_trace.update(choice_buffer);
    double prob_accept = model_trace.log_weight(); // TODO calculate it, including backward..
    std::bernoulli_distribution dist {prob_accept};
    bool accept = dist(rng);
    if (!accept) {
        model_trace.revert();
    }
    return accept;
}

template <typename TraceType, typename MakeProposalType, typename ProposalParametersType, typename RNGType>
num_accepted mala_chain(TraceType& model_trace, const SelectionType& selection, double tau, size_t num_steps, RNGType& rng) {
    typedef typename TraceType::choice_buffer_t ChoiceBufferType;
    // get a choice buffer to use (by calling the copy constructor on the buffer returned by model_trace.get_choices)
    auto ChoiceBufferType choice_buffer{model_trace.get_choices(selection)};
    // do iterations, reusing proposal trace memory (as well as model trace memory)
    size_t num_accepted = 0;
    for (size_t step = 0; step < num_steps; step++) {
        bool accepted = mala(model_trace, selection, tau, choice_buffer, rng);
        if (accepted)
            num_accepted++;
    }
    return num_accepted;
}

// *************************************
// *** Hamiltonian Monte Carlo (HMC) ***
// *************************************

template <typename TraceType, typename SelectionType, typename RNGType>
bool hmc(TraceType& trace, const SelectionType& selection,
         size_t leapfrog_steps, double eps,
         typename TraceType::choice_buffer_t& momenta_buffer,
         typename TraceType::choice_buffer_t& values_buffer) {
    typedef typename TraceType::choice_buffer_t ChoiceBufferType;

    // NOTE: this read-only buffer is only valid up until the next call to update.
    typedef typename TraceType::choice_buffer_t ChoiceBufferType;
    const ChoiceBufferType* gradient_buffer = trace.choice_gradients(selection);

    // this overwrites the memory in the buffer
    momenta_buffer = sample_momenta(gradient_buffer); // TODO implement
    double prev_momenta_score = assess_momenta(momenta_buffer); // TODO implement

    double log_weight = 0.0;
    for (size_t step = 0; step < leapfrog_steps; step++) {

        // half step on momenta
        momenta_buffer += (eps / 2.0) * gradient_buffer;

        // full step on positions
        values_buffer += eps * momenta_buffer; // TODO invalid...

        // get incremental log weight and new gradient
        trace.update(values_buffer, step == 0); // 'false' is 'overwrite swap'
        log_weight += trace.get_log_weight();
        gradient_buffer = trace.choice_gradients(selection);

        // half step on momenta
        momenta_buffer += (eps / 2.0) * gradient_buffer;
    }

    double new_momenta_score = assess_momenta_negative(momenta_buffer); // TODO implement

    double prob_accept = exp(log_weight + new_momenta_score - prev_momenta_score);
    std::bernoulli_distribution dist {prob_accept};
    bool accept = dist(rng);
    if (!accept) {
        model_trace.revert();
    }
    return accept;
}

// TODO elliptical slice sampling
// TODO involutive MCMC...

#endif //GEN_MH_H
