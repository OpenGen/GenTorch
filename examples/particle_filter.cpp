#include <gen/still/mcmc.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <array>
//#include <gen/distributions/normal.h>
#include <gen/utils/randutils.h>
#include <gen/still/particle_filter.h>

// TODO update needs to take an RNG
// TODO implement a version that uses Immer and supports rejuvenation without copying entire histories
//
template <typename T>
std::string join(std::initializer_list<T> list, const std::string& separator) {
    std::ostringstream result;
    auto begin = list.begin();
    auto end = list.end();
    if (begin != end)
        result << *begin++;
    while (begin != end)
        result << separator << *begin++;
    return result.str();
}

struct State {
    State() = default;
    State(double x_, double y_, double vx_, double vy_, double measured_bearing_) :
            x{x_}, y{y_}, vx{vx_}, vy{vy_}, measured_bearing{measured_bearing_} { }
    double x;
    double y;
    double vx;
    double vy;
    double measured_bearing;
    friend std::ostream& operator<<(std::ostream& os, const State& state);
};

std::ostream& operator<<(std::ostream& os, const State& state) {
    os << join({state.x, state.y, state.vx, state.vy, state.measured_bearing}, ",");
    return os;
}


// choice buffer types

struct NextTimeStepObservations {
    explicit NextTimeStepObservations(double measured_bearing_) : measured_bearing{measured_bearing_} {}
    double measured_bearing;
};

class EmptyConstraints {};
const EmptyConstraints empty_constraints{};

// model change types

class ExtendByOneTimeStep {};

// parameters

// TODO learn the parameters using gen/still/sgd.h within an EM algorithm....

struct Parameters {
    double measurement_noise;
    double velocity_stdev;
    double init_x_prior_mean;
    double init_x_prior_stdev;
    double init_y_prior_mean;
    double init_y_prior_stdev;
    double init_vx_prior_mean;
    double init_vx_prior_stdev;
    double init_vy_prior_mean;
    double init_vy_prior_stdev;
};

// return value difference (unused)

class RetDiff {};
const RetDiff ret_diff{};

// Model and trace interfaces

class Trace;

class Model {
    friend class Trace;

private:
    size_t num_time_steps_;

public:
    Model() : num_time_steps_(0) {}
    explicit Model(size_t num_time_steps) : num_time_steps_{num_time_steps} {}

    template <typename RNGType>
    Trace simulate(RNGType& rng, Parameters& parameters, bool prepare_for_gradient) const;

    template <typename RNGType>
    std::pair<std::unique_ptr<Trace>,double> generate(const NextTimeStepObservations& constraints, RNGType& rng,
                                                      Parameters& parameters, bool prepare_for_gradient) const;

    // NOT implemented:
    // 1. simulate in-place
    // 2. generate in-place
};


class Trace {
    friend class Model;
private:
    Model model_;
    const Parameters* parameters_;
    std::vector<State> states_;
    Trace(const Model& model, const Parameters& parameters, std::vector<State> states) :
            model_{model}, parameters_{&parameters}, states_{std::move(states)} {}
public:
    Trace() = delete;
    Trace(const Trace& other) = delete;
    Trace(Trace&& other) = delete;
    Trace& operator=(const Trace& other) = delete;
    Trace& operator=(Trace&& other) noexcept = delete;

    template <typename RNGType>
    std::tuple<double,const EmptyConstraints&,const RetDiff&> update(
            RNGType& rng, const ExtendByOneTimeStep& model_change, const NextTimeStepObservations& constraints,
            bool save_previous, bool prepare_for_gradient);

    template <typename RNGType>
    std::tuple<double,const EmptyConstraints&,const RetDiff&> update(
            RNGType& rng, const ExtendByOneTimeStep& model_change, const EmptyConstraints& constraints,
            bool save_previous, bool prepare_for_gradient);

    [[nodiscard]] State state(size_t i) const {
        return states_[i];
    }

    void revert();

    std::unique_ptr<Trace> fork();

    // NOT implemented:
    // 1. choice_gradients
    // 2. parameter_gradients
};

// Model and trace implementations

template <typename ValueType, typename RNGType>
ValueType normal(ValueType mean, ValueType stdev, RNGType& rng) {
    std::normal_distribution<ValueType> dist{mean, stdev};
    return dist(rng);
}

template <typename ValueType>
ValueType normal_logpdf(ValueType x, ValueType mean, ValueType stdev) {
    static ValueType logSqrt2Pi = 0.5*std::log(2*M_PI);
    ValueType diff = (x - mean);
    return -logSqrt2Pi - std::log(stdev) - 0.5 * diff * diff;
}

template <typename RNGType>
std::tuple<double,double,double,double> sample_init_prior(const Parameters& parameters, RNGType& rng) {
    double new_x = normal(parameters.init_x_prior_mean, parameters.init_x_prior_stdev, rng);
    double new_y = normal(parameters.init_y_prior_mean, parameters.init_y_prior_stdev, rng);
    double new_vx = normal(parameters.init_vx_prior_mean, parameters.init_vx_prior_stdev, rng);
    double new_vy = normal(parameters.init_vy_prior_mean, parameters.init_vy_prior_stdev, rng);
    return {new_x, new_y, new_vx, new_vy};
}

template <typename RNGType>
std::tuple<double,double,double,double> sample_dynamics_prior(const State& state, const Parameters& parameters,
                                                              RNGType& rng) {
    double new_vx = normal(state.vx, parameters.velocity_stdev, rng);
    double new_vy = normal(state.vy, parameters.velocity_stdev, rng);
    double new_x = state.x + new_vx;
    double new_y = state.y + new_vy;
    return {new_x, new_y, new_vx, new_vy};
}

template <typename RNGType>
double sample_observation_prior(double x, double y, const Parameters& parameters, RNGType& rng) {
    double bearing = std::atan2(y, x);
    return normal(bearing, parameters.measurement_noise, rng);
}

template <typename RNGType>
std::pair<State,double> initial_importance_sample(const Parameters& parameters, double measured_bearing,
                                                  RNGType& rng) {
    auto [new_x, new_y, new_vx, new_vy] = sample_init_prior(parameters, rng);
    double bearing = std::atan2(new_y, new_x);
    double new_measured_bearing = measured_bearing;
    double log_weight = normal_logpdf(new_measured_bearing, bearing, parameters.measurement_noise);
    State new_state{new_x, new_y, new_vx, new_vy, new_measured_bearing};
    return {new_state, log_weight};
}

template <typename RNGType>
std::pair<State,double> incremental_importance_sample(const Parameters& parameters, double measured_bearing,
                                                      const State& state, RNGType& rng) {
    auto [new_x, new_y, new_vx, new_vy] = sample_dynamics_prior(state, parameters, rng);
    double bearing = std::atan2(new_y, new_x);
    double new_measured_bearing = measured_bearing;
    double log_weight = normal_logpdf(new_measured_bearing, bearing, parameters.measurement_noise);
    State new_state{new_x, new_y, new_vx, new_vy, new_measured_bearing};
    return {new_state, log_weight};
}

template <typename RNGType>
State extend_without_observation(const Parameters& parameters, const State& state, RNGType& rng) {
    auto [new_x, new_y, new_vx, new_vy] = sample_dynamics_prior(state, parameters, rng);
    double new_measured_bearing = sample_observation_prior(new_x, new_y, parameters, rng);
    State new_state{new_x, new_y, new_vx, new_vy, new_measured_bearing};
    return new_state;
}

template <typename RNGType>
Trace Model::simulate(RNGType& rng, Parameters& parameters, bool prepare_for_gradient) const {
    auto [new_x, new_y, new_vx, new_vy] = sample_init_prior(parameters, rng);
    std::vector<State> states(num_time_steps_);
    double new_measured_bearing = sample_observation_prior(new_x, new_y, parameters, rng);
    states[0] = {new_x, new_y, new_vx, new_vy, new_measured_bearing};
    for (size_t i = 1; i < num_time_steps_; i++) {
        std::tie(new_x, new_y, new_vx, new_vy) = sample_dynamics_prior(states[i-1], parameters, rng);
        new_measured_bearing = sample_observation_prior(new_x, new_y, parameters, rng);
        states[i] = {new_x, new_y, new_vx, new_vy, new_measured_bearing};
    }
    return Trace{*this, parameters, states};
}


template <typename RNGType>
std::pair<std::unique_ptr<Trace>,double> Model::generate(const NextTimeStepObservations& constraints, RNGType& rng,
                                                         Parameters& parameters, bool prepare_for_gradient) const {
    if (prepare_for_gradient)
        throw std::logic_error("prepare_for_gradient in generate not implemented");
    if (num_time_steps_ != 0)
        throw std::logic_error("generate not implemented for num_time_steps != 0");
    // sample initial x, y, vx, vy from the prior and compute importance weight
    auto [state, log_weight] = initial_importance_sample(parameters, constraints.measured_bearing, rng);
    auto trace = std::unique_ptr<Trace>(new Trace(*this, parameters, {state}));
    return {std::move(trace), log_weight};
}

template <typename RNGType>
std::tuple<double,const EmptyConstraints&,const RetDiff&> Trace::update(
        RNGType& rng, const ExtendByOneTimeStep& model_change, const NextTimeStepObservations& constraints,
        bool save_previous, bool prepare_for_gradient) {
    if (prepare_for_gradient)
        throw std::logic_error("prepare_for_gradient in update not implemented");
    if (save_previous)
        throw std::logic_error("save_previous in update not implemented");
    auto [new_state, log_weight] = incremental_importance_sample(
            *parameters_, constraints.measured_bearing, states_.back(), rng);
    states_.emplace_back(new_state);
    return {log_weight, empty_constraints, ret_diff};
}

template <typename RNGType>
std::tuple<double,const EmptyConstraints&,const RetDiff&> Trace::update(
        RNGType& rng, const ExtendByOneTimeStep& model_change, const EmptyConstraints& constraints,
        bool save_previous, bool prepare_for_gradient) {
    if (prepare_for_gradient)
        throw std::logic_error("prepare_for_gradient in update not implemented");
    if (save_previous)
        throw std::logic_error("save_previous in update not implemented");
    auto new_state = extend_without_observation(*parameters_, states_.back(), rng);
    states_.emplace_back(new_state);
    return {0.0, empty_constraints, ret_diff};
}

void Trace::revert() {
    throw std::logic_error("revert not implemented");
}

std::unique_ptr<Trace> Trace::fork() {
    return std::unique_ptr<Trace>(new Trace(model_, *parameters_, states_));
}

// Example

int main(int argc, char* argv[]) {
    using std::endl;
    using std::cout;
    using std::cerr;

    Parameters parameters;
    parameters.measurement_noise = 0.005;
    parameters.velocity_stdev = std::sqrt(0.005);
    parameters.init_x_prior_mean = 0.01;
    parameters.init_x_prior_stdev = std::sqrt(0.01);
    parameters.init_y_prior_mean = 0.95;
    parameters.init_y_prior_stdev = std::sqrt(0.01);
    parameters.init_vx_prior_mean = 0.002;
    parameters.init_vx_prior_stdev = std::sqrt(0.01);
    parameters.init_vy_prior_mean = -0.013;
    parameters.init_vy_prior_stdev = std::sqrt(0.01);

    size_t num_time_steps = 50;

    randutils::seed_seq_fe128 seed_seq {1};
    std::mt19937 rng(seed_seq);

    // simulate data
    Model simulated_data_model{num_time_steps};
    Trace simulated_trace = simulated_data_model.simulate(rng, parameters, false);

    // create constraints
    std::vector<NextTimeStepObservations> observations;
    for (size_t i = 0; i < num_time_steps; i++)
        observations.emplace_back(simulated_trace.state(i).measured_bearing);

    // Actual model (initialize with zero time steps)
    Model model{};

    // run the particle filter
    size_t num_particles = 100;
    gen::still::smc::ParticleSystem<Trace,std::mt19937> particle_filter{model, parameters, observations[0],
                                                                        num_particles, rng};
    for (size_t i = 1; i < num_time_steps; i++) {
        particle_filter.step(ExtendByOneTimeStep{}, observations[i]);
        double ess = particle_filter.effective_sample_size();
        cerr << "effective sample size: " << ess << endl;
        double log_weight = particle_filter.resample();
        cerr << "log weight from resample: " << log_weight << endl;
    }

    // print out ground truth results
    std::ofstream ground_truth_fs ("pf_plots/ground_truth.csv", std::ofstream::out);
    for (size_t i = 0; i < num_time_steps; i++)
        ground_truth_fs << simulated_trace.state(i) << endl;

    // print out inferred traces
    size_t idx = 0;
    for (auto& trace : particle_filter.traces()) {
        std::stringstream fname;
        fname << "pf_plots/" << idx++ << ".csv";
        std::ofstream fs (fname.str(), std::ofstream::out);
        for (size_t i = 0; i < num_time_steps; i++)
            fs << trace->state(i) << endl;
    }



}

//std::ostream& operator<<(std::ostream& os, const State& state) {
//    os << join({state.x, state.y, state.vx, state.vy, state.measured_bearing}, ",") << std::endl;
//}
