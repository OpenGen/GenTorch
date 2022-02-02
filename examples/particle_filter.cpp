#include <gen/still/mcmc.h>

#include <utility>
#include <vector>
#include <stdexcept>
#include <array>
#include <gen/distributions/normal.h>
#include <gen/utils/randutils.h>

// TODO update needs to take an RNG
// TODO implement a version that uses Immer and supports rejuvenation without copying entire histories

// constraint types

struct NextTimeStepObservations {
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
    double velocity_var;
    double init_x_prior_mean;
    double init_x_prior_var;
    double init_y_prior_mean;
    double init_y_prior_var;
    double init_vx_prior_mean;
    double init_vx_prior_var;
    double init_vy_prior_mean;
    double init_vy_prior_var;
};

// return value difference (unused)

class RetDiff {};
const RetDiff ret_diff{};

// Model and trace interfaces

class Trace;

class Model {
    friend class Trace;
private:
    size_t num_time_steps;
public:
    Model() : num_time_steps_(0) {}
    template <typename RNGType>
    std::pair<Trace,double> generate(const NextTimeStepObservations& constraints, RNGType& rng,
                                     Parameters& parameters, bool prepare_for_gradient) const;
};

struct State {
    const double x;
    const double y;
    const double vx;
    const double vy;
    const double measured_bearing;
};

class Trace {
    friend class Model;
private:
    Model model_;
    const Parameters& parameters_;
    std::vector<State> states_;
    Trace(const Model& model, const Parameters& parameters, std::vector<State> states) :
            model_{model}, parameters_{parameters}, states_{states} {}
public:
    template <typename RNGType>
    std::tuple<double,const EmptyConstraints&,const RetDiff&> update(
            RNGType& rng, const ExtendByOneTimeStep& model_change, const NextTimeStepObservations& constraints,
            bool save_previous, bool prepare_for_gradient);
    void revert();
    Trace fork();
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

State initial_importance_sample(const Parameters& parameters, const NextTimeStepObservations& constraints) {
    double new_x = normal(0.01, 0.01);
    double new_y = normal(0.95, 0.01);
    double new_vx = normal(0.002, 0.01);
    double new_vy = normal(-0.013, 0.01);
    double bearing = std::atan2(new_y, new_x);
    double new_measured_bearing = constraints.measured_bearing;
    double log_weight = normal_logpdf(new_measured_bearing, bearing, parameters.measurement_noise);
    State new_state{new_x, new_y, new_vx, new_vy, new_measured_bearing};
    return {new_state, log_weight};
}

template <typename RNGType>
std::pair<Trace,double> Model::generate(const NextTimeStepObservations& constraints, RNGType& rng,
                                        Parameters& parameters, bool prepare_for_gradient) const {
    if (prepare_for_gradient)
        throw std::logic_error("prepare_for_gradient in generate not implemented");
    if (num_time_steps_ != 0)
        throw std::logic_error("generate not implemented for num_time_steps != 0");
    // sample initial x, y, vx, vy from the prior and compute importance weight
    auto [state, log_weight] = initial_importance_sample(parameters, constraints.measured_bearing);
    Trace trace {*this, parameters, {state}};
    return {trace, log_weight};
}

State incremental_importance_sample(const Parameters& parameters, const NextTimeStepObservations& constraints,
                                    const State& state) {
    double new_vx = normal(state.vx, parameters.velocity_stdev);
    double new_vy = normal(state.vy, parameters.velocity_stdev);
    double new_x = state.x + new_vx;
    double new_y = state.y + new_vy;
    double bearing = std::atan2(new_y, new_x);
    double new_measured_bearing = constraints.measured_bearing;
    double log_weight = normal_logpdf(new_measured_bearing, bearing, parameters.measurement_noise);
    State new_state{new_x, new_y, new_vx, new_vy, new_measured_bearing};
    return {new_state, log_weight};
}

template <typename RNGType>
std::tuple<double,const EmptyConstraints&,const RetDiff&> Trace::update(
        RNGType& rng, const ExtendByOneTimeStep& model_change, const NextTimeStepObservations& constraints,
        bool save_previous, bool prepare_for_gradient) {
    if (prepare_for_gradient)
        throw std::logic_error("prepare_for_gradient in update not implemented");
    if (save_previous)
        throw std::logic_error("save_previous in update not implemented");
    auto [new_state, log_weight] = incremental_importance_sample(parameters, constraints.measured_bearing, states_.back());
    states_.emplace_back(new_state);
    return {log_weight, empty_constraints, ret_diff};
}

State extend_without_observation(const Parameters& parameters, const State& state) {
    double new_vx = normal(state.vx, parameters.velocity_stdev);
    double new_vy = normal(state.vy, parameters.velocity_stdev);
    double new_x = state.x + new_vx;
    double new_y = state.y + new_vy;
    double bearing = std::atan2(new_y, new_x);
    double new_measured_bearing = normal(bearing, parameters.measurement_noise);
    State new_state{new_x, new_y, new_vx, new_vy, new_measured_bearing};
    return new_state;
}

template <typename RNGType>
std::tuple<double,const EmptyConstraints&,const RetDiff&> Trace::update(
        RNGType& rng, const ExtendByOneTimeStep& model_change, const EmptyConstraints& constraints,
        bool save_previous, bool prepare_for_gradient) {
    if (prepare_for_gradient)
        throw std::logic_error("prepare_for_gradient in update not implemented");
    if (save_previous)
        throw std::logic_error("save_previous in update not implemented");
    auto new_state = extend_without_observation(parameters, states_.back());
    states_.emplace_back(new_state);
    return {0.0, empty_constraints, ret_diff};
}

void Trace::revert() {
    throw std::logic_error("revert not implemented");
}

Trace Trace::fork() {
    // this copies the states (the entire history)
    Trace new_trace{model_, parameters_, states_};
    return new_trace; // copy elision
}

// Example

int main(int argc, char* argv[]) {

    Parameters parameters;
    parameters.measurement_noise = 0.005;
    parameters.velocity_var = 0.005;
    parameters.init_x_prior_mean = 0.01;
    parameters.init_x_prior_var = 0.01;
    parameters.init_y_prior_mean = 0.95;
    parameters.init_y_prior_var = 0.01;
    parameters.init_vx_prior_mean = 0.002;
    parameters.init_vx_prior_var = 0.01;
    parameters.init_vy_prior_mean = -0.013;
    parameters.init_vy_prior_var = 0.01;

    Model model{};


    // TODO simulate data for some seed

    // TODO then run the particele filter

    // TODO spit out all data for plotting

    size_t num_particles = 100;


}