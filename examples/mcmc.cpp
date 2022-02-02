#include <gen/still/mcmc.h>

#include <utility>
#include <vector>
#include <stdexcept>
#include <array>
#include <gen/distributions/normal.h>
#include <gen/utils/randutils.h>

#include <Eigen/Dense>


// TODO support parameters
// TODO support subset of addresses

// ********************
// *** Design notes ***
// ********************

/*
 * Each trace type may be compatible with multiple choice buffer types, and multiple selection types.
 *
 * The choice buffer type that is returned from update is the same as the choice buffer type passed into it.
 *
 * For a given selection type, the choice buffer returned by choice_gradients(selection) and the choice buffer returned
 * by get_choices(selection) are identical.
 *
 * Some of the MCMC operators accept a buffer as input. Where should they get this buffer? They could get it from
 * get_choices().
 *
 * Q: should we return const references or const pointers?
 *
 *
 */

// **************************************************
// *** double vector with element-wise operations ***
// **************************************************
//
//// TODO use eigen instead?
//
//template <typename base_t, size_t dimension>
//class ChoiceBuffer {
//private:
//    std::array<base_t,dimension> values_;
//public:
//    typedef typename std::array<base_t,dimension>::iterator iterator;
//    typedef typename std::array<base_t,dimension>::const_iterator const_iterator;
//
//    // default constructor (non-initialized values)
//    ChoiceBuffer() = default;
//
//    // copy constructor
//    ChoiceBuffer(const ChoiceBuffer& other) = default;
//
//    // move constructor
//    ChoiceBuffer(ChoiceBuffer&& other) noexcept = default;
//
//    // copy assignment
//    ChoiceBuffer& operator=(const ChoiceBuffer& other) = default;
//
//    // move assignment
//    ChoiceBuffer& operator=(ChoiceBuffer&& other) noexcept = default;
//
//    // iterators
//    [[nodiscard]] iterator begin() noexcept {
//        return values_.begin();
//    }
//
//    [[nodiscard]] iterator end() noexcept {
//        return values_.end();
//    }
//
//    [[nodiscard]] const_iterator begin() const noexcept {
//        return values_.cbegin();
//    }
//
//    [[nodiscard]] const_iterator end() const noexcept {
//        return values_.cend();
//    }
//
//    [[nodiscard]] const_iterator cbegin() const noexcept {
//        return values_.cbegin();
//    }
//
//    [[nodiscard]] const_iterator cend() const noexcept {
//        return values_.cend();
//    }
//
//    // in-place elementwise operations
//    ChoiceBuffer& operator+=(const ChoiceBuffer& other) {
//        for (size_t i = 0; i < dimension; i++)
//            values_[i] += other.values_[i];
//        return *this;
//    }
//    ChoiceBuffer& operator-=(const ChoiceBuffer& other) {
//        for (size_t i = 0; i < dimension; i++)
//            values_[i] -= other.values_[i];
//        return *this;
//    }
//    ChoiceBuffer& operator*=(base_t scalar) {
//        for (auto& value : values_)
//            value *= scalar;
//        return *this;
//    }
//
//    // element-wise operations
//    ChoiceBuffer operator+(const ChoiceBuffer& other) {
//        ChoiceBuffer result;
//        for (size_t i = 0; i < dimension; i++)
//            result.values_[i] = values_[i] + other.values_[i];
//        return result; // copy-elision
//    }
//    ChoiceBuffer operator-(const ChoiceBuffer& other) {
//        ChoiceBuffer result;
//        for (size_t i = 0; i < dimension; i++)
//            result.values_[i] = values_[i] - other.values_[i];
//        return result; // copy-elision
//    }
//    ChoiceBuffer operator*(const ChoiceBuffer& other) {
//        ChoiceBuffer result;
//        for (size_t i = 0; i < dimension; i++)
//            result.values_[i] = values_[i] * other.values_[i];
//        return result; // copy-elision
//    }
//    ChoiceBuffer operator/(const ChoiceBuffer& other) {
//        ChoiceBuffer result;
//        for (size_t i = 0; i < dimension; i++)
//            result.values_[i] = values_[i] / other.values_[i];
//        return result; // copy-elision
//    }
//
//    // unary negation
//    ChoiceBuffer operator-() const & {
//        ChoiceBuffer result(*this);
//        for (auto& value : result.values_) {
//            value = -value;
//        }
//        return result; // copy-elision
//    }
//
//    ChoiceBuffer operator-() const && {
//        for (auto& value : values_) {
//            value = -value;
//        }
//        return *this; // copy-elision
//    }
//
//    // TODO implement operations needed by gen/still/mcmc.h
//};
//


// ****************************
// *** Model implementation ***
// ****************************

class LatentsSelection {};
class ObservationsSelection {};

constexpr size_t latent_dimension = 2;
constexpr size_t observation_dimension = 1;

typedef Eigen::Array<double,latent_dimension,1> latent_choices_t;
typedef Eigen::Array<double,latent_dimension,1> observed_choices_t;

// The model is just a standard multivariate normal distribution over 3 dimensions
// The log-density is:
// The gradient of the log-density is:



// our model is not currently parameterized
class Parameters { };
class GradientAccumulator { };
class DiffType {};
class ModelDiff {}; // note: our model is not parametrized.


class Trace;

class Model {

private:
public:

    // simulate into a new trace object
    template <typename RNGType>
    Trace simulate(RNGType& rng, bool prepare_for_gradient=false) const;

    // simulate into an existing trace object (overwriting existing contents)
    template <typename RNGType>
    void simulate(Trace& trace, RNGType& rng, bool prepare_for_gradient=false) const;

    // generate into a new trace object
    template <typename RNGType>
    std::pair<Trace,double> generate(const observed_choices_t& constraints,
                                     RNGType& rng, bool prepare_for_gradient=false) const;

    // generate into an existing trace object (overwriting existing contents)
    template <typename RNGType>
    double generate(Trace& trace, const observed_choices_t& constraints,
                    RNGType& rng, bool prepare_for_gradient=false) const;
};

class Trace {
    friend class Model;
private:
    double score_;
    latent_choices_t latents_;
    observed_choices_t observations_;
    latent_choices_t alternate_latents_;
    latent_choices_t latent_gradient_;
    bool can_be_reverted_;
    bool gradients_computed_;
    DiffType diff_;
private:
    // initialize trace without precomputed gradient
    Trace(double score, latent_choices_t&& latents, observed_choices_t observations) :
        score_{score}, latents_{latents}, observations_{observations},
        can_be_reverted_{false}, gradients_computed_{false} {}
    // initialize trace with gradient precomputed
    Trace(double score, latent_choices_t&& latents, observed_choices_t observations,
          latent_choices_t&& latent_gradient) :
        score_{score}, latents_{latents}, observations_{observations},
        latent_gradient_{latent_gradient},
        can_be_reverted_{false}, gradients_computed_{true} {}
public:
    [[nodiscard]] double score() const;
    [[nodiscard]] const latent_choices_t& choices(const LatentsSelection& selection) const;
    [[nodiscard]] const observed_choices_t& choices(const ObservationsSelection& selection) const;
    const latent_choices_t& choice_gradients(const LatentsSelection& selection);
    std::tuple<double, const latent_choices_t&, const DiffType&> update(const latent_choices_t& values,
                                                                        bool save_previous, bool prepare_for_gradient);
    void revert();
    // TODO implement 'fork'
};


// ****************************
// *** Model implementation ***
// ****************************

// TODO replace these with a more interesting distribution..

template <typename RNGType>
double sample_joint(latent_choices_t& latents, observed_choices_t& observations, RNGType& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    double log_density;
    for (auto& value : latents) {
        value = dist(rng);
        log_density += gen::distributions::normal::log_density(0.0, 1.0, value);
    }
    for (auto& value : observations) {
        value = dist(rng);
        log_density += gen::distributions::normal::log_density(0.0, 1.0, value);
    }
    return log_density;
}

template <typename RNGType>
std::pair<double,double> sample_latents(latent_choices_t& latents, const observed_choices_t& observations,
                                        RNGType& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    double log_weight = 0.0;
    double log_density;
    for (auto& value: latents) {
        value = dist(rng);
        log_density += gen::distributions::normal::log_density(0.0, 1.0, value);;
    }
    for (const auto& value : observations) {
        double increment = gen::distributions::normal::log_density(0.0, 1.0, value);
        log_weight += increment;
        log_density += increment;
    }
    return {log_density, log_weight};
}

double logpdf(const latent_choices_t& latents, const observed_choices_t& observations) {
    double result = 0.0;
    for (const auto& x : latents)
        result += gen::distributions::normal::log_density(0.0, 1.0, x);
    for (const auto& x : observations)
        result += gen::distributions::normal::log_density(0.0, 1.0, x);
    return result;
}

void logpdf_grad(latent_choices_t& latent_gradient, const latent_choices_t& latents,
                 const observed_choices_t& observations) {
    // gradient wrt value x is -x
    latent_gradient = -latents;
}

template <typename RNGType>
Trace Model::simulate(RNGType& rng, bool prepare_for_gradient) const {
    latent_choices_t latents;
    observed_choices_t observations;
    double log_density = sample_joint(latents, observations, rng);
    if (prepare_for_gradient) {
        latent_choices_t latent_gradient;
        logpdf_grad(latent_gradient, latents, observations);
        return Trace(log_density, std::move(latents), std::move(observations), std::move(latent_gradient));
    } else {
        return Trace(log_density, std::move(latents), std::move(observations));
    }
}

template <typename RNGType>
void Model::simulate(Trace& trace, RNGType& rng, bool prepare_for_gradient) const {
    trace.score_ = sample_joint(trace.latents_, trace.observations_, rng);
    if (prepare_for_gradient) {
        logpdf_grad(trace.latent_gradient_, trace.latents_, trace.observations_);
    }
    trace.gradients_computed_ = prepare_for_gradient;
    trace.can_be_reverted_ = false;
}

template <typename RNGType>
std::pair<Trace,double> Model::generate(const observed_choices_t& observations, RNGType& rng,
                                        bool prepare_for_gradient) const {
    latent_choices_t latents;
    auto [log_density, log_weight] = sample_latents(latents, observations, rng);
    if (prepare_for_gradient) {
        latent_choices_t latent_gradient;
        logpdf_grad(latent_gradient, latents, observations);
        Trace trace {log_density, std::move(latents), observations, std::move(latent_gradient)};
        return {trace, log_weight};
    } else {
        Trace trace {log_density, std::move(latents), observations};
        return {trace, log_weight};
    }
}

template <typename RNGType>
double Model::generate(Trace& trace, const observed_choices_t& observations, RNGType& rng,
                                        bool prepare_for_gradient) const {
    auto [log_density, log_weight] = sample_latents(trace.latents_, observations, rng);
    trace.score_ = log_density;
    trace.observations_ = observations; // copy assignment
    double score = logpdf(trace.latents_, trace.observations_);
    if (prepare_for_gradient) {
        logpdf_grad(trace.latent_gradient_, trace.latents_, trace.observations_);
    }
    trace.gradients_computed_ = prepare_for_gradient;
    trace.can_be_reverted_ = false;
    return log_weight;
}

// ****************************
// *** Trace implementation ***
// ****************************

double Trace::score() const {
    return score_;
}

const latent_choices_t& Trace::choices(const LatentsSelection& selection) const {
    return latents_;
}

const observed_choices_t& Trace::choices(const ObservationsSelection& selection) const {
    return observations_;
}

void Trace::revert() {
    if (!can_be_reverted_)
        throw std::logic_error("log_weight is only available between calls to update and revert");
    can_be_reverted_ = false;
    std::swap(latents_, alternate_latents_);
    gradients_computed_ = false;
}

const latent_choices_t& Trace::choice_gradients(const LatentsSelection& selection) {
    if (!gradients_computed_) {
        logpdf_grad(latent_gradient_, latents_, observations_);
    }
    gradients_computed_ = true;
    return  latent_gradient_;
}

std::tuple<double, const latent_choices_t&, const DiffType&> Trace::update(
        const latent_choices_t& latents, bool save_previous, bool prepare_for_gradient) {
    if (save_previous) {
        std::swap(latents_, alternate_latents_);
        latents_ = latents; // copy assignment
        can_be_reverted_ = true;
    } else {
        latents_ = latents; // copy assignment
        // can_be_reverted_ keeps its previous value
    };
    double new_log_density = logpdf(latents_, observations_);
    double log_weight = new_log_density - score_;
    score_ = new_log_density;
    if (prepare_for_gradient) {
        logpdf_grad(latent_gradient_, latents_, observations_);
    }
    gradients_computed_ = prepare_for_gradient;
    return {log_weight, alternate_latents_, diff_};
}


// *********************
// *** Example usage ***
// *********************


int main() {
    randutils::seed_seq_fe128 seed {1};
    std::mt19937 rng(seed);

    // configuration and buffers for HMC
    size_t hmc_leapfrog_steps = 10;
    double hmc_eps = 0.001;
    LatentsSelection hmc_selection;

    // configuration and buffers for MALA
    double mala_tau = 0.001;
    LatentsSelection mala_selection;

    // define the model (note that there are no arguments)
    Model model;

    // observations
    observed_choices_t observations;
    observations(0) = 1.123;

    // generate initial trace and choice buffers
    auto [trace, log_weight] = model.generate(observations, rng, true);
    latent_choices_t hmc_momenta_buffer {trace.choices(hmc_selection)}; // copy constructor
    latent_choices_t hmc_values_buffer {trace.choices(hmc_selection)}; // copy constructor
    latent_choices_t mala_values_buffer {trace.choices(mala_selection)}; // copy constructor

    // do some MALA and HMC on the latent variables (without allocating any memory inside the loop)
    size_t num_iters = 100;
    for (size_t iter = 0; iter < num_iters; iter++) {
        gen::still::mcmc::hmc(trace, hmc_selection, hmc_leapfrog_steps, hmc_eps,
                              hmc_momenta_buffer, hmc_values_buffer, rng);
        gen::still::mcmc::mala(trace, mala_selection, mala_tau, mala_values_buffer, rng);
    }

    // TODO print the iterates...

}