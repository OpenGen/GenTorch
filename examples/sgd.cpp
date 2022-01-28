#include <barrier>

#include <torch/torch.h>
#include <gen/address.h>
#include <gen/dml.h>
#include <gen/parameters.h>
#include <gen/distributions/normal.h>

using torch::Tensor, torch::tensor;
using std::vector, std::cout, std::endl;
using gen::dml::DMLGenFn;
using gen::EmptyModule;
using gen::distributions::normal::Normal;

// task: see how SGD (with e.g. 64 minibatches) scales on a c6 instance, as function of the number of threads

// what are we learning?

// let's learn to predict (x, y) given z (all reals)

// we will sample z ~ Normal(0, 1) when generating the training data

// we will use a ground truth Module that has the right parameters, and we will
// use another copy of the module that we train to get the same parameters

// TODO consider renaming gen::Module to gen::Parameters,
//  since gen::Module does not have a forward like most torch::nn::Modules

namespace gen::examples::sgd {

    typedef std::nullptr_t Nothing;
    constexpr Nothing nothing = nullptr;
    EmptyModule empty_module {};

    struct GroundTruth;
    struct GroundTruth : public DMLGenFn<GroundTruth, Tensor, Nothing, EmptyModule> {
        explicit GroundTruth(Tensor z) : DMLGenFn<M,A,R,P>(z) {};
        template <typename T>
        return_type forward(T& tracer) const {
            const auto& z = tracer.get_args();
            auto x = tracer.call({"x"}, Normal(z.index({0}), tensor(1.0)));
            auto y = tracer.call({"y"}, Normal(z.index({1}), tensor(1.0)));
            return nothing;
        }
    };

    struct ModelModule : public gen::Module {
        torch::nn::Linear fc1 {nullptr};
        torch::nn::Linear fc2 {nullptr};
        ModelModule() {
            c10::InferenceMode guard {false};
            fc1 = register_torch_module("fc1", torch::nn::Linear(2, 50));
            fc2 = register_torch_module("fc2", torch::nn::Linear(50, 2));
        }
    };

    struct Model;
    struct Model : public DMLGenFn<Model, Tensor, Nothing, ModelModule> {
        explicit Model(Tensor z) : DMLGenFn<M,A,R,P>(z) {};
        template <typename T>
        return_type forward(T& tracer) const {
            const auto& z = tracer.get_args();
            if (tracer.prepare_for_gradients()) {
                assert(!c10::InferenceMode::is_enabled());
                assert(!z.is_inference());
            }
            auto& parameters = tracer.get_parameters();
            auto h1 = parameters.fc1->forward(z);
            assert(h1.sizes().equals({50}));
            auto h2 = parameters.fc2->forward(h1);
            assert(h2.sizes().equals({2}));
            auto x = tracer.call({"x"}, Normal(h2.index({0}), tensor(1.0)));
            auto y = tracer.call({"y"}, Normal(h2.index({1}), tensor(1.0)));
            return nothing;
        }
    };

    struct ProblemGenerator;
    struct ProblemGenerator : public DMLGenFn<ProblemGenerator, Nothing, Tensor, EmptyModule> {
        explicit ProblemGenerator() : DMLGenFn<M,A,R,P>(nothing) {};
        template <typename T>
        return_type forward(T& tracer) const {
            auto x = tracer.call({"z1"}, Normal(tensor(0.0), tensor(1.0)));
            auto y = tracer.call({"z2"}, Normal(tensor(0.0), tensor(1.0)));
            Tensor z = torch::stack({x, y});
            assert(z.sizes().equals({2}));
            return z;
        }
    };

    Tensor generate_z(std::mt19937& rng) {
        static ProblemGenerator generator {};
        // NOTE this copies th the return value
        return std::any_cast<Tensor>(generator.simulate(rng, empty_module, false).get_return_value());
    }

    typedef std::vector<std::tuple<Tensor,Tensor,Tensor>> dataset_t;

    template <typename GenFnType, typename ParametersType>
    double estimate_objective(std::mt19937& rng, ParametersType& parameters, const dataset_t& data) {
        double total = 0.0;
        for (const auto& [x, y, z] : data) {
            GenFnType model {z};
            ChoiceTrie constraints;
            constraints.set_value({"x"}, x);
            constraints.set_value({"y"}, y);
            auto trace_and_log_weight = model.generate(rng, parameters, constraints, false);
            total += trace_and_log_weight.second;
        }
        return total / data.size();
    }


    dataset_t generate_training_data(std::mt19937& rng, size_t n) {
        // generate training data from it
        std::cout << "generating training data.." << std::endl;
        dataset_t data;
        for (size_t i = 0; i < n; i++) {
            Tensor z = generate_z(rng);
            GroundTruth model {z};
            auto ground_truth_trace = model.simulate(rng, empty_module, false);
            ChoiceTrie choices = ground_truth_trace.get_choice_trie();
            auto x = std::any_cast<Tensor>(choices.get_value({"x"}));
            auto y = std::any_cast<Tensor>(choices.get_value({"y"}));
            data.emplace_back(std::make_tuple(x, y, z));
        }
        return data;
    }

    std::vector<size_t> generate_minibatch(std::mt19937& rng, size_t n, size_t minibatch_size) {
        std::vector<size_t> minibatch(minibatch_size);
        std::uniform_int_distribution<size_t> dist {0, n-1};
        for (int i = 0; i < minibatch_size; i++) {
            minibatch[i] = dist(rng);
        }
        return minibatch;
    }

    void single_threaded_gradient_estimation(std::mt19937& rng, ModelModule& parameters, const dataset_t& data,
                                             size_t minibatch_size, gen::GradientAccumulator& accum) {
        std::vector<size_t> minibatch = generate_minibatch(rng, data.size(), minibatch_size);
        double scaler = 1.0 / minibatch_size;
        for (size_t i : minibatch) {

            // obtain datum
            auto [x, y, z] = data[i];

            // obtain trace
            Model model {z};
            ChoiceTrie constraints;
            constraints.set_value({"x"}, x);
            constraints.set_value({"y"}, y);
            auto trace_and_log_weight = model.generate(rng, parameters, constraints, true);

            // compute gradients for this datum
            trace_and_log_weight.first.gradients(nothing, scaler, accum);
        }
        accum.update_module_gradients();
    }

    void single_threaded_sgd_training(std::mt19937& rng, ModelModule& parameters, const dataset_t& data,
                                      size_t iters, size_t minibatch_size, double learning_rate) {
        torch::optim::SGD sgd {parameters.all_parameters(), torch::optim::SGDOptions(learning_rate).
                                                                dampening(0.0).
                                                                momentum(0.0)};
        gen::GradientAccumulator accum {parameters};

        for (size_t iter = 0; iter < iters; iter++) {

            single_threaded_gradient_estimation(rng, parameters, data, minibatch_size, accum);
            sgd.step();

            if (iter % 100 == 0) {
                // evaluate
                double objective = estimate_objective<Model>(rng, parameters, data);
                cout << "iter: " << iter << ", objective: " << objective << endl;
            }

        }
    }

    void multi_threaded_sgd_training(std::mt19937& rng, ModelModule& parameters, const dataset_t& data,
                                      const size_t iters, const size_t minibatch_size, const double learning_rate,
                                      const size_t num_threads) {
        assert(c10::InferenceMode::is_enabled());
        torch::optim::SGD sgd {
                parameters.all_parameters(),
                torch::optim::SGDOptions(learning_rate).dampening(0.0).momentum(0.0)};

        // one accumulator per thread
        std::vector<gen::GradientAccumulator> accums;
        for (int i = 0; i < num_threads; i++) {
            accums.emplace_back(gen::GradientAccumulator{parameters});
        }

        const size_t n = data.size();
        const double scaler = 1.0 / minibatch_size;
        size_t iter = 0;
        bool done = false;

        // initial minibatch
        std::vector<size_t> minibatch = generate_minibatch(rng, data.size(), minibatch_size);

        auto iteration_serial_work = [iters,n,&iter,&done,&accums,&rng,&minibatch,&parameters,&data,&sgd]() {
            c10::InferenceMode guard {true};
            assert(iter < iters);

            if (iter % 100 == 0) {
                // evaluate
                double objective = estimate_objective<Model>(rng, parameters, data);
                cout << "iter: " << iter << ", objective: " << objective << endl;
            }

            for (auto& accum : accums) {
                accum.update_module_gradients();
            }

            sgd.step();

            // are we done? (NOTE: this can be replaced with some other temination condition)
            if (iter == iters-1) {
                done = true;
            } else {
                // compute minibatch for next iteration
                minibatch = generate_minibatch(rng, n, minibatch.size());
            }
            iter++;
        };

        std::barrier sync_point(num_threads, iteration_serial_work);

        auto gradients = [&sync_point,
                          &minibatch = std::as_const(minibatch),
                          &data = std::as_const(data),
                          &parameters,
                          scaler,
                          &done = std::as_const(done)](GradientAccumulator& accum, size_t start, size_t stop) {
            c10::InferenceMode guard {true};

            // TODO get this in a deterministic way from the input RNG
            std::random_device rd{};
            std::mt19937 rng{rd()};

            while (!done) {
                // do another iteration

                // process our portion of the minibatch
                for (size_t i = start; i < stop; i++) {

                    // obtain datum
                    auto[x, y, z] = data[minibatch[i]];

                    // obtain trace
                    Model model{z};
                    ChoiceTrie constraints;
                    constraints.set_value({"x"}, x);
                    constraints.set_value({"y"}, y);
                    auto trace_and_log_weight = model.generate(rng, parameters, constraints, true);

                    // compute gradients for this datum
                    trace_and_log_weight.first.gradients(nothing, scaler, accum);
                }

                sync_point.arrive_and_wait();
            }
        };

        // start threads
        std::vector<std::thread> threads;
        size_t start = 0;
        size_t stop;
        for (int i = 0; i < num_threads; i++) {
            size_t k = minibatch_size / num_threads;
            size_t rem = minibatch_size % num_threads;
            size_t block_size;
            if (i < rem) {
                block_size = k + 1;
            } else {
                block_size = k;
            }
            stop = start + block_size;
            threads.emplace_back(gradients, std::ref(accums[i]), start, stop);
            start = stop;
        }



        // join threads
        for (auto& thread : threads) {
            thread.join();
        }
    }

}



int main(int argc, char* argv[]) {
    using namespace gen::examples::sgd;
    torch::set_num_interop_threads(1);
    torch::set_num_threads(1);
    c10::InferenceMode guard {true};

    static const std::string usage = "Usage: ./sgd <minibatch_size> <num_threads> <num_iters>";
    if (argc != 4) {
        throw std::invalid_argument(usage);
    }
    size_t minibatch_size;
    size_t num_threads;
    size_t num_iters;
    try {
        minibatch_size = std::atol(argv[1]);
        num_threads = std::atol(argv[2]);
        num_iters = std::atol(argv[3]);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(usage);
    }
    cout << "minibatch_size: " << minibatch_size << " num_threads: " << num_threads << endl;

    std::random_device rd{};
    std::mt19937 rng{rd()};

    size_t num_train = 1000;
    auto data = generate_training_data(rng, num_train);

    // evaluate a few tiomes
    double ground_truth_objective = estimate_objective<GroundTruth>(rng, empty_module, data);
    std::cout << "ground truth parameters objective: " << ground_truth_objective << std::endl;

    // initialize fresh parameters
    ModelModule parameters {};
    double initial = estimate_objective<Model>(rng, parameters, data);
    std::cout << "initial objective for random parameters: " << initial << std::endl;

    // do training
//    single_threaded_sgd_training(rng, parameters, data, 100000, minibatch_size, 0.0000001);

    multi_threaded_sgd_training(rng, parameters, data, num_iters, minibatch_size, 0.0000001, num_threads);

}
