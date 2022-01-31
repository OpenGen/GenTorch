#include <torch/torch.h>

#include <gen/address.h>
#include <gen/dml.h>
#include <gen/parameters.h>
#include <gen/distributions/normal.h>
#include <gen/utils/randutils.h>
#include <gen/sgd.h>
#include <gen/conversions.h>


using torch::Tensor, torch::tensor;
using std::vector, std::cout, std::endl;
using gen::dml::DMLGenFn;
using gen::EmptyModule;
using gen::distributions::normal::Normal;
using randutils::seed_seq_fe128;


namespace gen::examples::sgd {

    EmptyModule empty_module{};

    struct GroundTruth;

    struct GroundTruth : public DMLGenFn<GroundTruth, Tensor, Nothing, EmptyModule> {
        explicit GroundTruth(Tensor z) : DMLGenFn<M, A, R, P>(z) {};

        template<typename T>
        return_type forward(T &tracer) const {
            const auto &z = tracer.get_args();
            auto x = tracer.call({"x"}, Normal(z.index({0}), tensor(1.0)));
            auto y = tracer.call({"y"}, Normal(z.index({1}), tensor(1.0)));
            return nothing;
        }
    };

    struct ModelModule : public gen::Parameters {
        torch::nn::Linear fc1{nullptr};
        torch::nn::Linear fc2{nullptr};

        ModelModule() {
            c10::InferenceMode guard{false};
            fc1 = register_torch_module("fc1", torch::nn::Linear(2, 50));
            fc2 = register_torch_module("fc2", torch::nn::Linear(50, 2));
        }
    };

    struct Model;

    struct Model : public DMLGenFn<Model, Tensor, Nothing, ModelModule> {
        explicit Model(Tensor z) : DMLGenFn<M, A, R, P>(z) {};

        template<typename T>
        return_type forward(T &tracer) const {
            const auto &z = tracer.get_args();
            if (tracer.prepare_for_gradients()) {
                assert(!c10::InferenceMode::is_enabled());
                assert(!z.is_inference());
            }
            auto &parameters = tracer.get_parameters();
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
        explicit ProblemGenerator() : DMLGenFn<M, A, R, P>(nothing) {};
        template<typename T>
        return_type forward(T &tracer) const {
            auto x = tracer.call({"z1"}, Normal(tensor(0.0), tensor(1.0)));
            auto y = tracer.call({"z2"}, Normal(tensor(0.0), tensor(1.0)));
            Tensor z = torch::stack({x, y});
            assert(z.sizes().equals({2}));
            return z;
        }
    };

    Tensor generate_z(std::mt19937 &rng) {
        static ProblemGenerator generator{};
        // NOTE this copies th the return value
        return std::any_cast<Tensor>(generator.simulate(rng, empty_module, false).get_return_value());
    }

    typedef std::tuple<Tensor, Tensor, Tensor> datum_t;

    std::vector<datum_t> generate_training_data(std::mt19937 &rng, size_t n) {
        // generate training data from it
        std::cout << "generating training data.." << std::endl;
        std::vector<datum_t> data;
        for (size_t i = 0; i < n; i++) {
            Tensor z = generate_z(rng);
            GroundTruth model{z};
            auto ground_truth_trace = model.simulate(rng, empty_module, false);
            ChoiceTrie choices = ground_truth_trace.get_choice_trie();
            auto x = std::any_cast<Tensor>(choices.get_value({"x"}));
            auto y = std::any_cast<Tensor>(choices.get_value({"y"}));
            data.emplace_back(std::make_tuple(x, y, z));
        }
        return data;
    }

}

int main(int argc, char* argv[]) {

    using namespace gen::examples::sgd;
    torch::set_num_interop_threads(1);
    torch::set_num_threads(1);
    c10::InferenceMode guard {true};

    static const std::string usage = "Usage: ./sgd <minibatch_size> <num_threads> <num_iters> <seed>";
    if (argc != 5) {
        throw std::invalid_argument(usage);
    }
    size_t minibatch_size;
    size_t num_threads;
    size_t num_iters;
    uint32_t seed;
    try {
        minibatch_size = std::atoi(argv[1]);
        num_threads = std::atoi(argv[2]);
        num_iters = std::atoi(argv[3]);
        seed = std::atoi(argv[4]);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(usage);
    }
    cout << "minibatch_size: " << minibatch_size << endl;
    cout << " num_threads: " << num_threads << endl;
    cout << " num_iters: " << num_iters << endl;
    cout << "seed: " << seed << endl;

    std::random_device rd{};
    seed_seq_fe128 seed_seq{rd()};
    std::mt19937 rng(seed_seq);

    auto unpack_datum_ground_truth = [](const datum_t &datum) -> std::pair<GroundTruth, gen::ChoiceTrie> {
        const auto& [x, y, z] = datum;
        GroundTruth model{z};
        gen::ChoiceTrie constraints;
        constraints.set_value({"x"}, x);
        constraints.set_value({"y"}, y);
        return {model, constraints};
    };

    auto unpack_datum = [](const datum_t &datum) -> std::pair<Model, gen::ChoiceTrie> {
        const auto& [x, y, z] = datum;
        Model model{z};
        gen::ChoiceTrie constraints;
        constraints.set_value({"x"}, x);
        constraints.set_value({"y"}, y);
        return {model, constraints};
    };

    size_t num_train = 1000;
    auto data = generate_training_data(rng, num_train);

    // evaluate the ground truth as a baseline
    double ground_truth_objective = gen::sgd::estimate_objective(rng, empty_module, data, unpack_datum_ground_truth);
    std::cout << "ground truth parameters objective: " << ground_truth_objective << std::endl;

    // initialize random model parameters
    ModelModule parameters {};

    auto evaluate = [&parameters,&data,&unpack_datum,&rng](size_t iter) -> double {
        double objective = gen::sgd::estimate_objective(rng, parameters, data, unpack_datum);
        std::cout << "iter " << iter << "; objective: " << objective << std::endl;
        return objective;
    };

    double learning_rate = 0.000001;
    torch::optim::SGD sgd {
            parameters.all_parameters(),
            torch::optim::SGDOptions(learning_rate).dampening(0.0).momentum(0.0)};

    size_t iter = 0;

    auto callback = [&iter,&evaluate,num_iters,&sgd](const std::vector<size_t>& minibatch) -> bool {
        sgd.step();
        sgd.zero_grad();
        if (iter % 100 == 0) {
            evaluate(iter);
        }
        return (iter++) == num_iters - 1;
    };

    cout << "doing multi-threaded training" << endl;
    evaluate(iter++);
    gen::sgd::train_supervised(parameters, callback, data, unpack_datum, minibatch_size, num_threads, seed_seq);
}
