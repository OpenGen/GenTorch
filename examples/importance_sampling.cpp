#include <torch/torch.h>
#include <gentorch/address.h>
#include <gentorch/dml/dml.h>
#include <gentorch/parameters.h>
#include <gentorch/distributions/normal.h>
#include <gentorch/distributions/uniform_continuous.h>
#include <gentl/util/randutils.h>
#include <gentl/types.h>

// TODO this example is not giving good estimates

using gentl::randutils::seed_seq_fe128;

using torch::Tensor, torch::tensor;
using std::vector, std::cout, std::endl;
using gentorch::dml::DMLGenFn;
using gentorch::EmptyModule;
using gentorch::distributions::normal::Normal;
using gentorch::distributions::uniform_continuous::UniformContinuous;


namespace gentorch::examples::importance_sampling {

    typedef std::nullptr_t Nothing;
    constexpr Nothing nothing = nullptr;

    struct Model;
    struct Model : public DMLGenFn<Model, Nothing, Tensor, EmptyModule> {
        explicit Model() : DMLGenFn<M,A,R,P>(nothing) {};
        template <typename T>
        return_type forward(T& tracer) const {
            auto x = tracer.call({"x"}, UniformContinuous(tensor(-1.0), tensor(1.0)));
            auto y = tracer.call({"y"}, UniformContinuous(tensor(-1.0), tensor(1.0)));
            return torch::norm(torch::stack({x, y}));
        }
    };

    struct Proposal;
    struct Proposal : public DMLGenFn<Proposal, Nothing, Tensor, EmptyModule> {
        explicit Proposal() : DMLGenFn<M,A,R,P>(nothing) {};
        template <typename T>
        return_type forward(T& tracer) const {
            auto x = tracer.call({"x"}, Normal(tensor(0.0), tensor(0.8)));
            auto y = tracer.call({"y"}, Normal(tensor(0.0), tensor(0.8)));
            return torch::norm(torch::stack({x, y}));
        }
    };

    void do_chunk(size_t thread_idx, std::vector<double>& result, seed_seq_fe128& seed_seq, size_t num_samples) {
        c10::InferenceMode guard{true}; // should be unnecessary, since no new Tensors are created
        EmptyModule parameters;
        std::mt19937 rng(seed_seq);
        auto model = Model();
        auto proposal = Proposal();
        double total = 0;
        for (size_t i = 0; i < num_samples; i++) {
            auto proposal_trace = proposal.simulate(rng, parameters, gentl::SimulateOptions());
            auto constraints = proposal_trace->choices();
            auto model_trace_and_log_weight = model.generate(rng, parameters, constraints, gentl::GenerateOptions());
            auto& trace = model_trace_and_log_weight.first;
            double log_weight = model_trace_and_log_weight.second - proposal_trace->score();
            const Tensor& retval = trace->return_value();
            float radius = *retval.data_ptr<float>();
            if (radius < 1.0) {
                total += std::exp(log_weight);
            }
        }
        result[thread_idx] = total / num_samples;
    }

    double sum(const std::vector<double>& arr) {
        double total = 0.0;
        for (double val : arr) {
            total += val;
        }
        return total;
    }

    double estimate(size_t num_threads, size_t num_samples_per_thread, uint32_t seed) {
        seed_seq_fe128 parent_seq = {seed};
        std::vector<seed_seq_fe128> child_seqs;
        std::vector<std::thread> threads {num_threads};
        std::vector<double> result(num_threads);
        for (size_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
            seed_seq_fe128& child_seq = child_seqs.emplace_back(parent_seq.spawn());
            threads[thread_idx] = std::thread(do_chunk, thread_idx, std::ref(result),
                                              std::ref(child_seq), num_samples_per_thread);
        }
        for (auto& thread: threads) {
            thread.join();
        }
        return 4.0 * sum(result) / num_threads;
    }

}

const std::string usage = "Usage: ./importance_sampling <num_threads> <num_samples_per_thread> <seed>";

int main(int argc, char* argv[]) {
    torch::set_num_interop_threads(1);
    torch::set_num_threads(1);
    if (argc != 4) {
        throw std::invalid_argument(usage);
    }
    size_t num_threads;
    size_t num_samples_per_thread;
    uint32_t seed;
    try {
        num_threads = std::atoi(argv[1]);
        num_samples_per_thread = std::atoi(argv[2]);
        seed = std::atoi(argv[3]);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(usage);
    }
    cout << "num_threads: " << num_threads << endl;
    cout << "num_samples_per_thread: " << num_samples_per_thread << endl;
    cout << "seed: " << seed << endl;
    double result = gentorch::examples::importance_sampling::estimate(num_threads, num_samples_per_thread, seed);
    cout << "result: " << result << endl;
}
