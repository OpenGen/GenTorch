#include <torch/torch.h>
#include <gen/address.h>
#include <gen/trie.h>
#include <gen/dml.h>
#include <gen/parameters.h>
#include <gen/distributions/normal.h>
#include <gen/distributions/uniform_continuous.h>

using torch::Tensor, torch::tensor;
using std::vector, std::cout, std::endl;
using gen::dml::DMLGenFn;
using gen::EmptyModule;
using gen::distributions::normal::Normal;
using gen::distributions::uniform_continuous::UniformContinuous;


namespace gen::examples::importance_sampling {

    typedef nullptr_t Nothing;
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

    void do_chunk(size_t thread_idx, std::vector<double>& result, size_t num_samples) {
        EmptyModule parameters;
        std::random_device rd{};
        std::mt19937 rng{rd()};
        auto model = Model();
        auto proposal = Proposal();
        double total = 0;
        for (size_t i = 0; i < num_samples; i++) {
            auto proposal_trace = proposal.simulate(rng, parameters, false);
            auto constraints = proposal_trace.get_choice_trie();
            auto model_trace_and_log_weight = model.generate(rng, parameters, constraints, false);
            auto& trace = model_trace_and_log_weight.first;
            double log_weight = model_trace_and_log_weight.second - proposal_trace.get_score();
            const Tensor& retval = trace.get_return_value();
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

    double estimate(size_t num_threads, size_t num_samples_per_thread) {
        std::vector<std::thread> threads {num_threads};
        std::vector<double> result(num_threads);
        for (size_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
            threads[thread_idx] = std::thread(do_chunk, thread_idx, std::ref(result), num_samples_per_thread);
        }
        for (auto& thread: threads) {
            thread.join();
        }
        return 4.0 * sum(result) / num_threads;
    }

}

const std::string usage = "Usage: ./importance_sampling <num_threads> <num_samples_per_thread>";

int main(int argc, char* argv[]) {
    torch::set_num_interop_threads(1);
    torch::set_num_threads(1);
    if (argc != 3) {
        throw std::invalid_argument(usage);
    }
    size_t num_threads;
    size_t num_samples_per_thread;
    try {
        num_threads = std::atoi(argv[1]);
        num_samples_per_thread = std::atoi(argv[2]);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(usage);
    }
    cout << "num_threads: " << num_threads << ", num_samples_per_thread: " << num_samples_per_thread << endl;
    double result = gen::examples::importance_sampling::estimate(num_threads, num_samples_per_thread);
    cout << "result: " << result << endl;
}