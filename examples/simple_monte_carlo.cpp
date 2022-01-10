#include <torch/torch.h>
#include <gen/address.h>
#include <gen/trie.h>
#include <gen/dml.h>
#include <gen/parameters.h>
#include <gen/distributions/uniform_continuous.h>

using torch::Tensor, torch::tensor;
using std::vector, std::cout, std::endl;
using gen::dml::DMLGenFn;
using gen::EmptyModule;
using gen::distributions::uniform_continuous::UniformContinuous;

namespace gen::examples::simple_monte_carlo {

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

    void do_chunk(size_t thread_idx, std::vector<double>& result, size_t num_samples) {
        EmptyModule parameters;
        std::random_device rd{};
        std::mt19937 rng{rd()};
        auto model = Model();
        size_t num_inside_circle = 0;
        for (size_t i = 0; i < num_samples; i++) {
            auto trace = model.simulate(rng, parameters, false);
            Tensor retval = std::any_cast<Tensor>(trace.get_return_value());
            float radius = *retval.data_ptr<float>();
            if (radius < 1.0) {
                num_inside_circle++;
            }
        }
        result[thread_idx] = static_cast<double>(num_inside_circle) / num_samples;
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

const std::string usage = "Usage: ./simple_monte_carlo <num_threads> <num_samples_per_thread>";

int main(int argc, char* argv[]) {
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
    double result = gen::examples::simple_monte_carlo::estimate(num_threads, num_samples_per_thread);
    cout << "result: " << result << endl;
}