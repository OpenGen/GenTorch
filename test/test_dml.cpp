#include <torch/torch.h>

#include "interfaces/address.h"
#include "interfaces/trie.h"
#include "dml/dml.h"
#include "distributions/normal.h"

using torch::Tensor;
using torch::TensorOptions;
using torch::tensor;
using distributions::normal::Normal;

bool tensor_scalar_bool(Tensor a) {
    if (a.ndimension() != 0) {
        throw std::logic_error("bool error");
    }
    return *(a.data_ptr<bool>());
}

class Foo;

class Foo : public DMLGenFn<Foo, std::pair<Tensor,int>, Tensor> {
public:
    explicit Foo(Tensor z, int depth) : DMLGenFn<Foo, args_type, return_type>({z, depth}) {}

    template <typename Tracer>
    return_type exec(Tracer& tracer) const {
        const auto& [z, depth] = get_args();
        const auto x = tensor(0.0);
        const auto y = tensor(1.0) + z + x;
        const auto a = tensor(0.0);
        const Tensor z1 = tracer.call(Address{"z1"}, Normal(a, tensor(1.0)));
        const Tensor z2 = tracer.call(Address{"z2"}, Normal(z1, tensor(0.1)));
        Tensor result;
        if (tensor_scalar_bool(z1 < 0.0)) {
            const Tensor z3 = tracer.call(Address{"recursive"}, Foo(z2, depth + 1));
            result = x + z1 + y + z2 + z3;
        } else {
            result = x + z1 + y + z2;
        }
        return result;// + w;
    }
};


void do_simulate(int idx, int n, std::vector<double>& scores) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    double score = 0.0;
    for (int j = 0; j < n; j++) {
        Tensor z = tensor(1.0, TensorOptions().dtype(torch::kFloat64));
        const auto model = Foo(z, 0);
        const auto trace = model.simulate(gen);
        score += trace.get_score();
    }
    scores[idx] = score;
}

void do_generate(int idx, int n, std::vector<double>& scores, const Trie& constraints) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    double total_log_weight = 0.0;
    for (int j = 0; j < n; j++) {
        Tensor z = tensor(1.0, TensorOptions().dtype(torch::kFloat64));
        const auto model = Foo(z, 0);
        auto trace_and_log_weight = model.generate(gen, constraints);
        auto trace = std::move(trace_and_log_weight.first);
        double log_weight = trace_and_log_weight.second;
        total_log_weight += log_weight;
    }
    scores[idx] = total_log_weight;
}

void multithreaded_simulate() {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    int n = 1000;
    int n_threads = 4;
    std::vector<double> scores(n_threads, 0.0);
    std::vector<std::thread> threads;
    for (int i = 0; i < n_threads; i++) {
        threads.push_back(std::thread(do_simulate, i, n, std::ref(scores)));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count()/1000000.0 << " seconds" << std::endl;
    std::cout << scores << std::endl;
}


void multithreaded_generate() {
    Trie constraints {};
    constraints.set_value(Address{"z2"}, tensor(1.0));
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    int n = 1000;
    int n_threads = 4;
    std::vector<double> scores(n_threads, 0.0);
    std::vector<std::thread> threads;
    for (int i = 0; i < n_threads; i++) {
        do_generate(i, n, scores, constraints);
        threads.push_back(
                std::thread(do_generate, i, n, std::ref(scores), std::ref(constraints)));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count()/1000000.0 << " seconds" << std::endl;
    std::cout << scores << std::endl;
}


int main() {

    multithreaded_simulate();
    multithreaded_generate();

}