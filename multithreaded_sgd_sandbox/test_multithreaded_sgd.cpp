#include <torch/torch.h>
#include <thread>
#include <functional>
#include <memory>
#include <chrono>
#include <random>

using std::cout, std::endl;
using torch::Tensor;
using torch::TensorOptions;
using torch::nn::LSTMCell;
using torch::nn::Linear;

struct Net : public torch::nn::Cloneable<Net> {
    // constructor
    Net(int input_data_dim, int lstm_hidden_dim, int output_data_dim) :
            input_data_dim_(input_data_dim),
            lstm_hidden_dim_(lstm_hidden_dim),
            output_data_dim_(output_data_dim),
            lstm_cell_(nullptr),
            linear_(nullptr) {
        reset();
    }

    // input validators

    void validate_x(const Tensor& x) const {
        // TODO: these would throw an exception and not be assertions in an actual implementation
        assert(x.ndimension() == 2);
        assert(x.size(0) == batch_size_); // batch size of 1
        assert(x.size(1) == input_data_dim_);
    }

    void validate_y(const Tensor& y) const {
        // TODO: these would throw an exception and not be assertions in an actual implementation
        assert(y.ndimension() == 3);
        assert(y.size(1) == batch_size_); // batch size of 1
        assert(y.size(2) == output_data_dim_);
    }

    void validate_hx(const Tensor& hx) const {
        // TODO: these would throw an exception and not be assertions in an actual implementation
        assert(hx.ndimension() == 2);
        assert(hx.size(0) == batch_size_);
        assert(hx.size(1) == lstm_hidden_dim_);
    }

    void validate_cx(const Tensor& cx) const {
        // TODO: these would throw an exception and not be assertions in an actual implementation
        assert(cx.ndimension() == 2);
        assert(cx.size(0) == batch_size_);
        assert(cx.size(1) == lstm_hidden_dim_);
    }

    void validate_linear_layer(const Tensor& linear_layer_output) const {
        // TODO: these would throw an exception and not be assertions in an actual implementation
        assert(linear_layer_output.ndimension() == 2);
        assert(linear_layer_output.size(0) == batch_size_);
        assert(linear_layer_output.size(1) == output_data_dim_+1);
    }

    Tensor per_time_step_continue_loss(const Tensor& linear_layer_output) const {
        // batch_size x output_data_dim
        using namespace torch::indexing;
        Tensor increment = torch::log(torch::sigmoid(linear_layer_output.index({Slice(), 0})));
        assert(increment.ndimension() == 1); // just the batch dimension
        return increment;
    }

    Tensor per_time_step_stop_loss(const Tensor& linear_layer_output) const {
        // batch_size x output_data_dim
        using namespace torch::indexing;
        Tensor increment = torch::log(1.0 - torch::sigmoid(linear_layer_output.index({Slice(), 0})));
        assert(increment.ndimension() == 1); // just the batch dimension
        return increment;
    }

    Tensor per_time_step_value_loss(const Tensor& y_t, const Tensor& linear_layer_output) const {
        // y_t is batch_size x output_data_dim
        // linear_layer_output is batch_size x (output_data_dim+1)
        using namespace torch::indexing;
        Tensor diff = (linear_layer_output.index({Slice(), Slice(1, None)}) - y_t).sum(1);
        Tensor increment = diff * diff;
        assert(increment.ndimension() == 1); // just the batch dimension
        return increment;
    }

    Tensor forward(Tensor x, Tensor y) {
        validate_x(x);
        validate_y(y);

        // let's just apply the same input at each time step
        // x.shape is (1, batch_size, input_data_dim)

        // output.shape is (num_time_steps, batch_size, output_data_dim)
        size_t num_time_steps = y.size(0);

        // intermediate outputs
        std::tuple<Tensor, Tensor> lstm_cell_output;
        Tensor linear_layer_output;

        // initial LSTM cell (it always runs at least once)
        lstm_cell_output = lstm_cell_(x);
        Tensor hx = std::get<0>(lstm_cell_output);
        validate_hx(hx);
        Tensor cx = std::get<1>(lstm_cell_output);
        validate_cx(cx);
        linear_layer_output = linear_(hx);
        validate_linear_layer(linear_layer_output);

        Tensor loss = torch::tensor(0.0);
        assert(loss.ndimension() == 0);

        // from linear_layer_output, we make two random choices:
        for (int t = 0; t < num_time_steps; t++) {

            // add the decision to continue to the loss
            loss += per_time_step_continue_loss(linear_layer_output).sum();
            assert(loss.ndimension() == 0);

            // add the decision of the values to the loss
            using namespace torch::indexing;
            loss += per_time_step_value_loss(y.index({t, Slice(), Slice()}), linear_layer_output).sum();
            assert(loss.ndimension() == 0);

            // compute the next linear layer
            lstm_cell_output = lstm_cell_(x, std::tuple<Tensor,Tensor>{hx, cx});
            hx = std::get<0>(lstm_cell_output);
            validate_hx(hx);
            cx = std::get<1>(lstm_cell_output);
            validate_cx(cx);
            linear_layer_output = linear_(hx);
            validate_linear_layer(linear_layer_output);
        }

        // add the decision to stop to the loss
        loss += per_time_step_stop_loss(linear_layer_output).sum();
        assert(loss.ndimension() == 0);

        return loss;
    }

    void reset() override {
        lstm_cell_ = register_module("lstm_cell",LSTMCell(input_data_dim_, lstm_hidden_dim_));
        linear_ = register_module("linear", Linear(lstm_hidden_dim_, output_data_dim_+1));
    }

    int batch_size_ = 1;
    int input_data_dim_;
    int lstm_hidden_dim_;
    int output_data_dim_;
    torch::nn::LSTMCell lstm_cell_;
    torch::nn::Linear linear_;
};

std::vector<std::tuple<Tensor, Tensor>> generate_training_data(
        int num_examples, int input_data_dim, int output_data_dim) {
    std::vector<std::tuple<Tensor, Tensor>> data;
    for (int i = 0; i < num_examples; ++i) {
        int num_time_steps = 10; // TODO make it random
        Tensor x = torch::randn({1, input_data_dim});
        Tensor y = torch::randn({num_time_steps, 1, output_data_dim});
        data.emplace_back(x, y);
    }
    return data;
}

void single_threaded_non_batched_training(Net& net, const std::vector<std::tuple<Tensor,Tensor>>& training_data,
                                          int minibatch_size, int max_iter) {
    torch::optim::Adam optimizer {net.parameters()};
    size_t num_examples = training_data.size();
    for (int iter = 0; iter < max_iter; ++iter) {
//        std::cout << "iter " << iter << std::endl;
        // sample a minibatch
        Tensor minibatch = torch::randint(0, num_examples, minibatch_size,
                                          TensorOptions().dtype(torch::ScalarType::Long));
        assert(minibatch.ndimension() == 1);
        assert(minibatch.size(0) == minibatch_size);
        assert(minibatch.dtype() == torch::ScalarType::Long);
        for (int i = 0; i < minibatch_size; i++) {
            const auto& [x, y] = training_data[minibatch.data_ptr<int64_t>()[i]];
            Tensor loss = net.forward(x, y);
            loss.backward();
        }
        optimizer.step();
    }
}

void process_minibatch_chunk(Net& net,
                             const std::vector<Tensor>& net_parameters,
                             const std::vector<Tensor>& parameter_grads_to_accumulate,
                             const std::vector<std::tuple<Tensor,Tensor>>& training_data,
                             const Tensor& minibatch,
                             int start_idx, int end_idx) {
    for (int i = start_idx; i < end_idx; ++i) {
        const auto&[x, y] = training_data[minibatch.data_ptr<int64_t>()[i]];
        Tensor loss = net.forward(x, y);
        int param_idx = 0;
        // NOTE: grad is unecessarily allocating.
        for (const auto& grad : torch::autograd::grad({loss}, net_parameters)) {
            parameter_grads_to_accumulate[param_idx++].add_(grad);
        }
    }
}

void process_minibatch_chunk_continuous(Net& net,
                                        const std::vector<Tensor>& net_parameters,
                                        const std::vector<Tensor>& parameter_grads_to_accumulate,
                                        const std::vector<std::tuple<Tensor,Tensor>>& training_data,
                                        const std::vector<size_t>& minibatch,
                                        int start_idx, int end_idx,
                                        int& num_workers_done,
                                        const bool& is_done,
                                        const int worker_idx,
                                        std::vector<bool>& workers_do_work,
                                        std::mutex& mutex,
                                        std::condition_variable& worker_threads_iteration_done_cv,
                                        std::condition_variable& main_thread_iteration_done_cv) {

    // upon being launched, all threads block here
    std::unique_lock<std::mutex> lock {mutex};
//    cout << "worker thread {" << start_idx << "..." << end_idx - 1 << "} acquired lock" << endl;
//
    //c10::InferenceMode guard(true);


    while (!is_done) {

//        cout << "worker thread {" << start_idx << "..." << end_idx - 1 << "} read is_done=" << is_done << endl;

        // do the work
//        cout << "worker thread {" << start_idx << "..." << end_idx - 1 << "} releasing lock and then doing work..." << endl;
        lock.unlock();
        for (int i = start_idx; i < end_idx; ++i) {
            const auto&[x, y] = training_data[minibatch[i]];
            // NOTE: this is extending the computation graph that is shared among all threads...
            Tensor loss = net.forward(x, y); // 6ms, good scaling up to 16, weak scaling at 32, slowdown at 64
            int param_idx = 0;
            // NOTE: grad is unnecessarily allocating.
            // NOTE: this is giving a runtime error when the optimizer is used between iterations...
            // "one of the variables needed for gradient computation has been modified by an inplace operation"
            for (const auto& grad : torch::autograd::grad({loss}, net_parameters)) { // slowdown for 32 
		assert(grad.size(0) > 0);
                parameter_grads_to_accumulate[param_idx++].add_(grad);
            }
        }

        // indicate that we're done with our work
        lock.lock();
        num_workers_done += 1;
        worker_threads_iteration_done_cv.notify_one();
//        cout << "worker thread {" << start_idx << "..." << end_idx - 1 << "} done with work, acquired lock, set num_workers_done=" << num_workers_done << " and notified main thread; waiting for main thread..." << endl;

        // wait for main thread to be done with its work
        workers_do_work[worker_idx] = false;
        main_thread_iteration_done_cv.wait(lock, [&]{ return workers_do_work[worker_idx]; });

//        cout << "worker thread {" << start_idx << "..." << end_idx - 1 << "} acquired lock after waiting on main_thread_iteration_done_cv" << endl;
    }

//    cout << "worker thread {" << start_idx << "..." << end_idx - 1 << "} read is_done=" << is_done << endl;

    num_workers_done += 1;
    worker_threads_iteration_done_cv.notify_one();
//    cout << "worker thread {" << start_idx << "..." << end_idx - 1 << "} acquired lock, set num_workers_done=" << num_workers_done << "and notified main thread; returning." << endl;
    lock.unlock();
}

std::vector<Tensor> clone_parameter_grads(const std::vector<Tensor>& params) {
    std::vector<Tensor> grad_clones;
    for (const auto& param : params) {
        grad_clones.emplace_back(torch::zeros_like(param.grad()).detach()); // TODO detach() here is probably not necessary..
    }
    return grad_clones;
}

template <typename Generator>
void fill_minibatch(std::vector<size_t>& minibatch, const size_t num_training_data, Generator& gen) {
    std::vector<double> weights(num_training_data, 1.0); // equal weights
    std::discrete_distribution<size_t> dist {weights.begin(), weights.end()};
    for (auto& minibatch_idx : minibatch) {
        minibatch_idx = dist(gen);
    }
}

void reduce_gradients(const std::vector<Tensor>& net_parameters,
                      const std::vector<std::vector<Tensor>>& per_thread_parameter_grads) {
    for (int i = 0; i < net_parameters.size(); ++i) {
        Tensor& net_mutable_grad = net_parameters[i].mutable_grad();
        for (const auto& parameter_grads : per_thread_parameter_grads) {
            net_mutable_grad.add_(parameter_grads[i]);
        }
    }
}

void multi_threaded_non_batched_training_old(Net& net, const std::vector<std::tuple<Tensor,Tensor>>& training_data,
                                             int minibatch_size, int max_iter, int num_threads) {

    std::vector<Tensor> net_parameters { net.parameters(true) };
    // run backward() once so that the gradients become defined..
    const auto& [dummy_x, dummy_y] { training_data[0] };
    net.forward(dummy_x, dummy_y).backward();

    cout << torch::get_num_threads() << endl;
    torch::set_num_threads(1);
    cout << torch::get_num_threads() << endl;

    assert(minibatch_size % num_threads == 0);
    size_t chunk_size = minibatch_size / num_threads;
    auto starts = std::make_unique<size_t[]>(num_threads);
    auto ends = std::make_unique<size_t[]>(num_threads);
    cout << "chunks: " << endl;
    for (size_t i = 0; i < num_threads; ++i) {
        starts[i] = chunk_size*i;
	ends[i] = chunk_size*(i+1);
	cout << starts[i] << " " << ends[i] << endl;
    }
    assert(ends[num_threads-1] == minibatch_size);

    std::vector<std::vector<Tensor>> per_thread_parameter_grads;
    for (int i = 0; i < num_threads; ++i)
        per_thread_parameter_grads.emplace_back(clone_parameter_grads(net_parameters));

    torch::optim::Adam optimizer {net_parameters };
    optimizer.zero_grad();
    size_t num_examples = training_data.size();

    for (int iter = 0; iter < max_iter; ++iter) {
        std::cout << "iter " << iter << std::endl;
        // sample a minibatch
        Tensor minibatch = torch::randint(0, num_examples, minibatch_size,
                                          TensorOptions().dtype(torch::ScalarType::Long));
        assert(minibatch.ndimension() == 1);
        assert(minibatch.size(0) == minibatch_size);
        assert(minibatch.dtype() == torch::ScalarType::Long);

        using std::ref, std::cref;

        // start timing parallel section
	namespace chrono = std::chrono;
	auto parallel_start = chrono::steady_clock::now();
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            int start = starts[i];
            int end = ends[i];
            threads.emplace_back(std::thread(process_minibatch_chunk, ref(net), cref(net_parameters),
                                             cref(per_thread_parameter_grads[i]), cref(training_data),
                                             cref(minibatch), start, end));
        }
        for (auto& thread : threads) {
            thread.join();
        }
	auto parallel_end = chrono::steady_clock::now();
	auto parallel_duration = chrono::duration <double, std::milli> (parallel_end - parallel_start).count();

        // merge the gradient accumulators from net_clone_1 and net_clone_2 into the gradient accumulators for net
        // TODO use a more efficient accumulation scheme e.g. tree-structured
	auto merge_start = chrono::steady_clock::now();
        for (int i = 0; i < net_parameters.size(); ++i) {
            Tensor& net_mutable_grad = net_parameters[i].mutable_grad();
            for (const auto& parameter_grads : per_thread_parameter_grads) {
                net_mutable_grad.add_(parameter_grads[i]);
            }
        }
	auto merge_end = chrono::steady_clock::now();
	auto merge_duration = chrono::duration <double, std::milli> (merge_end - merge_start).count();

	auto step_start = chrono::steady_clock::now();
        optimizer.step();
	auto step_end = chrono::steady_clock::now();
	auto step_duration = chrono::duration <double, std::milli> (step_end - step_start).count();

	using std::cout, std::endl;
	cout << "parallel grad: " << parallel_duration << "; merge: " << merge_duration << "; step: " << step_duration << endl;
    }
}


void multi_threaded_non_batched_training(Net& net, const std::vector<std::tuple<Tensor,Tensor>>& training_data,
                                         size_t minibatch_size, int max_iter, int num_threads, bool verbose=false) {

    // TODO verify that this actually gives the right gradients!
//    torch::autograd::DetectAnomalyGuard detect_anomaly;
    std::vector<Tensor> net_parameters { net.parameters(true) };

    // run backward() once so that the gradients become defined..
    const auto& [dummy_x, dummy_y] { training_data[0] };
    net.forward(dummy_x, dummy_y).backward();

    cout << torch::get_num_threads() << " " << torch::get_num_interop_threads() << endl;
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    cout << torch::get_num_threads() << " " << torch::get_num_interop_threads() << endl;

    assert(minibatch_size % num_threads == 0);
    size_t chunk_size = minibatch_size / num_threads;
    auto starts = std::make_unique<size_t[]>(num_threads);
    auto ends = std::make_unique<size_t[]>(num_threads);
    cout << "chunks: " << endl;
    for (size_t i = 0; i < num_threads; ++i) {
        starts[i] = chunk_size*i;
	ends[i] = chunk_size*(i+1);
	cout << starts[i] << " " << ends[i] << endl;
    }
    assert(ends[num_threads-1] == minibatch_size);

    std::vector<std::vector<Tensor>> per_thread_parameter_grads;
    for (int i = 0; i < num_threads; ++i)
        per_thread_parameter_grads.emplace_back(clone_parameter_grads(net_parameters));

    torch::optim::Adam optimizer {net_parameters };
    optimizer.zero_grad();

    std::vector<size_t> minibatch(minibatch_size);

    std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex); // acquires the lock

    std::condition_variable worker_threads_iteration_done_cv;
    std::condition_variable main_thread_iteration_done_cv;
    int num_workers_done = 0;
    std::vector<bool> workers_do_work(num_threads, false);
    bool is_done = false;

    // start the threads
    using std::ref, std::cref;
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i)
        threads.emplace_back(std::thread(process_minibatch_chunk_continuous,
                                         ref(net),
                                         cref(net_parameters),
                                         cref(per_thread_parameter_grads[i]),
                                         cref(training_data),
                                         cref(minibatch),
                                         starts[i], ends[i],
                                         ref(num_workers_done),
                                         cref(is_done),
                                         i,
                                         ref(workers_do_work),
                                         ref(mutex),
                                         ref(worker_threads_iteration_done_cv),
                                         ref(main_thread_iteration_done_cv)));

    std::random_device rd;
    std::mt19937 gen {rd()};
    for (int iter = 0; iter < max_iter; ++iter) {
        std::cout << "main thread iter " << iter << std::endl;

        // sample a minibatch
        verbose && (cout << "main thread sampling minibatch..." << endl);
        fill_minibatch(minibatch, training_data.size(), gen);

        // start timing parallel section
        namespace chrono = std::chrono;
        auto parallel_start = chrono::steady_clock::now();

        if (iter > 0) {
            // notify the workers
            // NOTE: on the first iteration, this is a no-op
            verbose && (cout << "main thread notifying workers..." << endl);
            main_thread_iteration_done_cv.notify_all();
        }

        // wait for workers to finish
        verbose && (cout << "main thread waiting for workers to finish..." << endl);
        std::fill(workers_do_work.begin(), workers_do_work.end(), true);
        worker_threads_iteration_done_cv.wait(lock, [&]{ return num_workers_done == num_threads; });
        num_workers_done = 0;
        //std::fill(workers_do_work.begin(), workers_do_work.end(), false);

        verbose && (cout << "main thread processing result from workers..." << endl);

        // end timer for parallel section
        auto parallel_end = chrono::steady_clock::now();
        auto parallel_duration = chrono::duration <double, std::milli> (parallel_end - parallel_start).count();

        // TODO use a more efficient accumulation scheme (NOTE: it's not the bottleneck right now..)
	    auto merge_start = chrono::steady_clock::now();
        reduce_gradients(net_parameters, per_thread_parameter_grads);
        auto merge_end = chrono::steady_clock::now();
        auto merge_duration = chrono::duration <double, std::milli> (merge_end - merge_start).count();

        auto step_start = chrono::steady_clock::now();
        optimizer.step();
        auto step_end = chrono::steady_clock::now();
        auto step_duration = chrono::duration <double, std::milli> (step_end - step_start).count();

        using std::cout, std::endl;
        cout << "main thread parallel grad: " << parallel_duration << "; merge: " << merge_duration << "; step: " << step_duration << endl;
    }

    // notify the workers that we are done
    verbose && (cout << "main thread notifying workers that we are done" << endl);
    is_done = true;
    std::fill(workers_do_work.begin(), workers_do_work.end(), true);
    main_thread_iteration_done_cv.notify_all();

    // wait for workers to finish
    verbose && (cout << "main thread waiting for workers to finish" << endl);
    worker_threads_iteration_done_cv.wait(lock, [&]{ return num_workers_done == num_threads; });

    verbose && (cout << "main thread joining on child threads" << endl);
    lock.unlock();
    for (auto& thread : threads) {
        thread.join();
    }
}

int main() {
    int input_data_dim = 3;
    int batch_size = 1;
    int lstm_hidden_dim = 64;
    int output_data_dim = 2; // output of linear layer
    Net net {input_data_dim, lstm_hidden_dim, output_data_dim};

    Tensor x = torch::randn({batch_size, input_data_dim});
    int num_time_steps = 10;
    Tensor y = torch::randn({num_time_steps, batch_size, output_data_dim});

    using std::cout, std::endl;
    Tensor loss = net.forward(x, y);
    cout << loss << endl;

    int num_examples = 1000;
    auto training_data = generate_training_data(num_examples, input_data_dim, output_data_dim);

    int minibatch_size = 64;
    int num_threads = 2;
    int max_iter = 1000;
//    single_threaded_non_batched_training(net, training_data, minibatch_size, max_iter);
    multi_threaded_non_batched_training(net, training_data, minibatch_size, max_iter, num_threads);
    //multi_threaded_non_batched_training_old(net, training_data, minibatch_size, max_iter, num_threads);
}
