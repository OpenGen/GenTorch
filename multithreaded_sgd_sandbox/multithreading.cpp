#include <thread>
#include <iostream>
#include <chrono>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cassert>
#include <random>

#include <torch/torch.h>

namespace chrono = std::chrono;

using std::cout, std::endl;

double do_job(double datum, size_t ms) {
    double x = 0;
    for (int i = 0; i < 100'000; ++i) {
	x = x * datum;
	x = x / datum;
    }
    //torch::TensorOptions opt = torch::TensorOptions().dtype(torch::ScalarType::Double);
    //torch::Tensor a = torch::tensor(datum, opt);
    //torch::Tensor b = torch::ones({64, 64, 64}, opt);
    //torch::Tensor c = a * b;
    //torch::Tensor s = torch::sum(c);
    //std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    //return *s.data_ptr<double>();
    return x;
}

template <typename Generator>
void fill_jobs(std::vector<size_t>& jobs, const size_t num_training_data, Generator& gen) {
    std::vector<double> weights(num_training_data, 1.0); // equal weights
    std::discrete_distribution<size_t> dist {weights.begin(), weights.end()};
    for (auto& job : jobs) {
        job = dist(gen);
    }
}

void worker(const std::vector<double>& data, std::vector<double>& results,
            const std::vector<size_t>& jobs, size_t job_ms, size_t start, size_t end,
            int& num_workers_done, const bool& is_done, int worker_idx, std::vector<bool>& workers_do_work,
            std::mutex& mutex,
            std::condition_variable& worker_threads_iteration_done_cv,
            std::condition_variable& main_thread_iteration_done_cv) {
    std::unique_lock<std::mutex> lock {mutex};

    while (!is_done) {
        lock.unlock();
        double result = 0.0;
        for (size_t i = start; i < end; ++i) {
            result += do_job(data[jobs[i]], job_ms);
        }
        results[worker_idx] = result;

        // indicate that we're done with our work
        lock.lock();
        num_workers_done += 1;
        worker_threads_iteration_done_cv.notify_one();

        // wait for main thread to be done with its work
        workers_do_work[worker_idx] = false;
        main_thread_iteration_done_cv.wait(lock, [&]{ return workers_do_work[worker_idx]; });

    }

    num_workers_done += 1;
    worker_threads_iteration_done_cv.notify_one();
    lock.unlock();
}

void do_experiment(size_t num_data, size_t job_ms, size_t main_thread_ms, size_t num_iters, size_t num_jobs_per_iter,
                   size_t num_threads) {

    // data
    std::vector<double> data(num_data, 0.0);

    // compute chunks
    assert(num_jobs_per_iter % num_threads == 0);
    size_t chunk_size = num_jobs_per_iter / num_threads;
    auto starts = std::make_unique<size_t[]>(num_threads);
    auto ends = std::make_unique<size_t[]>(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        starts[i] = chunk_size*i;
        ends[i] = chunk_size*(i+1);
    }
    assert(ends[num_threads-1] == num_jobs_per_iter);

    // set up synchronization
    std::vector<size_t> jobs(num_jobs_per_iter);
    std::mutex mutex;
    std::unique_lock<std::mutex> lock {mutex};
    std::condition_variable worker_threads_iteration_done_cv;
    std::condition_variable main_thread_iteration_done_cv;
    int num_workers_done = 0;
    bool is_done = false;
    std::vector<bool> workers_do_work(num_threads, false);

    // storage for results of workers
    std::vector<double> results(num_threads, 0.0);

    // start the threads
    using std::ref, std::cref;
    std::vector<std::thread> threads;
    for (int worker_idx = 0; worker_idx < num_threads; ++worker_idx)
        threads.emplace_back(std::thread(worker,
                                         cref(data), ref(results),
                                         cref(jobs), job_ms, starts[worker_idx], ends[worker_idx],
                                         ref(num_workers_done),
                                         cref(is_done), worker_idx,
                                         ref(workers_do_work),
                                         ref(mutex),
                                         ref(worker_threads_iteration_done_cv),
                                         ref(main_thread_iteration_done_cv)));


    std::random_device rd;
    std::mt19937 gen {rd()};
    for (int iter = 0; iter < num_iters; ++iter) {
        std::cout << "main thread iter " << iter << std::endl;

        // sample a minibatch
        fill_jobs(jobs, data.size(), gen);

        // start timing parallel section
        namespace chrono = std::chrono;
        auto parallel_start = chrono::steady_clock::now();

        if (iter > 0) {
            // NOTE: on the first iteration, this is a no-op
            main_thread_iteration_done_cv.notify_all();
        }

        // wait for workers to finish
//        verbose && (cout << "main thread waiting for workers to finish..." << endl);
        std::fill(workers_do_work.begin(), workers_do_work.end(), true);
        worker_threads_iteration_done_cv.wait(lock, [&]{ return num_workers_done == num_threads; });
        num_workers_done = 0;
        std::fill(workers_do_work.begin(), workers_do_work.end(), false);

//        verbose && (cout << "main thread processing result from workers..." << endl);


        // end timer for parallel section
        auto parallel_end = chrono::steady_clock::now();
        auto parallel_duration = chrono::duration <double, std::milli> (parallel_end - parallel_start).count();

        // start timer for main thread work
        auto serial_start = chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(main_thread_ms));
        auto serial_end = chrono::steady_clock::now();
        auto serial_duration = chrono::duration <double, std::milli> (serial_end - serial_start).count();

        using std::cout, std::endl;
        cout << "main thread parallel section: " << parallel_duration << "; serial section: " << serial_duration << endl;
    }

    // notify the workers that we are done
    cout << "main thread notifying workers that we are done" << endl;
    is_done = true;
    std::fill(workers_do_work.begin(), workers_do_work.end(), true);
    main_thread_iteration_done_cv.notify_all();

    // wait for workers to finish
    cout << "main thread waiting for workers to finish" << endl;
    worker_threads_iteration_done_cv.wait(lock, [&]{ return num_workers_done == num_threads; });

    cout << "main thread joining on child threads" << endl;
    lock.unlock();
    for (auto& thread : threads) {
        thread.join();
    }
}

int main() {

    size_t num_data = 1000;
    size_t job_ms = 2;
    size_t main_thread_ms = 0;
    size_t num_iters = 10000;
    size_t num_jobs_per_iter = 64;
    size_t num_threads = 1;

    cout << torch::get_num_threads() << endl;
    torch::set_num_threads(1);
    cout << torch::get_num_threads() << endl;

    do_experiment(num_data, job_ms, main_thread_ms, num_iters, num_jobs_per_iter, num_threads);

}
