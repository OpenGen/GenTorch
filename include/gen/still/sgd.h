#ifndef GEN_SGD_H
#define GEN_SGD_H

#include <cassert>
#include <vector>
#include <utility>
#include <random>
#include <barrier> // NOTE: requires C++20 or c++17 with experimental (consider using boost version or re-implementing)
#include <gen/utils/randutils.h>

// NOTE: this file does not depend on Torch
// TODO provide a callback for generating data on-demand, that defaults to reading from a data set
// TODO rename this namespace
namespace gen::sgd {

    // TODO move to sgd.cpp or a utility cpp file (is reusable for e.g. importance sampling)
    std::vector<std::pair<size_t,size_t>> even_blocks(size_t num_elements, size_t num_blocks) {
        std::vector<std::pair<size_t,size_t>> blocks {num_blocks};
        size_t start = 0;
        size_t stop;
        for (int i = 0; i < num_blocks; i++) {
            size_t k = num_elements / num_blocks;
            size_t rem = num_elements % num_blocks;
            size_t block_size;
            if (i < rem) {
                block_size = k + 1;
            } else {
                block_size = k;
            }
            stop = start + block_size;
            blocks[i] = {start, stop};
            start = stop;
        }
        assert((*(blocks.end()-1)).second == num_elements);
        return blocks;
    }

    template <typename RNGType>
    std::vector<size_t> generate_minibatch(RNGType& rng, size_t dataset_size, size_t minibatch_size) {
        std::vector<size_t> minibatch(minibatch_size);
        std::uniform_int_distribution<size_t> dist {0, dataset_size-1};
        for (int i = 0; i < minibatch_size; i++) {
            minibatch[i] = dist(rng);
        }
        return minibatch;
    }

    template <typename RNGType, typename ParametersType, typename DatumType, typename UnpackDatumType>
    double estimate_objective(RNGType& rng, ParametersType& parameters,
                              const std::vector<DatumType>& data,
                              UnpackDatumType& unpack_datum) {
        double total = 0.0;
        for (const auto& datum : data) {
            auto [model, constraints] = unpack_datum(datum);
            auto [trace, log_weight] = model.generate(rng, parameters, constraints, false);
            total += log_weight;
        }
        return total / static_cast<double>(data.size());
    }

    template <typename ParametersType, typename RNGType,
            typename StepCallbackType, typename DatasetType, typename UnpackDatumType>
    void train_supervised_single_threaded(ParametersType& parameters,
                                          const StepCallbackType& callback,
                                          const DatasetType& data,
                                          const UnpackDatumType& unpack_datum,
                                          const size_t minibatch_size,
                                          RNGType& rng) {
        typedef typename ParametersType::accumulator_t GradientAccumulatorType;
        GradientAccumulatorType accum {parameters};
        bool done = false;
        const double scaler = 1.0 / static_cast<double>(minibatch_size);
        while (!done) {
            std::vector<size_t> minibatch = generate_minibatch(rng, data.size(), minibatch_size);
            for (size_t i = 0; i < minibatch_size; i++) {
                auto [model, constraints] = unpack_datum(data[minibatch[i]]);
                auto [trace, log_weight] = model.generate(rng, parameters, constraints, true);
                const auto& retval = trace->get_return_value();
                const auto retval_grad = zero_gradient(retval);
                trace->parameter_gradient(retval_grad, scaler, accum);
            }
            accum.update_module_gradients();
            done = callback(minibatch);
        }
   }

    template <typename ParametersType,
              typename StepCallbackType, typename DatasetType, typename UnpackDatumType,
              typename SeedSequenceType>
    void train_supervised(ParametersType& parameters,
                          const StepCallbackType& callback,
                          const DatasetType& data,
                          const UnpackDatumType& unpack_datum,
                          const size_t minibatch_size,
                          const size_t num_threads,
                          SeedSequenceType& seed) {
        typedef typename ParametersType::accumulator_t GradientAccumulatorType;

        // one gradient accumulator per thread
        std::vector<GradientAccumulatorType> accums;
        for (int i = 0; i < num_threads; i++) {
            accums.emplace_back(GradientAccumulatorType{parameters});
        }

        const size_t data_size = data.size();
        const double scaler = 1.0 / static_cast<double>(minibatch_size);
        size_t iter = 0;
        bool done = false;

        // initialize RNG that will be used to generate minibatches
        std::mt19937 rng(seed);

        std::vector<size_t> minibatch = generate_minibatch(rng, data.size(), minibatch_size);

        auto iteration_serial_stage = [data_size,&iter,&done,&accums,&rng,
                                      &minibatch,&parameters,&data,&callback]() {

            // increments the gradients in the shared parameters object
            // and resets the per-thread gradient accumulators to zero
            for (auto& accum : accums) {
                accum.update_module_gradients();
            }

            // user callback, which implements the gradient step and decides whether we are done or not
            done = callback(minibatch);

            // compute minibatch for next iteration
            if (!done) {
                minibatch = generate_minibatch(rng, data_size, minibatch.size());
            }

            iter++;
        };

        std::barrier sync_point(num_threads, iteration_serial_stage);

        auto stochastic_gradient_chunk = [
                &sync_point,
                &minibatch = std::as_const(minibatch),
                &data = std::as_const(data),
                &parameters,
                scaler,
                &done = std::as_const(done),
                &unpack_datum](GradientAccumulatorType& accum, SeedSequenceType& seed,
                               size_t start, size_t stop) {
            std::mt19937 rng(seed);
            while (!done) {
                for (size_t i = start; i < stop; i++) {
                    auto [model, constraints] = unpack_datum(data[minibatch[i]]);
                    auto [trace, log_weight] = model.generate(rng, parameters, constraints, true);
                    const auto& retval = trace->get_return_value();
                    const auto retval_grad = zero_gradient(retval);
                    trace->parameter_gradient(retval_grad, scaler, accum);
                }
                sync_point.arrive_and_wait();
            }
        };

        // launch worker threads
        std::vector<std::pair<size_t,size_t>> blocks = even_blocks(minibatch_size, num_threads);
        std::vector<std::thread> threads;
        std::vector<SeedSequenceType> worker_seeds;
        for (size_t i = 0; i < num_threads; i++) {
            auto [start, stop] = blocks[i];
            auto& worker_seed = worker_seeds.emplace_back(seed.spawn());
            threads.emplace_back(stochastic_gradient_chunk, std::ref(accums[i]), std::ref(worker_seed), start, stop);
            start = stop;
        }

        // wait for each worker threads to exit its loop
        for (auto& thread : threads) {
            thread.join();
        }
    }
}

#endif //GEN_SGD_H
