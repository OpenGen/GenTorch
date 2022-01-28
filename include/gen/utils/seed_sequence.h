#ifndef GEN_SEED_SEQUENCE_H
#define GEN_SEED_SEQUENCE_H

#include <random>
#include <cstdint>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <memory>

#include <gen/utils/randutils.h>

// TODO seed_seq_fe256 can't be copied or moved??

// Initializing the N-word entropy store with M words requires O(N * M)
// *   time precisely because of the avalanche requirements.  Once constructed,
// *   calls to generate are linear in the number of words generated.

class SpawnableSeedSequence {
public:

    // for SeedSequence concept
    typedef uint32_t result_type;
private:
    randutils::seed_seq_fe256 ss_;
    const std::vector<uint32_t> seeds_;
    std::vector<uint32_t> prev_spawn_seeds_;

public:

    explicit SpawnableSeedSequence(uint32_t seed)
            : seeds_{seed}, ss_{seed}, prev_spawn_seeds_(seeds_) {}

    SpawnableSeedSequence(std::initializer_list<uint32_t> init)
            : ss_(init), seeds_(init), prev_spawn_seeds_(seeds_) {}

    template <typename InputIter>
    SpawnableSeedSequence(InputIter begin, InputIter end)
            : ss_(begin, end), seeds_(begin, end), prev_spawn_seeds_(seeds_) {}

    SpawnableSeedSequence(const SpawnableSeedSequence& other) = default;

    // generate

    template<typename RandomIt>
    void generate(RandomIt begin, RandomIt end) {
        ss_.generate(begin, end);
    }

    // spawn

    SpawnableSeedSequence spawn() {
        uint32_t& last = *(prev_spawn_seeds_.end()-1);
        if (last < std::numeric_limits<uint32_t>::max()) {
            last += 1;
        } else {
            prev_spawn_seeds_.emplace_back(0);
        }
        return SpawnableSeedSequence(prev_spawn_seeds_.begin(), prev_spawn_seeds_.end());
    }
};


#endif //GEN_SEED_SEQUENCE_H
