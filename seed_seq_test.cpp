#include <random>
#include <cstdint>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <memory>
#include "include/randutils.h"

// TODO seed_seq_fe256 can't be copied or moved??

class SpawnableSeedSequence {
private:
    randutils::seed_seq_fe256 ss_;
    std::vector<uint32_t> entropy_;

    void init_entropy() {
        ss_.param(std::back_insert_iterator<std::vector<uint32_t>>(entropy_));
        entropy_.push_back(0);
    }

public:

    // new constructor

    SpawnableSeedSequence(uint32_t seed) : ss_{seed} {
        init_entropy();
    }

    //SpawnableSeedSequence(SpawnableSeedSequence&& other)
        //: ss_{std::move(other.ss_)},
        //entropy_{std::move(other.entropy_)} {}

    // constructors matching those of randutils::seed_seq_fe256 

    SpawnableSeedSequence(const SpawnableSeedSequence&) = delete;

    template<typename InputIt>
    SpawnableSeedSequence(InputIt begin, InputIt end) : ss_(begin, end) {
        init_entropy();
    }

    template<typename T>
    SpawnableSeedSequence(std::initializer_list<T> il) : ss_(il) {
        init_entropy();
    }

    // generate

    template<typename RandomIt>
    void generate(RandomIt begin, RandomIt end) {
        ss_.generate(begin, end);
    }
    
    // spawn
    
    std::unique_ptr<SpawnableSeedSequence> spawn() {
        uint32_t& last = *(entropy_.end()-1);
        if (last < std::numeric_limits<uint32_t>::max()) { 
            last += 1;
        } else {
            entropy_.push_back(0);
        }
        return std::make_unique<SpawnableSeedSequence>(entropy_.begin(), entropy_.end());
    } 
};

void print_seeds(SpawnableSeedSequence& ss) {
    std::vector<std::uint32_t> seeds(624);
    ss.generate(seeds.begin(), seeds.end());
    std::cout << std::endl;
    for (std::uint32_t n : seeds) {
        std::cout << n << std::endl;
    }
}

int main()
{
    std::vector<std::unique_ptr<SpawnableSeedSequence>> sss;
    sss.emplace_back(std::make_unique<SpawnableSeedSequence>(1));
    sss.emplace_back(sss[0]->spawn());
    sss.emplace_back(sss[1]->spawn());
    sss.emplace_back(sss[2]->spawn());
    sss.emplace_back(sss[3]->spawn());
    sss.emplace_back(sss[1]->spawn());
    sss.emplace_back(sss[1]->spawn());
    sss.emplace_back(sss[5]->spawn());

    for (auto& ss : sss) {
        print_seeds(*ss);
    }
}
