/* Copyright 2021 The LibGen Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include <catch2/catch.hpp>
#include <gen/utils/seed_sequence.h>

#include <set>

TEST_CASE("initialization", "[seed_sequence]") {

    std::vector<SpawnableSeedSequence> seqs;
    auto ss0 = seqs.emplace_back(SpawnableSeedSequence{0});
    auto ss1 = seqs.emplace_back(SpawnableSeedSequence{1});
    auto ss2 = seqs.emplace_back(SpawnableSeedSequence{0, 0});

    // check that they generate different seeds
    std::set<uint32_t> seeds;
    for (auto& ss : seqs) {
        std::vector<uint32_t> seed(1); // just generate a single uint32_t seed
        ss.generate(seed.begin(), seed.end());
        uint32_t seed_int = seed[0];
        REQUIRE(seeds.find(seed_int) == seeds.end());
    }

    // check that a MT19337 initialized from them has a different sequence
    std::set<uint32_t> random_values;
    for (auto& ss : seqs) {
        std::mt19937 rng(ss);
        uint32_t value = rng();
        REQUIRE(random_values.find(value) == random_values.end());
    }

}

TEST_CASE("spawning", "[seed_sequence]") {

    // create a tree of seed sequences and check that:
    // 1. they 'generate' different seeds
    // 2. mt19337 RNGs initialized with seeds generated from them produce different streams

    std::vector<SpawnableSeedSequence> seqs;
    auto ss0 = seqs.emplace_back(SpawnableSeedSequence{1});
    auto ss1 = seqs.emplace_back(ss0.spawn());
    auto ss2 = seqs.emplace_back(ss0.spawn());
    auto ss3 = seqs.emplace_back(ss1.spawn());
    auto ss4 = seqs.emplace_back(ss2.spawn());
    auto ss5 = seqs.emplace_back(ss4.spawn());

    // check that they generate different seeds
    std::set<uint32_t> seeds;
    for (auto& ss : seqs) {
        std::vector<uint32_t> seed(1); // just generate a single uint32_t seed
        ss.generate(seed.begin(), seed.end());
        uint32_t seed_int = seed[0];
        REQUIRE(seeds.find(seed_int) == seeds.end());
    }

    // check that a MT19337 initialized from them has a different sequence
    std::set<uint32_t> random_values;
    for (auto& ss : seqs) {
        std::mt19937 rng(ss);
        uint32_t value = rng();
        REQUIRE(random_values.find(value) == random_values.end());
    }

}
