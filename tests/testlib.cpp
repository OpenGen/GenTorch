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

// TODO put real test suite here

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <gen/trie.h>

#include <torch/torch.h>

using torch::tensor;
using torch::Tensor;

TEST_CASE( "Quick check", "[main]" ) {

Tensor t1 = tensor(1.123);
Tensor t2 = tensor(2.234);

Trie trie { };
REQUIRE(!trie.has_value());

}