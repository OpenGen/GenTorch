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

#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <gen/address.h>
#include <gen/trie.h>

#include <cassert>
#include <iostream>
#include <memory>

#include <torch/torch.h>

using torch::Tensor;
using torch::TensorOptions;
using torch::tensor;


TEST_CASE( "empty address", "[choice addresses]" ) {
    Address empty_addr; // empty address
    REQUIRE(empty_addr.empty());
}

TEST_CASE( "list constructed address", "[choice addresses]" ) {
    Address a {"x" };
    Address b {"x", "y" };
    REQUIRE(!a.empty());
    REQUIRE(!b.empty());
    REQUIRE(a.first() == "x");
    REQUIRE(a.rest().empty());
    REQUIRE(b.first() == "x");
    REQUIRE(!b.rest().empty());
    REQUIRE(b.rest().first() == "y");

    Address addr {"0", "1"};
    addr = addr.rest();
    REQUIRE(addr.first() == "1");
}

TEST_CASE( "address iostream", "[choice addresses]" ) {
    Address addr_ostream {"0", "1", "2"};
    std::stringstream ss;
    ss << addr_ostream;
    REQUIRE(ss.str() == "{\"0\", \"1\", \"2\"}");
    std::stringstream ss_empty_addr;
    ss_empty_addr << Address();
    REQUIRE(ss_empty_addr.str() == "{}");
}

TEST_CASE( "address with strings keys immutable", "[choice addresses]" ) {
    std::string f = "x";
    Address c {f };
    f = "y";
    REQUIRE(c.first() == "x");
}