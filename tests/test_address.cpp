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

TEST_CASE("dummy", "[dummy]") {
    dummy();
}

//TEST_CASE( "empty address", "[choice addresses]" ) {
//    Address empty_addr; // empty address
//    REQUIRE(empty_addr.empty());
//}

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

TEST_CASE( "choice trie basics", "[choice trie]" ) {
    Tensor t1 = tensor(1.123);
    Tensor t2 = tensor(2.234);
    Trie trie { };
    REQUIRE(!trie.has_value());
    trie.set_value(Address{"a", "b", "c", "d"}, t1);
    trie.set_value(Address{"a", "b", "e", "f"}, t2);
    trie.set_value(Address{"a", "b", "g"}, std::string("asdf"));
    REQUIRE(std::any_cast<Tensor>(trie.get_value(Address{"a", "b", "c", "d"})).equal(t1));
    REQUIRE(std::any_cast<Tensor>(trie.get_value(Address{"a", "b", "e", "f"})).equal(t2));
}

TEST_CASE( "choice trie with bool tensor", "[choice trie]" ) {
    Tensor bool_tensor = tensor(0, TensorOptions().dtype(torch::ScalarType::Bool));
    Trie trie_with_value {};
    trie_with_value.set_value(bool_tensor);
    REQUIRE(trie_with_value.has_value());
    std::cout << trie_with_value << std::endl;
}

void walk_choice_trie(const Trie& trie, size_t indent) {
    for (const auto& [key, subtrie] : trie.subtries()) {
        std::cout << std::string(indent, ' ') << "> enter " << key << std::endl;
        if (subtrie.has_value()) {
            std::cout << std::string(indent+4, ' ') << "value" << std::endl;
        } else {
            walk_choice_trie(subtrie, indent+4);
        }
        std::cout << std::string(indent, ' ') << "< exit " << key << std::endl;
    }
}

TEST_CASE( "set_subtrie", "[choice trie]" ) {
    Trie trie_with_value { };
    trie_with_value.set_value(tensor(1.123));
    Trie super_trie { };
    super_trie.set_subtrie(Address{"a", "b", "c", "d", "e"}, std::move(trie_with_value));
    std::cout << super_trie << std::endl;

    Trie trie_with_value_2 { };
    trie_with_value_2.set_value(tensor(1.123));
    trie_with_value_2.set_value(tensor(1, TensorOptions().dtype(torch::ScalarType::Bool)));
    super_trie.set_subtrie(Address{"x", "y", "z"}, std::move(trie_with_value_2));
    std::cout << trie_with_value_2 << std::endl;
    std::cout << super_trie << std::endl;
    REQUIRE(!super_trie.has_value());
    walk_choice_trie(super_trie, 0);
}

TEST_CASE("set_value", "[choice trie]") {
    Trie t {};
    string value {"asdf"};
    string& set_value_ref = t.set_value(value);
    std::any& stored_value_any_ref = t.get_value();
    string& stored_string_ref = *std::any_cast<string>(&stored_value_any_ref);
    stored_string_ref.replace(0, 2, "zz");
    REQUIRE(value == "asdf");
    REQUIRE(std::any_cast<string>(t.get_value()) == "zzdf");

    set_value_ref.replace(0, 2, "xx");
    REQUIRE(value == "asdf");
    REQUIRE(std::any_cast<string>(t.get_value()) == "xxdf");
}

class Base {
public:
    virtual int print_my_val() {
        std::cout << 2 << std::endl;
        return 2;
    }
    virtual double print_my_val_pure() const = 0;
};

class Derived : public Base {
public:
    explicit Derived(double my_val) : my_val_{my_val} {}
    double print_my_val_pure() const override {
        std::cout << my_val_ << std::endl;
        return my_val_;
    }
private:
    double my_val_;
};