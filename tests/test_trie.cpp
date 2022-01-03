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
#include <gen/address.h>
#include <gen/trie.h>

#include <cassert>
#include <iostream>
#include <memory>

#include <torch/torch.h>

using std::cout, std::endl;
using std::string;
using std::any, std::make_any, std::any_cast;
using torch::Tensor;
using torch::TensorOptions;
using torch::tensor;

TEST_CASE("empty", "[trie]")  {
    ChoiceTrie trie {};
    REQUIRE(trie.empty());
    trie.set_value({"a", "b"}, 1);
    REQUIRE(!trie.empty());

    ChoiceTrie trie2 {};
    trie2.set_value(1);
    REQUIRE(!trie.empty());
}

TEST_CASE("basic set_value and get_value", "[trie]" ) {

    // hierarchical address
    ChoiceTrie trie { };
    REQUIRE(!trie.has_value());
    trie.set_value({"a", "b", "c", "d"}, 1);
    trie.set_value({"a", "b", "e", "f"}, 2);
    trie.set_value({"a", "b", "g"}, std::string("3"));
    REQUIRE(!trie.has_value());
    REQUIRE(any_cast<int>(trie.get_value({"a", "b", "c", "d"})) == 1);
    REQUIRE(any_cast<int>(trie.get_value({"a", "b", "e", "f"})) == 2);
    REQUIRE(any_cast<string>(trie.get_value({"a", "b", "g"})) == "3");

    // empty address
    ChoiceTrie trie2 {};
    string str {"asdf"};
    trie2.set_value(str);
    REQUIRE(trie2.has_value());
    REQUIRE(any_cast<string>(trie2.get_value()) == "asdf");
}


TEST_CASE("mutability of references from get_value", "[trie]") {
    ChoiceTrie trie {};
    string str {"asdf"};
    trie.set_value(str);

    // we can change the value by assigning to the reference returned by get_value
    any& value_ref = trie.get_value();
    value_ref = make_any<string>("zzzz");
    REQUIRE(any_cast<string>(trie.get_value()) == "zzzz");

    // any_cast<T>(trie.get_value()) creates a copy, not a reference to the stored value
    auto str2 = any_cast<string>(trie.get_value());
    str2.replace(0, 2, "xx");
    REQUIRE(any_cast<string>(trie.get_value()) == "zzzz");

    // *any_cast<T>(&trie.get_value()) can be used to obtain a typed mutable reference to the stored value
    auto& str3 = *any_cast<string>(&trie.get_value());
    str3.replace(0, 2, "yy");
    REQUIRE(any_cast<string>(trie.get_value()) == "yyzz");

    // but note that Tensor behaves like a pointer, so although any_cast<T>(trie.get_value()) creates a copy,
    // it still refers to the same underlying memory
    Tensor t = tensor(1.0);
    trie.set_value(t);
    auto t2 = any_cast<Tensor>(trie.get_value());
    t2.add_(1.0);
    REQUIRE(t.equal(tensor(2.0)));
    REQUIRE(any_cast<Tensor>(trie.get_value()).equal(tensor(2.0)));
}

TEST_CASE("mutability of references from set_value", "[trie]") {

    ChoiceTrie trie {};
    string str = "asdf";
    string& str_ref = *any_cast<string>(&trie.set_value(str));

    // because set_value accepts the value by-value, it stores a copy of the provided value
    str.replace(0, 2, "xx");
    REQUIRE(any_cast<string>(trie.get_value()) == "asdf");

    // but the reference that is returned does refer to the stored value
    str_ref.replace(0, 2, "xx");
    REQUIRE(any_cast<string>(trie.get_value()) == "xxdf");

    // but since Tensor behaves like pointer, even the copy still refers to the provided value
    Tensor t = tensor(1.0);
    trie.set_value(t);
    t.add_(1.0);
    REQUIRE(any_cast<Tensor>(trie.get_value()).equal(tensor(2.0)));
}

TEST_CASE("get_subtrie", "[trie]") {

    ChoiceTrie trie {};
    trie.set_value({"a", "b", "c"}, 1);
    trie.set_value({"a", "b", "d"}, 2);

    ChoiceTrie subtrie = trie.get_subtrie({"a", "b"});
    REQUIRE(!subtrie.has_value());
    REQUIRE(subtrie.subtries().size() == 2);
    REQUIRE(any_cast<int>(subtrie.get_value({"c"})) == 1);
    REQUIRE(any_cast<int>(subtrie.get_value({"d"})) == 2);

    ChoiceTrie subtrie2 = trie.get_subtrie({});
    REQUIRE(!subtrie2.has_value());
    REQUIRE(subtrie2.subtries().size() == 1);
    REQUIRE(any_cast<int>(subtrie2.get_subtrie({"a", "b", "c"}).get_value()) == 1);
    REQUIRE(any_cast<int>(subtrie2.get_subtrie({"a", "b", "d"}).get_value()) == 2);

    ChoiceTrie subtrie3 = trie.get_subtrie({"a", "b", "c"});
    REQUIRE(any_cast<int>(subtrie3.get_value()) == 1);

}

TEST_CASE("set_subtrie", "[trie]" ) {

    ChoiceTrie subtrie {};
    subtrie.set_value({"b", "c"}, 1);

    ChoiceTrie trie {};
    trie.set_subtrie({"a"}, subtrie);
    REQUIRE(trie.subtries().size() == 1);
    REQUIRE(any_cast<int>(trie.get_value({"a", "b", "c"})) == 1);

    // mutating the subtrie also mutates the parent trie
    subtrie.set_value({"b", "c"}, 2);
    REQUIRE(any_cast<int>(trie.get_value({"a", "b", "c"})) == 2);

    // set_subtrie with empty address
    ChoiceTrie trie2 {};
    trie2.set_subtrie({}, subtrie);
    REQUIRE(any_cast<int>(trie2.get_value({"b", "c"})) == 2);
}

TEST_CASE("copy constructor", "[trie]") {
    // Trie behaves like a reference, like torch::Tensor
    ChoiceTrie trie {};
    trie.set_value({"a", "b"}, 1);
    REQUIRE(any_cast<int>(trie.get_value({"a", "b"})) == 1);
    ChoiceTrie trie2 {trie};
    trie2.set_value({"a", "b"}, 2);
    REQUIRE(any_cast<int>(trie.get_value({"a", "b"})) == 2);
}

void walk_choice_trie(const ChoiceTrie& trie, size_t indent) {
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