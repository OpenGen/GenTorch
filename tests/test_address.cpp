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

// TODO turn into real test that is part of build

#include "interfaces/address.h"
#include "interfaces/trie.h"

#include <torch/torch.h>
#include <cassert>
#include <iostream>
#include <memory>

using torch::Tensor;
using torch::TensorOptions;
using torch::tensor;

void test_choice_address() {
    Address empty_addr; // empty address
    assert(empty_addr.empty());

    Address a {"x" };
    Address b {"x", "y" };
    assert(!a.empty());
    assert(!b.empty());
    assert(a.first() == "x");
    assert(a.rest().empty());
    assert(b.first() == "x");
    assert(!b.rest().empty());
    assert(b.rest().first() == "y");

    Address addr {"0", "1"};
    addr = addr.rest();
    assert(addr.first() == "1");

    Address addr_ostream {"0", "1", "2"};
    std::stringstream ss;
    ss << addr_ostream;
    assert(ss.str() == "{\"0\", \"1\", \"2\"}");
    std::stringstream ss_empty_addr;
    ss_empty_addr << Address();
    assert(ss_empty_addr.str() == "{}");

    std::string f = "x";
    Address c {f };
    f = "y";
    assert(c.first() == "x");

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

void test_choice_trie() {
    Tensor t1 = tensor(1.123);
    Tensor t2 = tensor(2.234);

    Trie trie { };
    assert(!trie.has_value());
    trie.set_value(Address{"a", "b", "c", "d"}, t1);
    trie.set_value(Address{"a", "b", "e", "f"}, t2);
    trie.set_value(Address{"a", "b", "g"}, std::string("asdf"));
    assert(std::any_cast<Tensor>(trie.get_value(Address{"a", "b", "c", "d"})).equal(t1));
    assert(std::any_cast<Tensor>(trie.get_value(Address{"a", "b", "e", "f"})).equal(t2));

    std::cout << trie << std::endl;
    std::cout << trie.get_subtrie(Address{"a"}) << std::endl;
    std::cout << trie.get_subtrie(Address{"a"}) << std::endl;

    Tensor bool_tensor = tensor(0, TensorOptions().dtype(torch::ScalarType::Bool));
    Trie trie_with_value {};
    trie_with_value.set_value(bool_tensor);
    assert(trie_with_value.has_value());
    std::cout << trie_with_value << std::endl;

    Trie super_trie { };
    super_trie.set_subtrie(Address{"a", "b", "c", "d", "e"}, trie_with_value);
    std::cout << super_trie << std::endl;
    trie_with_value.set_value(tensor(1, TensorOptions().dtype(torch::ScalarType::Bool)));
    super_trie.set_subtrie(Address{"x", "y", "z"}, trie_with_value);
    std::cout << trie_with_value << std::endl;
    std::cout << super_trie << std::endl;
    assert(!super_trie.has_value());

    walk_choice_trie(super_trie, 0);

}

int main() {

    test_choice_address();
    test_choice_trie();

    // basic experiments with std::string
    std::string str {"asdf"};
    std::string str_copy { str }; // copy constructor
    std::string str_assignment;
    str_assignment = str; // assignment operator
    std::cout << "before append str is: " << str << std::endl;
    std::cout << "before append str_copy is: " << str_copy << std::endl;
    std::cout << "before append str_assignment is: " << str_assignment << std::endl;
    str.append("asdf");
    std::cout << "after append str is: " << str << std::endl;
    std::cout << "after append str_copy is: " << str_copy << std::endl;
    std::cout << "after append str_assignment is: " << str_assignment << std::endl;

}