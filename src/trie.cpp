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


#include <gen/address.h>

#include <stdexcept>
#include <any>
#include <memory>
#include <string>
#include <exception>
#include <sstream>
#include <unordered_map>

#include <torch/torch.h>

#include <gen/trie.h>

using std::shared_ptr;
using std::make_shared;
using std::unique_ptr;
using std::make_unique;
using std::optional;
using std::string;
using std::unordered_map;
using torch::Tensor;

void pretty_print_value(std::ostream& out, const std::any& value) {
    if (const auto* tensor = std::any_cast<Tensor>(&value)) {
        if (tensor->ndimension() == 0) {
            // a scalar
            if (tensor->dtype() == torch::ScalarType::Float) {
                out << tensor->item<float>();
            } else if (tensor->dtype() == torch::ScalarType::Double) {
                out << tensor->item<double>();
            } else if (tensor->dtype() == torch::ScalarType::Bool) {
                out << (tensor->item<bool>() ? "true" : "false");
            } else {
                std::stringstream ss;
                ss << "printing choices not implemented for Tensor dtype: ";
                ss << tensor->dtype();
                throw std::logic_error(ss.str()); // TODO implement support for this
            }
        } else {
            // not a scalar
            out << "Tensor"; // TODO give a better representation
        }
    } else if (const auto* str = std::any_cast<std::string>(&value)) {
        out << *str;
    } else if (const auto* str = std::any_cast<const char*>(&value)) {
        out << *str;
    } else {
        throw std::logic_error("Could not pretty-print choice value"); // TODO give a better error msg
    }
}

const std::any& Trie::get_value() const {
    try {
        return value_->value();
    } catch (const std::bad_optional_access&) {
        throw TrieKeyError(Address{});
    }
}

template <typename T>
void Trie::set_value(const T& value, bool overwrite) {
    if (!overwrite && !empty()) {
        throw TrieOverwriteError(Address{});
    }
    value_->emplace(value); // calls constructor of std::any with argument of type T
    map_->clear();
}

[[nodiscard]] const std::any& Trie::get_value(const Address& address) const {
    if (address.empty()) {
        return get_value();
    } else {
        try {
            return get_subtrie(address).get_value();
        } catch (const TrieKeyError&) {
            throw TrieKeyError(address);
        }
    }
}

template <typename T>
void Trie::set_value(const Address& address, const T& value, bool overwrite) {
    if (address.empty()) {
        set_value(value, overwrite);
    } else {
        Trie subtrie = get_subtrie(address, false);
        if (!overwrite && !subtrie.empty()) {
            throw TrieOverwriteError(address);
        }
        subtrie.set_value(value);
        set_subtrie(address, subtrie);
    }
}

[[nodiscard]] bool Trie::has_subtrie(const Address& address) const {
    if (address.empty()) {
        return true;
    }
    if (has_value()) {
        return false;
    }
    auto iter = map_->find(address.first());
    if (iter == map_->end()) {
        return false;
    } else {
        return iter->second.has_subtrie(address.rest());
    }
}

[[nodiscard]] Trie Trie::get_subtrie(const Address& address, bool strict) const {
    if (address.empty()) {
        throw TrieKeyError(address);
    }
    if (has_value()) {
        throw TrieKeyError(address);
    }
    auto iter = map_->find(address.first());
    if (iter == map_->end()) {
        if (strict) {
            throw TrieKeyError(address);
        } else {
            return Trie {}; // return new empty choice trie
        }
    } else {
        const Address& rest = address.rest();
        if (rest.empty()) {
            return iter->second;
        } else {
            try {
                return iter->second.get_subtrie(rest, strict);
            } catch (const TrieKeyError&) {
                throw TrieKeyError(address);
            }
        }
    }
}

void Trie::set_subtrie(const Address& address, const Trie& trie, bool overwrite) {
    if (address.empty()) {
        if (!overwrite && !empty()) {
            throw TrieOverwriteError(address);
        }
        if (trie.has_value()) {
            set_value(trie.get_value(), overwrite);
        } else {
            map_ = trie.map_; // copy the shared pointer
        }
    } else {
        value_->reset(); // delete choice value, if there is one
        auto iter = map_->find(address.first());
        Trie rest_subtrie {};
        if (iter == map_->end()) {
            map_->insert({address.first(), rest_subtrie});
        } else {
            rest_subtrie = iter->second; // copy assignment
        }
        try {
            rest_subtrie.set_subtrie(address.rest(), trie, overwrite);
        } catch (const TrieOverwriteError&) {
            throw TrieOverwriteError(address);
        }
    };
}

void Trie::add_vert_bars(std::string& str, const std::vector<int>& vert_bars) {
    for (auto i : vert_bars) {
        str.replace(i, 1, VERT);
    }
}

std::string Trie::get_indent_vert(int pre, const std::vector<int>& vert_bars) {
    std::string str = std::string(pre, ' ') + VERT + "\n";
    add_vert_bars(str, vert_bars);
    return str;
}

std::string Trie::get_indent_vert_last(int pre, const std::vector<int>& vert_bars) {
    return std::string(pre, ' ');
}

std::string Trie::get_indent(int pre, const std::vector<int>& vert_bars) {
    std::string str = std::string(pre, ' ') + PLUS + HORZ + HORZ + " ";
    add_vert_bars(str, vert_bars);
    return str;
}

std::string Trie::get_indent_last(int pre, const std::vector<int>& vert_bars) {
    std::string str = std::string(pre, ' ') + LAST + HORZ + HORZ + " ";
    add_vert_bars(str, vert_bars);
    return str;
}

std::ostream& Trie::pretty_print(std::ostream& out, int pre, const std::vector<int>& vert_bars,
                           bool extra_space) const {
    if (has_value()) {
        pretty_print_value(out, get_value());
        return out;
    }
    auto indent_vert = get_indent_vert(pre, vert_bars);
    auto indent_vert_last = get_indent_vert_last(pre, vert_bars);
    auto indent = get_indent(pre, vert_bars);
    auto indent_last = get_indent_last(pre, vert_bars);
    size_t cur = 0;
    size_t n = map_->size();
    for (const auto& key_and_subtrie : *map_) {
        bool is_last = (cur == n - 1);
        std::string key = key_and_subtrie.first;
        if (extra_space) {
            out << indent_vert;
        }
        if (is_last) {
            out << indent_last;
        } else {
            out << indent;
        }
        if (key_and_subtrie.second.has_value()) {
            out << key << " : ";
            pretty_print_value(out, key_and_subtrie.second.get_value());
            out << std::endl;
        } else {
            out << key << std::endl;
            std::vector<int> next_vert_bars = vert_bars;
            if (!is_last) {
                next_vert_bars.push_back(pre);
            }
            key_and_subtrie.second.pretty_print(out, pre + 4, next_vert_bars);
        }
        cur += 1;
    }
    return out;
}

std::ostream& operator<< (std::ostream& out, const Trie& trie) {
    return trie.pretty_print(out, 0, {});
}