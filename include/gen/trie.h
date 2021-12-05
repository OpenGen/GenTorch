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

#pragma once

#include <gen/address.h>

#include <stdexcept>
#include <any>
#include <memory>
#include <string>
#include <exception>
#include <sstream>
#include <unordered_map>

#include <torch/torch.h>

using std::shared_ptr;
using std::make_shared;
using std::unique_ptr;
using std::make_unique;
using std::optional;
using std::string;
using std::unordered_map;
using torch::Tensor;

class TrieKeyError : public std::exception {
public:
    explicit TrieKeyError(const Address& address) : msg_ {
        [&](){
            std::stringstream ss;
            ss << "Address not found: " << address;
            return ss.str(); }()
        } { }
    [[nodiscard]] const char* what() const noexcept override {
        return msg_.c_str();
    }
private:
    const std::string msg_;
};

class TrieOverwriteError : public std::exception {
public:
    explicit  TrieOverwriteError(const Address& address) : msg_ {
            [&](){
                std::stringstream ss;
                ss << "Attempted to overwrite at address: " << address;
                return ss.str(); }()
        } {}
    [[nodiscard]] const char* what() const noexcept override {
        return msg_.c_str();
    }
private:
    const std::string msg_;
};

void pretty_print_value(std::ostream& out, const std::any& value);

class Trie {
public:

    // default constructor produces empty trie
    Trie() = default;

    // copy constructor
    Trie(const Trie& value) = default;

    // move constructor
    Trie(Trie&& value) noexcept = default;

    // copy assignment
    Trie& operator= (const Trie& other) = default;

    // move assignment
    Trie& operator= (Trie&& other) = default;

    // destructor
    ~Trie() = default;

    [[nodiscard]] bool empty() const {
        return !value_->has_value() && map_->empty();
    }

    [[nodiscard]] bool has_value() const {
        return value_->has_value();
    }

    [[nodiscard]] const std::any& get_value() const;

    template <typename T>
    void set_value(const T& value, bool overwrite=true);

    [[nodiscard]] const std::any& get_value(const Address& address) const;

    template <typename T>
    void set_value(const Address& address, const T& value, bool overwrite=true);

    [[nodiscard]] bool has_subtrie(const Address& address) const;

    [[nodiscard]] Trie get_subtrie(const Address& address, bool strict=true) const;

    void set_subtrie(const Address& address, const Trie& trie, bool overwrite=true);

    [[nodiscard]] const unordered_map<string,Trie>& subtries() const { return *map_; }

    friend std::ostream& operator<< (std::ostream& out, const Trie& trie);

    // TODO choices() - actually rename to 'values()' or 'flat_iter()'
    // TODO: update(other)
    // TODO: equality testing..
    // TODO: clone()

protected:
    shared_ptr<optional<std::any>> value_ { make_shared<optional<std::any>>(std::nullopt) };
    shared_ptr<unordered_map<string,Trie>> map_ { make_shared<unordered_map<string,Trie>>() };

    static const inline std::string VERT = "\u2502";
    static const inline std::string PLUS = "\u251C";
    static const inline std::string HORZ = "\u2500";
    static const inline std::string LAST = "\u2514";

    static void add_vert_bars(std::string& str, const std::vector<int>& vert_bars);

    static std::string get_indent_vert(int pre, const std::vector<int>& vert_bars);

    static std::string get_indent_vert_last(int pre, const std::vector<int>& vert_bars);

    static std::string get_indent(int pre, const std::vector<int>& vert_bars);

    static std::string get_indent_last(int pre, const std::vector<int>& vert_bars);

    std::ostream& pretty_print(std::ostream& out, int pre, const std::vector<int>& vert_bars,
                               bool extra_space=false) const;
};

std::ostream& operator<< (std::ostream& out, const Trie& trie);