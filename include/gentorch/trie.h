/* Copyright 2021-2022 Massachusetts Institute of Technology

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

#ifndef GENTORCH_TRIE_H
#define GENTORCH_TRIE_H

#include <gentorch/address.h>
#include <torch/torch.h>

#include <stdexcept>
#include <any>
#include <memory>
#include <string>
#include <exception>
#include <sstream>
#include <unordered_map>

using std::shared_ptr;
using std::make_shared;
using std::unique_ptr;
using std::make_unique;
using std::optional;
using std::string;
using std::unordered_map;
using torch::Tensor;
using gentorch::Address;

namespace gentorch {

    class TrieKeyError : public std::exception {
    public:
        explicit TrieKeyError(const Address &address) : msg_{
                [&]() {
                    std::stringstream ss;
                    ss << "Address not found: " << address;
                    return ss.str();
                }()
        } {}

        [[nodiscard]] const char *what() const noexcept override {
            return msg_.c_str();
        }

    private:
        const std::string msg_;
    };

    class TrieOverwriteError : public std::exception {
    public:
        explicit TrieOverwriteError(const Address &address) : msg_{
                [&]() {
                    std::stringstream ss;
                    ss << "Attempted to overwrite at address: " << address;
                    return ss.str();
                }()
        } {}

        [[nodiscard]] const char *what() const noexcept override {
            return msg_.c_str();
        }

    private:
        const std::string msg_;
    };

    /**
     * A tree whose nodes are indexed by `::Address`'s, with values stored at leaf nodes.
     */
    template<typename ValueType>
    class Trie {
    public:

        // default constructor produces empty trie
        Trie() = default;

        // copy constructor
        Trie(const Trie &value) = default;

        // move constructor
        Trie(Trie &&value) noexcept = default;

        // copy assignment
        Trie &operator=(const Trie &other) = default;

        // move assignment
        Trie &operator=(Trie &&other) = default;

        // destructor
        ~Trie() = default;

        /**
         *
         * @return `true` if the Trie has no value and no subtries.
         */
        [[nodiscard]] bool empty() const {
            return !value_->has_value() && map_->empty();
        }

        /**
         *
         * @return `true` if the Trie has a value (in which case it must not have any subtries)
         */
        [[nodiscard]] bool has_value() const {
            return value_->has_value();
        }

        /**
         * Return a reference to the stored value, or throw an error if there is no value.
         *
         * Note that `any_cast<T>(get_value())` can be used to obtain a typed value.
         * @return
         */
        [[nodiscard]] ValueType &get_value() const;

        /**
         * Return a reference to value stored at the given address, or throw an error if there is no value.
         * @return
         */
        [[nodiscard]] ValueType &get_value(const Address &address) const;

        /**
         * Set the value of the trie to a copy of the given value, clearing any other value and any subtries if `overwrite` is `true`.
         * @tparam T
         * @param value The value to be copied and stored in the trie  (note that `Tensor` behaves like a shared pointer, and copying it does not copy the underlying data).
         * @param overwrite
         * @return A reference to the value stored in the trie.
         */
        ValueType &set_value(ValueType value, bool overwrite = true) {
            if (!overwrite && !empty()) {
                throw TrieOverwriteError(Address{});
            }
            value_->emplace(std::move(value));
            map_->clear();
            return value_->value();
        }

        void clear() {
            value_->reset();
            map_->clear();
        }

        /**
         * Set the value at the given address to a copy of the given value, clearing any other value and any subtries if `overwrite` is `true`.
         * @tparam T
         * @param value The value to be copied and stored in the trie  (note that `Tensor` behaves like a shared pointer, and copying it does not copy the underlying data).
         * @param overwrite
         * @return A reference to the value stored in the trie.
         */
        ValueType &set_value(const Address &address, ValueType value, bool overwrite = true) {
            if (address.empty()) {
                return set_value(std::move(value), overwrite);
            } else {
                // TODO this is non-optimized
                Trie subtrie = get_subtrie(address, false);
                if (!overwrite && !subtrie.empty()) {
                    throw TrieOverwriteError(address);
                }
                subtrie.set_value(std::move(value));
                set_subtrie(address, std::move(subtrie));
                return get_value(address);
            }
        }

        [[nodiscard]] bool has_subtrie(const Address &address) const;

        [[nodiscard]] Trie<ValueType> get_subtrie(const Address &address, bool strict = true) const;

        void set_subtrie(const Address &address, Trie<ValueType> trie, bool overwrite = true);

        // TODO return value is only a non-const reference because we need the empty iterator functionality of the map
        [[nodiscard]] unordered_map<string, Trie<ValueType>> &subtries() const { return *map_; }


        template<typename T>
        friend std::ostream &operator<<(std::ostream &out, const Trie<T> &trie);


        // TODO choices() - actually rename to 'values()' or 'flat_iter()'
        // TODO: update(other)
        // NOTE: we cannot implement == (at least not easily) since we use std::any for values
        // if we restricted choices to be Tensor, then we could implement == (and approximate variants)
        // TODO: clone()

        // TODO why are these not private?

        void remove_empty_subtries();


    protected:
        shared_ptr<optional<ValueType>> value_{make_shared<optional<ValueType>>()};
        shared_ptr<unordered_map<string, Trie<ValueType>>> map_{make_shared<unordered_map<string, Trie<ValueType>>>()};

        std::ostream &pretty_print(std::ostream &out, int pre, const std::vector<int> &vert_bars,
                                   bool extra_space = false) const;
    private:
        void remove_empty_subtries(unordered_map<string, Trie<ValueType>>& map);

    };


    template<typename T>
    std::ostream &operator<<(std::ostream &out, const Trie<T> &trie);

    typedef Trie<std::any> ChoiceTrie;

//




    template<typename ValueType>
    ValueType &Trie<ValueType>::get_value() const {
        try {
            return value_->value();
        } catch (const std::bad_optional_access &) {
            throw TrieKeyError(Address{});
        }
    }

    template<typename ValueType>
    [[nodiscard]] ValueType &Trie<ValueType>::get_value(const Address &address) const {
        if (address.empty()) {
            return get_value();
        } else {
            try {
                return get_subtrie(address).get_value();
            } catch (const TrieKeyError &) {
                throw TrieKeyError(address);
            }
        }
    }

    template<typename ValueType>
    [[nodiscard]] bool Trie<ValueType>::has_subtrie(const Address &address) const {
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

    template<typename ValueType>
    void Trie<ValueType>::remove_empty_subtries(unordered_map<string, Trie<ValueType>>& map) {
        for (auto& [address, subtrie] : map) {
            if (subtrie.empty())
                map.erase(address);
            else
                remove_empty_subtries(subtrie);
        }
    }


template<typename ValueType>
void Trie<ValueType>::remove_empty_subtries() {
    remove_empty_subtries(*map_);
}


template<typename ValueType>
    [[nodiscard]] Trie<ValueType> Trie<ValueType>::get_subtrie(const Address &address, bool strict) const {
        if (address.empty()) {
            return *this; // calls copy constructor, returns Trie with shared data
        }
        if (has_value()) {
            throw TrieKeyError(address);
        }
        auto iter = map_->find(address.first());
        if (iter == map_->end()) {
            if (strict) {
                throw TrieKeyError(address);
            } else {
                return Trie<ValueType>{}; // return new empty choice trie
            }
        } else {
            const Address &rest = address.rest();
            if (rest.empty()) {
                return iter->second;
            } else {
                try {
                    return iter->second.get_subtrie(rest, strict);
                } catch (const TrieKeyError &) {
                    throw TrieKeyError(address);
                }
            }
        }
    }

    template<typename ValueType>
    void Trie<ValueType>::set_subtrie(const Address &address, Trie<ValueType> trie, bool overwrite) {
        if (address.empty()) {
            if (!overwrite && !empty()) {
                throw TrieOverwriteError(address);
            }
            // copy the shared pointers
            value_ = trie.value_;
            map_ = trie.map_;
        } else {
            value_->reset(); // delete choice value, if there is one
            auto iter = map_->find(address.first());
            Trie<ValueType> *rest_subtrie_ptr = nullptr;
            if (iter == map_->end()) {
                // there is no subtrie at this key, add it
                Trie<ValueType> rest_subtrie{};
                map_->insert({address.first(), std::move(rest_subtrie)});
                rest_subtrie_ptr = &((*map_)[address.first()]);
            } else {
                // there is a subtrie at this key
                rest_subtrie_ptr = &iter->second; // copy assignment
            }
            try {
                rest_subtrie_ptr->set_subtrie(address.rest(), std::move(trie), overwrite);
            } catch (const TrieOverwriteError &) {
                throw TrieOverwriteError(address);
            }
        };
    }

    static const std::string VERT = "\u2502";
    static const std::string PLUS = "\u251C";
    static const std::string HORZ = "\u2500";
    static const std::string LAST = "\u2514";

    inline void pretty_print_value(std::ostream &out, const std::any &value) {
        if (const auto *tensor = std::any_cast<Tensor>(&value)) {
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
        } else if (const auto *str = std::any_cast<std::string>(&value)) {
            out << *str;
        } else if (const auto *str = std::any_cast<const char *>(&value)) {
            out << *str;
        } else if (const auto *i = std::any_cast<int>(&value)) {
            out << *i;
        } else {
            throw std::logic_error("Could not pretty-print choice value"); // TODO give a better error msg
        }
    }

    inline void add_vert_bars(std::string &str, const std::vector<int> &vert_bars) {
        for (auto i: vert_bars) {
            str.replace(i, 1, VERT);
        }
    }

    inline std::string get_indent_vert(int pre, const std::vector<int> &vert_bars) {
        std::string str = std::string(pre, ' ') + VERT + "\n";
        add_vert_bars(str, vert_bars);
        return str;
    }

    inline std::string get_indent_vert_last(int pre, const std::vector<int> &vert_bars) {
        return std::string(pre, ' ');
    }

    inline std::string get_indent(int pre, const std::vector<int> &vert_bars) {
        std::string str = std::string(pre, ' ') + PLUS + HORZ + HORZ + " ";
        add_vert_bars(str, vert_bars);
        return str;
    }

    inline std::string get_indent_last(int pre, const std::vector<int> &vert_bars) {
        std::string str = std::string(pre, ' ') + LAST + HORZ + HORZ + " ";
        add_vert_bars(str, vert_bars);
        return str;
    }


    template<typename ValueType>
    std::ostream &Trie<ValueType>::pretty_print(std::ostream &out, int pre, const std::vector<int> &vert_bars,
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
        for (const auto &key_and_subtrie: *map_) {
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


    template<typename ValueType>
    std::ostream &operator<<(std::ostream &out, const Trie<ValueType> &trie) {
        return trie.pretty_print(out, 0, {});
    }

}

#endif // GENTORCH_TRIE_H
