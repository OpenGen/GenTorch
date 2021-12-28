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

#include <optional>
#include <memory>
#include <initializer_list>
#include <iostream>

class Address;

/**
 * An immutable type representing a hierarchical address into a call site within a generative function.
 */
class Address {
public:
    Address() : first_(std::optional<std::string>()) { } // empty address
    Address(std::initializer_list<std::string> list) : Address(list.begin(), list.end()) { }
    [[nodiscard]] bool empty() const { return !first_.has_value(); }
    explicit Address(std::initializer_list<std::string>::const_iterator begin,
                     std::initializer_list<std::string>::const_iterator end);
    [[nodiscard]] const std::string& first() const;
    [[nodiscard]] const Address& rest() const;
    friend std::ostream& operator<< (std::ostream& out, const Address& address);
private:
    std::optional<std::string> first_;
    std::optional<std::shared_ptr<Address>> rest_;
};

std::ostream& operator<< (std::ostream& out, const Address& address);
