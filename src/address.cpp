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

#include <optional>
#include <memory>
#include <initializer_list>
#include <iostream>

namespace gen {

    Address::Address(std::initializer_list<std::string>::const_iterator begin,
                     std::initializer_list<std::string>::const_iterator end) {
        if (begin != end) {
            first_.emplace(*begin);
            std::initializer_list<std::string>::const_iterator rest_begin = std::next(begin);
            auto rest_ptr = std::make_unique<Address>(rest_begin, end);
            rest_.emplace(std::move(rest_ptr));
        }
    }

    const std::string &Address::first() const {
        if (empty()) throw std::domain_error("called first() on an empty address");
        return first_.value();
    }

    const Address &Address::rest() const {
        if (empty()) throw std::domain_error("called rest() on an empty address");
        return *rest_.value();
    }

    std::ostream &operator<<(std::ostream &out, const Address &address) {
        out << "{";
        for (const Address *a = &address; !a->empty(); a = &(a->rest())) {
            out << "\"" << a->first() << "\"";
            if (!a->rest().empty()) {
                out << ", ";
            }
        }
        out << "}";
        return out;
    }

}