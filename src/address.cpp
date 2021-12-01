#include <gen/address.h>

#include <optional>
#include <memory>
#include <initializer_list>
#include <iostream>

Address::Address(std::initializer_list<std::string>::const_iterator begin,
                 std::initializer_list<std::string>::const_iterator end) {
    if (begin != end) {
        first_.emplace(*begin);
        std::initializer_list<std::string>::const_iterator rest_begin = std::next(begin);
        auto rest_ptr = std::make_unique<Address>(rest_begin, end);
        rest_.emplace(std::move(rest_ptr));
    }
}

const std::string& Address::first() const {
    if (empty()) throw std::domain_error("called first() on an empty address");
        return first_.value();
}

const Address& Address::rest() const {
    if (empty()) throw std::domain_error("called rest() on an empty address");
        return *rest_.value();
}

std::ostream& operator<< (std::ostream& out, const Address& address) {
    out << "{";
    for (const Address* a = &address; !a->empty(); a = &(a->rest())) {
        out << "\"" << a->first() << "\"";
        if (!a->rest().empty()) {
            out << ", ";
        }
    }
    out << "}";
    return out;
}
