#ifndef GEN_ADDRESS_H
#define GEN_ADDRESS_H

#include <optional>
#include <memory>
#include <initializer_list>
#include <iostream>

class Address;

class Address {
public:

    Address() : first_(std::optional<std::string>()) { } // empty address

    explicit Address(std::initializer_list<std::string>::const_iterator begin,
                     std::initializer_list<std::string>::const_iterator end) {
        if (begin != end) {
            first_.emplace(*begin);
            std::initializer_list<std::string>::const_iterator rest_begin = std::next(begin);
            auto rest_ptr = std::make_unique<Address>(rest_begin, end);
            rest_.emplace(std::move(rest_ptr));
        }
    }

    Address(std::initializer_list<std::string> list) : Address(list.begin(), list.end()) { }

    [[nodiscard]] bool empty() const { return !first_.has_value(); }

    [[nodiscard]] const std::string& first() const {
        if (empty()) throw std::domain_error("called first() on an empty address");
        return first_.value();
    }

    [[nodiscard]] const Address& rest() const {
        if (empty()) throw std::domain_error("called rest() on an empty address");
        return *rest_.value();
    }

    friend std::ostream& operator<< (std::ostream& out, const Address& address);

private:
    std::optional<std::string> first_;
    std::optional<std::shared_ptr<Address>> rest_;
};

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

#endif //GEN_ADDRESS_H
