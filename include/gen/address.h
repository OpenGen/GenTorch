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

#endif //GEN_ADDRESS_H
