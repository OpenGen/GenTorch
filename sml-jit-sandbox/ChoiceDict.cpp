#include "ChoiceDict.h"

ChoiceDict::ChoiceDict() : schema_{} {}

void ChoiceDict::set_value(size_t address, double value) {
    auto [_, inserted] = choices_.insert({address, value});
    if (inserted) {
        schema_.add(address);
    }
}

double ChoiceDict::get_value(size_t address) const {
    return choices_.at(address);
}

const AddressSchema& ChoiceDict::schema() const {
    return schema_;
}
