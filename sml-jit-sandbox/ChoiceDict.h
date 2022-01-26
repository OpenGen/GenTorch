#ifndef CHOICE_DICT_H
#define CHOICE_DICT_H

#include <unordered_map>

#include "AddressSchema.h"

class ChoiceDict {
    std::unordered_map<size_t, double> choices_;
    AddressSchema schema_;
public:
    ChoiceDict();
    void set_value(size_t address, double value);
    double get_value(size_t address) const;
    const AddressSchema& schema() const;
};

#endif
