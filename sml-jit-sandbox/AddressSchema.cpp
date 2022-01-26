#include "AddressSchema.h"

#include <cstdlib>
#include <vector>
#include <algorithm>

size_t keys_hash(const std::vector<size_t>& keys) {
    size_t seed = keys.size();
    for(auto& i : keys) {
        seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

AddressSchema::AddressSchema(std::vector<size_t> keys)
        : keys_{keys}, hash_{keys_hash(keys_)} {}

AddressSchema::AddressSchema(std::initializer_list<size_t> list)
        : keys_{list}, hash_{keys_hash(keys_)} {}

std::vector<size_t>::const_iterator AddressSchema::begin() const {
    return keys_.cbegin();
}

std::vector<size_t>::const_iterator AddressSchema::end() const {
    return keys_.cend();
}

size_t AddressSchema::hash() const {
    return hash_;
}

bool AddressSchema::operator==(const AddressSchema& other) const {
    return other.keys_ == keys_;
}

void AddressSchema::add(size_t key) {
    keys_.emplace_back(key);
    hash_ = keys_hash(keys_);
}

bool AddressSchema::contains(size_t key) const {
    return std::find(keys_.begin(), keys_.end(), key) != keys_.end();
}

size_t AddressSchemaHasher::operator()(const AddressSchema& s) const {
    return s.hash();
};


