#ifndef ADDRESS_SCHEMA_H
#define ADDRESS_SCHEMA_H

#include <vector>
#include <initializer_list>

class AddressSchema {
    std::vector<size_t> keys_;
    size_t hash_;
public:
    AddressSchema(std::vector<size_t> keys);
    AddressSchema(std::initializer_list<size_t> list);
    std::vector<size_t>::const_iterator begin() const;
    std::vector<size_t>::const_iterator end() const;
    size_t hash() const;
    bool operator==(const AddressSchema& other) const;
    void add(size_t key);
    bool contains(size_t key) const;
};

struct AddressSchemaHasher {
    size_t operator()(const AddressSchema& s) const;
};

#endif
