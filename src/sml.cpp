#include <iostream>
#include <random>
#include <vector>
#include <unordered_map>
#include <initializer_list>
#include <cassert>
#include <chrono>
#include <dlfcn.h>
#include <string>
#include <sstream> 
#include <fstream>
#include <filesystem>
#include <cstdlib>

#include "interface.h"


// address schema

using std::cout;
using std::endl;

size_t keys_hash(const std::vector<size_t>& keys) {
    size_t seed = keys.size();
    for(auto& i : keys) {
        seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

class AddressSchema {
    std::vector<size_t> keys_;
    size_t hash_;
public:
    AddressSchema(std::initializer_list<size_t> list) : keys_{list}, hash_{keys_hash(keys_)} { }
    std::vector<size_t>::const_iterator begin() const { return keys_.cbegin(); }
    std::vector<size_t>::const_iterator end() const { return keys_.cend(); }
    size_t hash() const { return hash_; }
    bool operator==(const AddressSchema& other) const {
        return other.keys_ == keys_;
    }
};

struct AddressSchemaHasher {
    size_t operator()(const AddressSchema& s) const {
        return s.hash();
    }
};

// choice trie

// TODO

// TODO write an SML implementation that generates C++ code for a specific address schema, compiles the code into a dynamic library using clang++
// loads it, puts the function pointer into a hash table that maps fromthe address schema, and then loads it..

// TODO use clang API (e.g. maybe its C++ AST facilities) instead

void generate_code(const AddressSchema& schema, const char* cpp_filename) {
    std::stringstream ss;
    ss << "#include \"interface.h\"" << std::endl;
    ss << "double simulate(double x) { return x + 1; }" << std::endl;
    std::remove(cpp_filename); // TODO just keep one file descriptor open?
    std::ofstream file(cpp_filename);
    file << ss.str();
}

void compile_code(const char* cpp_filename, const char* so_filename) {
    std::stringstream ss;
    ss << "clang++ -std=c++17 --shared " << cpp_filename << " -o " << so_filename;
    std::system(ss.str().c_str());
}

typedef double (*simulate_t)(double);

void* open_library(const char* path) {
    void* handle = dlopen(path, RTLD_NOW);
    if (!handle) {
        std::stringstream err;
        err << "Error. Cannot open the library " << dlerror() << std::endl;
        throw std::logic_error(err.str());
    }
    return handle;
}

simulate_t load_simulate(void* handle) {
    dlerror();
    simulate_t simulate = (simulate_t) dlsym(handle, "simulate");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::stringstream err;
        err << "Error. Cannot load symbol 'simulate': " << dlsym_error << std::endl;
        throw std::logic_error(err.str());
    }
    return simulate;
}

class FunctionTable {
    std::vector<void*> handles_;
    std::unordered_map<AddressSchema, simulate_t, AddressSchemaHasher> schemas_;
    static const char* cpp_filename;
    static const char* so_filename;
public:
    FunctionTable() {}
    simulate_t get_simulate(const AddressSchema& schema) {
        auto search = schemas_.find(schema);
        if (search == schemas_.end()) {
            cout << "generating code.." << endl;
            // not found
            generate_code(schema, cpp_filename);
            compile_code(cpp_filename, so_filename);
            cout << "loading code.." << endl;
            auto handle = open_library(so_filename);
            handles_.emplace_back(handle);
            simulate_t simulate = load_simulate(handle);
            // TODO can we call dlclose()?
            cout << "caching function pointer.." << endl;
            schemas_.insert({schema, simulate});
            return simulate;
        } else {
            // found
            return *(search->second);
        }
    }
    ~FunctionTable() {
        for (auto handle : handles_) {
            dlclose(handle);
        }
    }
};

const char* FunctionTable::cpp_filename = "./tmp.cpp";
const char* FunctionTable::so_filename = "./tmp.so";


constexpr size_t FOO = 1;
constexpr size_t BAR = 2;
constexpr size_t BAZ = 3;

int main() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 3);

    std::vector<AddressSchema> queries;
    for (size_t i = 0; i < 1000000; i++) {
        size_t a = distrib(gen);
        size_t b = distrib(gen);
        size_t c = distrib(gen);
        queries.emplace_back(AddressSchema{a,b,c});
    }

    FunctionTable table;

    //std::unordered_map<AddressSchema, size_t, AddressSchemaHasher> schemas;
    //schemas.insert({AddressSchema{FOO, BAR, BAZ}, 123});

    //size_t found = 0;
    //size_t not_found = 0;
    size_t i = 0;
    double sum = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& schema: queries) {
        simulate_t simulate = table.get_simulate(schema);
        double result = simulate(1.123);
        cout << "i: " << i << ", result: " << result << std::endl;
        sum += result;
        i += 1;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "duration (microseconds): " << duration.count() << ", ";
    cout << "queries per microsecond: " << (static_cast<double>(i) / duration.count()) << ", ";
    cout << "sum: " << sum << endl;
}
