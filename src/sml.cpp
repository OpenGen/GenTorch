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


struct AddressSchemaHasher {
    size_t operator()(const AddressSchema& s) const {
        return s.hash();
    }
};

// choice dict

ChoiceDict::ChoiceDict() : schema_{} {}

void ChoiceDict::set_value(size_t address, double value) {
    auto [_, inserted] = choices_.insert({address, value});
    if (!inserted) {
        throw std::logic_error("key already added");
    }
    schema_.add(address);
}

double ChoiceDict::get_value(size_t address) const {
    return choices_.at(address);
}

const AddressSchema& ChoiceDict::schema() const {
    return schema_;
}

// ops

class Op {}

std::unordered_map<std::string,const Op&> ops;

void register_op(std::string name, const Op& op) {
    ops.insert({name, op});
}

const Op& op get_op(std::string name) {
    return ops.at(name);
}

class Normal : Op {
public:
    // TODO add args
    double logpdf(double value) {
        return 0.0; // TODO
    }
    double simulate() {
        return 0.0; // TODO
    }
};

constexpr Normal normal;
register_op("normal", normal);

// a simple DAG IR

struct Node {
    std::string name;
    std::string op;
    std::vector<Node> parents;
    size_t address;
};

class Model {
    std::vector<Node> nodes_list_;
    std::unordered_map<std::string,Node&> nodes_map_;
public:
    const std::vector<Node>& nodes() {
        return nodes_list_;
    }
    const Node& add_node(
            std::string name, std::vector<std::string> arguments,
            std::string op, size_t address) {
        std::vector<Node> parents;
        for (auto parent_name : arguments) {
            parents.emplace_back(nodes_map_.at(parent_name));
        }
        Node node {name, op, std::move(parents), address};
        auto [_, inserted] = nodes_map_.insert({name, std::move(node)});
        if (!inserted) {
            std::stringstream ss;
            ss << "name already used found: " << name;
            throw std::logic_error(ss.str());
        }
        return nodes_list_.emplace_back(std::move(node));
    }
};


// code generation and loading
// TODO use clang API (e.g. maybe its C++ AST facilities) instead

void generate_importance(const Model& model, const AddressSchema& schema, const char* cpp_filename) {
    std::stringstream ss;
    static std::string INDENT = "    ";
    ss << "#include \"interface.h\"" << std::endl;
    ss << "std::pair<double,double> importance(const ChoiceDict& constraints) {" << std::endl;
    ss << INDENT << "double log_weight_ = 0.0;" << std::endl; // TODO use a reserved symbol that can't be a name
    for (auto node : model.nodes()) {
        if (schema.contains(node.address)) {
            // get value from constraints
            ss << INDENT << model.name << " = " << "constraints.at(" << node.address << ");" << endl;
            // increment log_weight
            ss << INDENT << "log_weight_ += " << model.op << ".logpdf()" << endl; // TODO add args.
        } else {
            // sample value
            ss << INDENT << model.name << " = " << model.op << ".simulate()" << endl; // TODO add args.
        }
    }
    ss << "}" << std::endl;
    cout << ss.str() << endl;
    std::remove(cpp_filename); // TODO just keep one file descriptor open?
    std::ofstream file(cpp_filename);
    file << ss.str();
}

void compile_code(const char* cpp_filename, const char* so_filename) {
    std::stringstream ss;
    ss << "clang++ -std=c++17 --shared " << cpp_filename << " -o " << so_filename;
    std::system(ss.str().c_str());
}

typedef double (*importance_t)(double);

void* open_library(const char* path) {
    void* handle = dlopen(path, RTLD_NOW);
    if (!handle) {
        std::stringstream err;
        err << "Error. Cannot open the library " << dlerror() << std::endl;
        throw std::logic_error(err.str());
    }
    return handle;
}

importance_t load_importance(void* handle) {
    dlerror();
    importance_t importance = (importance_t) dlsym(handle, "importance");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::stringstream err;
        err << "Error. Cannot load symbol 'importance': " << dlsym_error << std::endl;
        throw std::logic_error(err.str());
    }
    return importance;
}

class FunctionTable {
    std::vector<void*> handles_;
    std::unordered_map<AddressSchema, importance_t, AddressSchemaHasher> schemas_;
    static const char* cpp_filename;
    static const char* so_filename;
public:
    FunctionTable() {}
    importance_t get_importance(const AddressSchema& schema) {
        auto search = schemas_.find(schema);
        if (search == schemas_.end()) {
            // not found
            generate_importance(schema, cpp_filename);
            compile_code(cpp_filename, so_filename);
            auto handle = open_library(so_filename);
            handles_.emplace_back(handle);
            importance_t importance = load_importance(handle);
            schemas_.insert({schema, importance});
            return importance;
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

// example

constexpr size_t FOO = 1;
constexpr size_t BAR = 2;
constexpr size_t BAZ = 3;

int main() {
    Model model;
    model.add_node("a", {}, "op1");
    model.add_node("b", {}, "op2");
    model.add_node("c", {"a", "b"}, "op3");
}

//int main() {
//
    //std::random_device rd;
    //std::mt19937 gen(rd());
    //std::uniform_int_distribution<> distrib(1, 3);
//
    //std::vector<AddressSchema> queries;
    //for (size_t i = 0; i < 1000000; i++) {
        //size_t a = distrib(gen);
        //size_t b = distrib(gen);
        //size_t c = distrib(gen);
        //queries.emplace_back(AddressSchema{a,b,c});
    //}
//
    //FunctionTable table;
//
    ////std::unordered_map<AddressSchema, size_t, AddressSchemaHasher> schemas;
    ////schemas.insert({AddressSchema{FOO, BAR, BAZ}, 123});
//
    ////size_t found = 0;
    ////size_t not_found = 0;
    //size_t i = 0;
    //double sum = 0;
    //auto start = std::chrono::high_resolution_clock::now();
    //for (const auto& schema: queries) {
        //importance_t importance = table.get_importance(schema);
        //double result = importance(1.123);
        //if (i % 100000 == 0) {
            //auto stop = std::chrono::high_resolution_clock::now();
            //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            //cout << "i: " << i << ", ";
            //cout << "elapsed (microseconds) " << duration.count() << ", ";
            //cout << "result: " << result;
            //cout << endl;
        //}
        //sum += result;
        //i += 1;
    //}
//}
