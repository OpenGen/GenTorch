#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <initializer_list>
#include <cassert>
#include <chrono>
#include <dlfcn.h>
#include <string>
#include <sstream> 
#include <fstream>
#include <filesystem>
#include <optional>
#include <cstdlib>

#include "AddressSchema.h"
#include "ChoiceDict.h"
#include "Interface.h"

using std::cout;
using std::endl;

// ops
std::set<std::string> ops;

void register_op(std::string name) {
    ops.insert(name);
}

bool is_registered(std::string name) {
    auto it = ops.find(name);
    return it != ops.end();
}

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
    std::optional<const Node*> return_node_;
public:
    const std::vector<Node>& nodes() const {
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
        Node& node_ref = nodes_list_.emplace_back(std::move(node));
        auto [_, inserted] = nodes_map_.insert({name, node_ref});
        if (!inserted) {
            std::stringstream ss;
            ss << "name already used found: " << name;
            throw std::logic_error(ss.str());
        }
        return node_ref;
    }
    void set_return_node(const Node& node) {
        return_node_ = &node;
    }
    bool has_return_node() const {
        return return_node_.has_value();
    }
    const Node& get_return_node() const { 
        return *(return_node_.value());
    }
};


// code generation and loading
// TODO use clang API (e.g. maybe its C++ AST facilities) instead

void generate_importance(const Model& model, const AddressSchema& schema, const char* cpp_filename) {
    std::stringstream ss;
    static std::string INDENT = "    ";
    ss << "#include \"Interface.h\"" << std::endl;
    ss << "#include \"Normal.h\"" << std::endl;
    ss << "std::pair<double,double> importance(const ChoiceDict& constraints) {" << std::endl;
    ss << INDENT << "double log_weight_ = 0.0;" << std::endl; // TODO use a reserved symbol that can't be a name
    for (auto node : model.nodes()) {
        if (!is_registered(node.op)) {
            throw std::logic_error("unregistered op");
        }
        if (schema.contains(node.address)) {
            // get value from constraints
            ss << INDENT << "auto " << node.name << " = " << "constraints.get_value(" << node.address << ");" << endl;
            // increment log_weight
            ss << INDENT << "log_weight_ += " << node.op << "().logpdf(" << node.name << ");" << endl; // TODO add args.
        } else {
            // sample value
            ss << INDENT << "auto " << node.name << " = " << node.op << "().simulate();" << endl; // TODO add args.
        }
    }
    if (model.has_return_node()) {
        ss << INDENT << "return {" << model.get_return_node().name << ", log_weight_};" << endl;
    } else {
        ss << INDENT << "return {0.0, log_weight_};" << endl;
    }
    ss << "}" << std::endl;
    cout << ss.str() << endl;
    std::remove(cpp_filename); // TODO just keep one file descriptor open?
    std::ofstream file(cpp_filename);
    file << ss.str();
}

void compile_code(const char* cpp_filename, const char* so_filename) {
    std::stringstream ss;
    ss << "clang++ -g -O0 -std=c++17 AddressSchema.o ChoiceDict.o Normal.o " << cpp_filename << " -Wno-return-type-c-linkage --shared -fPIC -Wl,--export-dynamic -o " << so_filename;
    int code = std::system(ss.str().c_str());
    if (code != 0) {
        throw std::logic_error("error compiling code");
    }
}

typedef std::pair<double,double> (*importance_t)(const ChoiceDict&);

void* open_library(const char* path) {
    void* handle = dlopen(path, RTLD_NOW); // | RTLD_DEEPBIND
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
    const Model& model_;
    std::vector<void*> handles_;
    std::unordered_map<AddressSchema, importance_t, AddressSchemaHasher> schemas_;
    static const char* cpp_filename;
    size_t num_found = 0;
    size_t next_ = 0;
public:
    FunctionTable(const Model& model) : model_{model} {}
    std::string get_fresh_so_filename() {
        std::stringstream fname;
        fname << "./tmp-" << next_ << ".so";
        next_++;
        if (next_ == 0) {
            // there was a wraparound, clear the cache
            for (auto handle : handles_) {
                dlclose(handle);
            }
            handles_.clear();
            schemas_.clear();
        }
        return fname.str();
    }
    importance_t get_importance(const AddressSchema& schema) {
        auto search = schemas_.find(schema);
        if (search == schemas_.end()) {
            // not found
            generate_importance(model_, schema, cpp_filename);
            std::string so_filename = get_fresh_so_filename();
            compile_code(cpp_filename, so_filename.c_str());
            auto handle = open_library(so_filename.c_str());
            handles_.emplace_back(handle);
            importance_t importance = load_importance(handle);
            //cout << "not found; generated and loaded code into: " << ((void*)importance) << endl;
            schemas_.insert({schema, importance});
            return importance;
        } else {
            // found
            num_found += 1;
            importance_t importance = *(search->second);
            //cout << "found; returning cached function pointer: " << ((void*)importance) << endl;
            return importance;
        }
    }
    size_t get_num_found() const { return num_found; }
    ~FunctionTable() {
        for (auto handle : handles_) {
            dlclose(handle);
        }
    }
};

const char* FunctionTable::cpp_filename = "./tmp.cpp";

// example
constexpr size_t FOO = 1;
constexpr size_t BAR = 2;
constexpr size_t BAZ = 3;

void simple_example () {
    register_op("Normal");
    Model model;
    auto a = model.add_node("a", {}, "Normal", 1);
    auto b = model.add_node("b", {}, "Normal", 2);
    auto c = model.add_node("c", {"a", "b"}, "Normal", 3);
    model.set_return_node(c);

    FunctionTable table {model};

    ChoiceDict dict {};
    dict.set_value(1, 1.123);
    dict.set_value(2, 2.234);
    dict.set_value(3, 2.234);
    dict.set_value(4, 2.234);
    dict.set_value(5, 2.234);

    importance_t importance = table.get_importance(dict.schema());
    auto [return_value, log_weight] = importance(dict);
    cout << "return value: " << return_value << ", log weight: " << log_weight << endl;

    // now do it over and over again with different values
    cout << "now using the cached function.." << endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(-1.0, 1.0);
    double sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    size_t num_calls = 1000000;
    for (size_t i = 0; i < num_calls; i++) {
        //dict.set_value(1, static_cast<double>(i));
        importance_t importance = table.get_importance(dict.schema());
        auto [return_value, log_weight] = importance(dict);
        sum += log_weight;
    }
    assert(table.get_num_found() == num_calls); // check that we actually looked it up
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "elapsed (microseconds) " << duration.count() << ", ";
    cout << "calls per microsecond " << num_calls / static_cast<double>(duration.count()) << ", ";
    cout << "sum of log_weight: " << sum << endl;
}

void complex_example() {

    register_op("Normal");
    Model model;
    auto a = model.add_node("a", {}, "Normal", 1);
    auto b = model.add_node("b", {}, "Normal", 2);
    auto c = model.add_node("c", {"a", "b"}, "Normal", 3);
    model.set_return_node(c);
    FunctionTable table {model};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(0.5);
    std::vector<ChoiceDict> queries;
    std::vector<size_t> addrs = {1, 2, 3};
    for (size_t i = 0; i < 1000000; i++) {
        ChoiceDict dict;
        for (auto addr : addrs) {
            if (dist(gen)) {
                dict.set_value(addr, static_cast<double>(i));
            }
        }
        queries.emplace_back(std::move(dict));
    }

    size_t i = 0;
    double sum = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& dict: queries) {
        importance_t importance = table.get_importance(dict.schema());
        auto [return_value, log_weight] = importance(dict);
        if (i % 100000 == 0) {
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            cout << "i: " << i << ", ";
            cout << "elapsed (microseconds) " << duration.count() << ", ";
            cout << "return_value: " << return_value << ", log_weight: " << log_weight << endl;
            cout << endl;
        }
        sum += log_weight;
        i += 1;
    }
    cout << "total log weight: " << sum << endl;

}

int main() {
    //simple_example();
    complex_example();
}
