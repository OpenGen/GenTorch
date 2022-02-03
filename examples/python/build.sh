#!/bin/bash
g++-11 -O3 -Wall -shared -std=c++17 -fPIC $(python3-config --includes) -I../../third_party/pybind11/include model.cpp -I../../include bindings.cpp -o tracker$(python3-config --extension-suffix)
