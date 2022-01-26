#!/bin/bash
set -u
set -e

for file in AddressSchema.cpp ChoiceDict.cpp Normal.cpp; do
    clang++ -g -O0 -std=c++17 -c -fPIC ${file}
done

clang++ -g -O0 -std=c++17 sml.cpp AddressSchema.o ChoiceDict.o -ldl -o sml -Wl,--export-dynamic
