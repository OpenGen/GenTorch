# Testing library
FetchContent_Declare(
        Catch2 
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.13.6)
FetchContent_MakeAvailable(Catch2)
# Adds Catch2::Catch2

# Tests need to be added as executables first
add_executable(testlib
        dml/update.cpp
        test_seed_sequence.cpp
        test_address.cpp
        test_parameters.cpp
        test_trie.cpp
        test_conversions.cpp
        test_dml.cpp
        test_sgd.cpp)

# I'm using C++17 in the test
target_compile_features(testlib PRIVATE cxx_std_17)

target_include_directories(testlib PRIVATE ../third_party/gentl/include/)

# Should be linked to the main library, as well as the Catch2 testing library
target_link_libraries(testlib PRIVATE gentorch_library Catch2::Catch2)

# discover and register Catch test cases
message(STATUS "catch source dir: ${catch2_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(CTest)
include(Catch)
catch_discover_tests(testlib)

## for experimentation
#add_executable(scrap scrap.cpp)
#target_link_libraries(scrap PRIVATE gentorch_library)
