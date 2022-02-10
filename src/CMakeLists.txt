set(HEADER_LIST
        "${GenTorch_SOURCE_DIR}/include/gen/utils/randutils.h"
        "${GenTorch_SOURCE_DIR}/include/gen/address.h"
        "${GenTorch_SOURCE_DIR}/include/gen/conversions.h"
        "${GenTorch_SOURCE_DIR}/include/gen/trie.h"
        "${GenTorch_SOURCE_DIR}/include/gen/parameters.h"
        "${GenTorch_SOURCE_DIR}/include/gen/still/sgd.h"
        "${GenTorch_SOURCE_DIR}/include/gen/still/mcmc.h"
        "${GenTorch_SOURCE_DIR}/include/gen/trace.h"
        "${GenTorch_SOURCE_DIR}/include/gen/dml.h"
        "${GenTorch_SOURCE_DIR}/include/gen/distributions/distributions.h"
        "${GenTorch_SOURCE_DIR}/include/gen/distributions/normal.h"
        "${GenTorch_SOURCE_DIR}/include/gen/distributions/bernoulli.h"
        "${GenTorch_SOURCE_DIR}/include/gen/distributions/uniform_continuous.h")

add_library(gentorch_library
        parameters.cpp
        address.cpp
        conversions.cpp
        distributions/normal.cpp
        distributions/bernoulli.cpp
        distributions/uniform_continuous.cpp
        ${HEADER_LIST})

target_include_directories(gentorch_library PUBLIC ../include ../third_party/gentl/include/)
target_link_libraries(gentorch_library ${TORCH_LIBRARIES})

# All users of this library will need at least C++17
target_compile_features(gentorch_library PUBLIC cxx_std_17)

# IDEs should put the headers in a nice place
source_group(
        TREE "${PROJECT_SOURCE_DIR}/include"
        PREFIX "Header Files"
        FILES ${HEADER_LIST})
