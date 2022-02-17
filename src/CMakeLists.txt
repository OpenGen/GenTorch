set(HEADER_LIST
        "${GenTorch_SOURCE_DIR}/include/gentorch/address.h"
        "${GenTorch_SOURCE_DIR}/include/gentorch/conversions.h"
        "${GenTorch_SOURCE_DIR}/include/gentorch/trie.h"
        "${GenTorch_SOURCE_DIR}/include/gentorch/parameters.h"
        "${GenTorch_SOURCE_DIR}/include/gentorch/trace.h"
        "${GenTorch_SOURCE_DIR}/include/gentorch/dml/dml.h"
        "${GenTorch_SOURCE_DIR}/include/gentorch/distributions/distributions.h"
        "${GenTorch_SOURCE_DIR}/include/gentorch/distributions/normal.h"
        "${GenTorch_SOURCE_DIR}/include/gentorch/distributions/bernoulli.h"
        "${GenTorch_SOURCE_DIR}/include/gentorch/distributions/uniform_continuous.h")

add_library(gentorch_library
        parameters.cpp
        address.cpp
        conversions.cpp
        distributions/normal.cpp
        distributions/bernoulli.cpp
        distributions/uniform_continuous.cpp
        dml/update.cpp
        ${HEADER_LIST})

target_include_directories(gentorch_library PUBLIC ../third_party/gentl/include/ ../include/)
target_link_libraries(gentorch_library ${TORCH_LIBRARIES})

# All users of this library will need at least C++17
target_compile_features(gentorch_library PUBLIC cxx_std_17)

# IDEs should put the headers in a nice place
source_group(
        TREE "${PROJECT_SOURCE_DIR}/include"
        PREFIX "Header Files"
        FILES ${HEADER_LIST})
