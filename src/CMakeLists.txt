set(HEADER_LIST
        "${Gen_SOURCE_DIR}/include/gen/utils/randutils.h"
        "${Gen_SOURCE_DIR}/include/gen/address.h"
        "${Gen_SOURCE_DIR}/include/gen/conversions.h"
        "${Gen_SOURCE_DIR}/include/gen/trie.h"
        "${Gen_SOURCE_DIR}/include/gen/parameters.h"
        "${Gen_SOURCE_DIR}/include/gen/still/sgd.h"
        "${Gen_SOURCE_DIR}/include/gen/still/mcmc.h"
        "${Gen_SOURCE_DIR}/include/gen/trace.h"
        "${Gen_SOURCE_DIR}/include/gen/dml.h"
        "${Gen_SOURCE_DIR}/include/gen/distributions/distributions.h"
        "${Gen_SOURCE_DIR}/include/gen/distributions/normal.h"
        "${Gen_SOURCE_DIR}/include/gen/distributions/bernoulli.h"
        "${Gen_SOURCE_DIR}/include/gen/distributions/uniform_continuous.h")

add_library(gen_library
        parameters.cpp
        address.cpp
        conversions.cpp
        distributions/normal.cpp
        distributions/bernoulli.cpp
        distributions/uniform_continuous.cpp
        ${HEADER_LIST})

target_include_directories(gen_library PUBLIC ../include ../third_party/gentl/include/)
target_link_libraries(gen_library ${TORCH_LIBRARIES})

# All users of this library will need at least C++17
target_compile_features(gen_library PUBLIC cxx_std_17)

# IDEs should put the headers in a nice place
source_group(
        TREE "${PROJECT_SOURCE_DIR}/include"
        PREFIX "Header Files"
        FILES ${HEADER_LIST})
