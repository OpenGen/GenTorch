set(HEADER_LIST
        "${Gen_SOURCE_DIR}/include/gen/address.h"
        "${Gen_SOURCE_DIR}/include/gen/conversions.h"
        "${Gen_SOURCE_DIR}/include/gen/trie.h"
        "${Gen_SOURCE_DIR}/include/gen/module.h"
        "${Gen_SOURCE_DIR}/include/gen/trace.h"
        "${Gen_SOURCE_DIR}/include/gen/dml.h"
        "${Gen_SOURCE_DIR}/include/gen/normal.h")

add_library(gen_library
        address.cpp
        normal.cpp
        conversions.cpp
        ${HEADER_LIST})

target_include_directories(gen_library PUBLIC ../include)
target_link_libraries(gen_library "${TORCH_LIBRARIES}")

# All users of this library will need at least C++17
target_compile_features(gen_library PUBLIC cxx_std_17)


################

set(MODULE_HEADER_LIST
        "${Gen_SOURCE_DIR}/include/gen/module.h")

add_executable(gen_modules
        module.cpp
        ${MODULE_HEADER_LIST})

target_include_directories(gen_modules PUBLIC ../include)
target_link_libraries(gen_modules "${TORCH_LIBRARIES}")

# All users of this library will need at least C++17
target_compile_features(gen_modules PUBLIC cxx_std_17)

###############






# IDEs should put the headers in a nice place
source_group(
        TREE "${PROJECT_SOURCE_DIR}/include"
        PREFIX "Header Files"
        FILES ${HEADER_LIST})