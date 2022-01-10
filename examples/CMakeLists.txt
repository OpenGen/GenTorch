add_executable(scrap scrap.cpp)
target_link_libraries(scrap PRIVATE gen_library)
target_compile_features(scrap PRIVATE cxx_std_17)