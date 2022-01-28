
# simple_monte_carlo
add_executable(simple_monte_carlo simple_monte_carlo.cpp)
target_link_libraries(simple_monte_carlo PRIVATE gen_library)
target_compile_features(simple_monte_carlo PRIVATE cxx_std_17)

# importance_sampling
add_executable(importance_sampling importance_sampling.cpp)
target_link_libraries(importance_sampling PRIVATE gen_library)
target_compile_features(importance_sampling PRIVATE cxx_std_17)

# sgd
add_executable(sgd sgd.cpp)
target_link_libraries(sgd PRIVATE gen_library)
target_compile_features(sgd PRIVATE cxx_std_17)
