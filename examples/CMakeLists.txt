
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

# sgd
add_executable(mcmc mcmc.cpp)
target_include_directories(mcmc PRIVATE ../third_party/eigen)
target_link_libraries(mcmc PRIVATE gen_library)
target_compile_features(mcmc PRIVATE cxx_std_17)

# particle_filter
add_executable(particle_filter particle_filter.cpp)
target_include_directories(particle_filter PRIVATE ../third_party/eigen)
target_link_libraries(particle_filter PRIVATE gen_library)
target_compile_features(particle_filter PRIVATE cxx_std_17)
