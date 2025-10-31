add_executable(lru lru.cc)
target_compile_features(lru PRIVATE cxx_std_17)
target_link_libraries(lru PRIVATE Threads::Threads)

add_executable(lru-generator lru_generator.cc)
target_compile_features(lru-generator PRIVATE cxx_std_17)
