add_executable(vision main.cpp)
target_link_libraries(vision PRIVATE cuda_proxy)