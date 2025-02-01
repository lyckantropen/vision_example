cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

project(Example LANGUAGES CXX CUDA)

# vcpkg dependencies
find_package(OpenCV REQUIRED)

add_library(cuda_proxy INTERFACE)
target_compile_features(cuda_proxy INTERFACE cxx_std_20)
target_include_directories(cuda_proxy INTERFACE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cuda_proxy INTERFACE ${OpenCV_LIBS})

add_subdirectory(vision)

if (PROJECT_IS_TOP_LEVEL)
    # Create symlink to compile_commands.json for IDE to pick it up
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_BINARY_DIR}/compile_commands.json
            ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
    )
endif()