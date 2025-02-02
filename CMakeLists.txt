cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

project(Example LANGUAGES CXX CUDA)

# vcpkg dependencies
find_package(OpenCV REQUIRED)
find_package(argparse CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_library(vision_proxy INTERFACE)
target_compile_features(vision_proxy INTERFACE cxx_std_23)
target_include_directories(vision_proxy INTERFACE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision_proxy INTERFACE ${OpenCV_LIBS} argparse::argparse Eigen3::Eigen)

add_subdirectory(vision)

if (PROJECT_IS_TOP_LEVEL)
    # Create symlink to compile_commands.json for IDE to pick it up
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_BINARY_DIR}/compile_commands.json
            ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
    )
endif()