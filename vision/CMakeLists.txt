add_executable(vision vision_app.cpp main.cpp)
target_link_libraries(vision PRIVATE vision_proxy)