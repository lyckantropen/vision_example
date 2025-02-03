add_executable(vision vision_app.cpp tracking.cpp main.cpp)
target_link_libraries(vision PRIVATE vision_proxy)