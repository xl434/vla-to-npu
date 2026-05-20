cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_C_COMPILER gcc-13)
set(CMAKE_CXX_COMPILER g++-13)

project(vision_encoder_pipeline)

set(BOOST_ROOT /usr/include/boost CACHE STRING "Path to Boost install")
set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT include dir")
set(XRT_LIB_DIR /opt/xilinx/xrt/lib CACHE STRING "Path to XRT lib dir")

find_package(Boost REQUIRED)

add_executable(vision_encoder vision_encoder.cpp)

target_compile_definitions(vision_encoder PUBLIC DISABLE_ABI_CHECK=1)

target_include_directories(vision_encoder PUBLIC
    ${XRT_INC_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_link_directories(vision_encoder PUBLIC
    ${XRT_LIB_DIR}
    ${Boost_LIBRARY_DIRS}
)

target_link_libraries(vision_encoder PUBLIC
    xrt_coreutil
    boost_program_options
    boost_filesystem
    uuid
)
