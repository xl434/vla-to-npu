cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_program(WSL NAMES powershell.exe)

if (NOT WSL)
    set(CMAKE_C_COMPILER gcc-13)
    set(CMAKE_CXX_COMPILER g++-13)
    set(BOOST_ROOT /usr/include/boost CACHE STRING "Path to Boost install")
    set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT include dir")
    set(XRT_LIB_DIR /opt/xilinx/xrt/lib CACHE STRING "Path to XRT lib dir")
else()
    set(BOOST_ROOT C:/Technical/thirdParty/boost_1_83_0 CACHE STRING "Path to Boost install")
    set(XRT_INC_DIR C:/Technical/XRT/src/runtime_src/core/include CACHE STRING "Path to XRT include dir")
    set(XRT_LIB_DIR C:/Technical/xrtNPUfromDLL CACHE STRING "Path to XRT lib dir")
endif()

set(TARGET_NAME test CACHE STRING "Target to be built")
set(MLIR_AIE_DIR /home/dl2239/allo/mlir-aie CACHE STRING "MLIR-AIE root directory")

set(ProjectName ${TARGET_NAME})
set(currentTarget ${TARGET_NAME})

if (WSL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
endif()

project(${ProjectName})

find_package(Boost REQUIRED)

add_executable(${currentTarget}
    ${MLIR_AIE_DIR}/runtime_lib/test_lib/test_utils.cpp
    unified.cpp
    test.cpp
)

target_compile_definitions(${currentTarget} PUBLIC DISABLE_ABI_CHECK=1)

target_include_directories(${currentTarget} PUBLIC
    ${XRT_INC_DIR}
    ${Boost_INCLUDE_DIRS}
    ${MLIR_AIE_DIR}/runtime_lib/test_lib
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
    ${Boost_LIBRARY_DIRS}
)

if (NOT WSL)
    target_link_libraries(${currentTarget} PUBLIC
        xrt_coreutil
        boost_program_options
        boost_filesystem
    )
else()
    target_link_libraries(${currentTarget} PUBLIC
        xrt_coreutil
    )
endif()
