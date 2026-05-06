# CMakeLists.txt — fused im2col + GEMM preprocessing unified C++ test
#
# Build parameters (pass via -D):
#   -DBOOST_ROOT      Path to Boost install
#   -DXRT_INC_DIR     Full path to XRT include dir
#   -DXRT_LIB_DIR     Path to XRT lib dir (xrt_coreutil)
#   -DMLIR_AIE_DIR    MLIR-AIE root (for test_utils)
#   -DTARGET_NAME     Executable name (default: test)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_program(WSL NAMES powershell.exe)

if (NOT WSL)
    set(CMAKE_C_COMPILER   gcc-13)
    set(CMAKE_CXX_COMPILER g++-13)
    set(BOOST_ROOT    /usr/include/boost   CACHE STRING "Boost install path")
    set(XRT_INC_DIR   /opt/xilinx/xrt/include CACHE STRING "XRT include dir")
    set(XRT_LIB_DIR   /opt/xilinx/xrt/lib     CACHE STRING "XRT lib dir")
else()
    set(BOOST_ROOT    C:/Technical/thirdParty/boost_1_83_0         CACHE STRING "Boost install path")
    set(XRT_INC_DIR   C:/Technical/XRT/src/runtime_src/core/include CACHE STRING "XRT include dir")
    set(XRT_LIB_DIR   C:/Technical/xrtNPUfromDLL                   CACHE STRING "XRT lib dir")
endif()

set(TARGET_NAME  test                             CACHE STRING "Executable name")
set(MLIR_AIE_DIR /home/dl2239/allo/mlir-aie      CACHE STRING "MLIR-AIE root dir")

project(${TARGET_NAME})
find_package(Boost REQUIRED)

add_executable(${TARGET_NAME}
    ${MLIR_AIE_DIR}/runtime_lib/test_lib/test_utils.cpp
    fused_unified.cpp
    test.cpp
)

target_compile_definitions(${TARGET_NAME} PUBLIC DISABLE_ABI_CHECK=1)

target_include_directories(${TARGET_NAME} PUBLIC
    ${XRT_INC_DIR}
    ${Boost_INCLUDE_DIRS}
    ${MLIR_AIE_DIR}/runtime_lib/test_lib
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_directories(${TARGET_NAME} PUBLIC
    ${XRT_LIB_DIR}
    ${Boost_LIBRARY_DIRS}
)

if (NOT WSL)
    target_link_libraries(${TARGET_NAME} PUBLIC
        xrt_coreutil
        boost_program_options
        boost_filesystem
    )
else()
    target_link_libraries(${TARGET_NAME} PUBLIC
        xrt_coreutil
    )
endif()
