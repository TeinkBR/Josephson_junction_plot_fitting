cmake_minimum_required(VERSION 3.14)
project(usadel_solver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Force Apple Clang on macOS to avoid GCC header conflicts
if(APPLE)
    # Use Apple Clang stdlib
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    # Remove any GCC include paths that might be in the environment
    unset(CPLUS_INCLUDE_PATH)
    unset(ENV{CPLUS_INCLUDE_PATH})
endif()

# OpenMP is not available with Apple Clang by default - skip it
message(STATUS "Building without OpenMP (serial execution)")

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
    # Fetch pybind11 if not found
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/usadel_core.cpp
)

# Create the Python module
pybind11_add_module(usadel_cpp 
    ${SOURCES}
    src/python_bindings.cpp
)

# Compiler flags for Apple Clang (no OpenMP)
if(APPLE)
    target_compile_options(usadel_cpp PRIVATE 
        -O3 
        -ffast-math
        -Wall
        -Wextra
    )
else()
    # Non-Apple: try OpenMP
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(usadel_cpp PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_compile_options(usadel_cpp PRIVATE 
        -O3 
        -march=native 
        -ffast-math
        -Wall
        -Wextra
    )
endif()

# Installation
install(TARGETS usadel_cpp DESTINATION .)
