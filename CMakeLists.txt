cmake_minimum_required(VERSION 3.5)

project(fft)

set(HALIDE_DISTRIB_DIR ${PROJECT_SOURCE_DIR}/ext/halide)
include("${HALIDE_DISTRIB_DIR}/halide.cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TRACE_STORES "Tracing stores" OFF)
if( TRACE_STORES )
	message("Tracing stores")
	add_definitions(-DTRACE_STORES)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -pthread")
set(CMAKE_CXX_DEBUG "-g")
set(CMAKE_CXX_RELEASE "-Ofast")

if(${CMAKE_BUILD_TYPE} MATCHES "Release")
	message("Release build with: ${CMAKE_CXX_RELEASE}")
endif()

include_directories("${HALIDE_INCLUDE_DIR}" "${HALIDE_TOOLS_DIR}")
halide_generator(fft.generator SRCS src/fft.cpp GENERATOR_NAME fftgen)
halide_library_from_generator(fft fftsgen)
