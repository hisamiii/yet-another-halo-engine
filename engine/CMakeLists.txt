cmake_minimum_required(VERSION 3.20)
project(yah LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# core: GGUF reader, mmap, tensor table, tokenizer. No GPU or NPU dependency.
add_library(yah_core STATIC core/gguf.cpp)
target_include_directories(yah_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(yah_core PRIVATE -Wall -Wextra -Wpedantic)

# yah-dump: print the tensor table and a per-type byte tally. The first
# verification step: its numbers must match the GGUF census byte for byte.
add_executable(yah-dump core/yah_dump.cpp)
target_link_libraries(yah-dump PRIVATE yah_core)
