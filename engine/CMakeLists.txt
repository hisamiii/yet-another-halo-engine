cmake_minimum_required(VERSION 3.20)
project(yah LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# core: GGUF reader, mmap, tensor table, tokenizer. No GPU or NPU dependency.
find_package(ICU COMPONENTS uc REQUIRED)

add_library(yah_core STATIC core/gguf.cpp core/config.cpp core/tokenizer.cpp model/weights.cpp)
target_include_directories(yah_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(yah_core PUBLIC ICU::uc)
target_compile_options(yah_core PRIVATE -Wall -Wextra -Wpedantic)

# yah-dump: print the tensor table and a per-type byte tally. The first
# verification step: its numbers must match the GGUF census byte for byte.
add_executable(yah-dump core/yah_dump.cpp)
target_link_libraries(yah-dump PRIVATE yah_core)

# yah-tokenize: encode a string and print the ids, then decode them back. The
# verification step for the tokenizer is the HF golden hash fixture.
add_executable(yah-tokenize core/yah_tokenize.cpp)
target_link_libraries(yah-tokenize PRIVATE yah_core)

# yah-weights: resolve and validate the typed weight table the layer loop needs.
add_executable(yah-weights model/yah_weights.cpp)
target_link_libraries(yah-weights PRIVATE yah_core)

# The HRX-native runners (loom_forward_pp prefill, loom_decode decode, hal_bench,
# hal_run) are built by engine/build_hrx.sh against the HRX runtime library.
