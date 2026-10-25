cmake_minimum_required(VERSION 3.20)
project(yah LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# core: GGUF and HFQ readers, mmap, tensor table, tokenizer. No GPU or NPU dependency.
find_package(ICU COMPONENTS uc REQUIRED)

find_package(Threads REQUIRED)
add_library(yah_core STATIC core/gguf.cpp core/gguf_convert.cpp core/gguf_quant.cpp core/hfq.cpp core/config.cpp core/tokenizer.cpp model/weights.cpp)
target_include_directories(yah_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(yah_core PUBLIC ICU::uc Threads::Threads)
target_compile_options(yah_core PRIVATE -Wall -Wextra -Wpedantic)

# yah-dump: print the tensor table and a per-type byte tally. The first
# verification step: its numbers must match the GGUF census byte for byte.
add_executable(yah-dump core/yah_dump.cpp)
target_link_libraries(yah-dump PRIVATE yah_core)

# yah-tokenize: encode a string and print the ids, then decode them back. The
# verification step for the tokenizer is the HF golden hash fixture.
add_executable(yah-tokenize core/yah_tokenize.cpp)
target_link_libraries(yah-tokenize PRIVATE yah_core)

# json_check: core/json.hpp on fixed cases.
add_executable(json_check tests/json_check.cc)
target_include_directories(json_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# gguf_rows: the GGUF conversion's dequantizers and MQ4 quantizer on files (tests/gguf_check.py).
add_executable(gguf_rows tests/gguf_rows.cc)
target_link_libraries(gguf_rows PRIVATE yah_core)

# yah-weights: resolve and validate the typed weight table the layer loop needs.
add_executable(yah-weights model/yah_weights.cpp)
target_link_libraries(yah-weights PRIVATE yah_core)

# Serving without the GPU: yah_server_fake (yah_server with only --fake), the yah_serve_test HTTP tests, the yah_chat client.
add_library(yah_http STATIC serve/http.cpp)
target_include_directories(yah_http PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(yah_http PUBLIC Threads::Threads)
add_executable(yah_server_fake serve/yah_server.cc serve/chat_template.cpp serve/responses.cpp serve/chat_completions.cpp)
target_compile_definitions(yah_server_fake PRIVATE YAH_SERVE_FAKE_ONLY)
target_link_libraries(yah_server_fake PRIVATE yah_core yah_http)
add_executable(yah_serve_test serve/yah_serve_test.cc serve/chat_template.cpp serve/responses.cpp serve/chat_completions.cpp)
target_link_libraries(yah_serve_test PRIVATE yah_http)
add_executable(yah_chat serve/yah_chat.cc)
target_link_libraries(yah_chat PRIVATE yah_http)

# The HRX-native runners (loom_forward_pp prefill, loom_decode decode, hal_run) and yah_server are built by engine/build_hrx.sh against the HRX runtime library.
