#include <cstdio>
#include <iostream>
#include <iterator>
#include <string>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "core/tokenizer.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: yah-tokenize <model.gguf> [--no-special] [--add-bos] "
                 "<text>\n"
                 "       yah-tokenize <model.gguf> --stdin [--no-special] "
                 "[--add-bos]\n");
    return 2;
  }
  try {
    auto gguf = yah::core::Gguf::Open(argv[1]);
    const auto config = yah::core::TokenizerConfig::FromGguf(gguf);
    const auto tokenizer = yah::core::Tokenizer::FromGguf(gguf, config);
    yah::core::TokenizerOptions options;
    std::string text;
    for (int i = 2; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--stdin") {
        text.assign(std::istreambuf_iterator<char>(std::cin),
                    std::istreambuf_iterator<char>());
      } else if (arg == "--no-special") {
        options.parse_special_tokens = false;
      } else if (arg == "--add-bos") {
        options.add_bos = true;
      } else {
        if (!text.empty()) text.push_back(' ');
        text += arg;
      }
    }
    const auto ids = tokenizer.Encode(text, options);
    std::printf("count=%zu\n", ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      std::printf("%u%s", ids[i], i + 1 == ids.size() ? "\n" : " ");
    }
    if (ids.empty()) std::printf("\n");
    std::printf("decoded=%s\n", tokenizer.Decode(ids).c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "yah-tokenize: %s\n", error.what());
    return 1;
  }
}
