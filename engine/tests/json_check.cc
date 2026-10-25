// core/json.hpp on fixed cases whose expected text came from nlohmann::ordered_json (the library it replaced): number
// layout, escapes, invalid UTF-8 replacement, inputs RFC 8259 rejects, brace-list objects and key order.
#include <cstdio>
#include <string>

#include "core/json.hpp"

using yah::core::Json;

namespace {

int bad = 0;
void Expect(bool ok, const std::string& what) {
  if (!ok) ++bad, std::printf("FAIL %s\n", what.c_str());
}
void Round(const char* in, const char* want) {
  std::string got;
  try {
    got = Json::parse(in).dump();
  } catch (const std::exception& e) {
    got = std::string("throws: ") + e.what();
  }
  Expect(got == want, std::string("parse(") + in + ").dump() = " + got + ", want " + want);
}

}  // namespace

int main() {
  Round("1e5", "100000.0");
  Round("0.5e-3", "0.0005");
  Round("1e-5", "1e-05");
  Round("1e16", "1e+16");
  Round("-0", "0");
  Round("-1.5", "-1.5");
  Round("0.1", "0.1");
  Round("123456789012345678", "123456789012345678");
  Round("9223372036854775808", "9223372036854775808");
  Round("\"\\ud83d\\ude00\\u00e9\\/\"", "\"\xf0\x9f\x98\x80\xc3\xa9/\"");
  Round(" {\"b\" : [ 1 , {\"a\":null} ], \"a\": true } ", "{\"b\":[1,{\"a\":null}],\"a\":true}");
  Round("{\"k\":1,\"j\":2,\"k\":3}", "{\"k\":3,\"j\":2}");
  Round("\"\\u0001\\t\\\"\"", "\"\\u0001\\t\\\"\"");
  for (const char* m : {"",
                        "{",
                        "[1,]",
                        "{\"a\":}",
                        "01",
                        "1.",
                        "-",
                        "1e",
                        "\"\\x\"",
                        "\"\x01\"",
                        "\"\\ud800\"",
                        "\"\\udc00\"",
                        "\"\xff\"",
                        "\"\xc0\xaf\"",
                        "\"\xed\xa0\x80\"",
                        "[1 2]",
                        "nul",
                        "{\"a\" 1}",
                        "1 2",
                        "+1",
                        ".5",
                        "{\"a\":1,}",
                        "\"\\u12g4\"",
                        "\"abc",
                        "1e400"}) {
    bool threw = false;
    try {
      (void)Json::parse(m);
    } catch (const Json::parse_error&) {
      threw = true;
    }
    Expect(threw, std::string("parse rejects [") + m + "]");
  }
  const auto replaced = [](const char* s) { return Json(s).dump(-1, ' ', false, Json::error_handler_t::replace); };
  const std::string r = "\xef\xbf\xbd";
  Expect(replaced("ok\xff"
                  "x") == "\"ok" + r + "x\"",
         "replace a bad byte");
  Expect(replaced("\xe2\x82") == "\"" + r + "\"", "replace a truncated sequence once");
  Expect(replaced("\xed\xa0\x80z") == "\"" + r + r + r + "z\"", "replace a surrogate's bytes");
  Expect(replaced("\xc0\xaf") == "\"" + r + r + "\"", "replace an overlong form");
  const Json j = {{"a", 1}, {"b", {1, 2}}, {"c", {{"d", nullptr}}}};
  Expect(j.dump() == "{\"a\":1,\"b\":[1,2],\"c\":{\"d\":null}}", "brace lists: " + j.dump());
  Expect(j.dump(2) == "{\n  \"a\": 1,\n  \"b\": [\n    1,\n    2\n  ],\n  \"c\": {\n    \"d\": null\n  }\n}",
         "indent 2");
  Expect(Json(1) == Json(1.0) && Json(1) != Json("1") && j["b"][1] == 2 && j.value("a", 0) == 1 &&
             j.value("zz", "d") == "d" && j["c"]["missing"].is_null() && !j.contains("zz"),
         "access and comparison");
  std::printf(bad ? "json_check: %d FAILED\n" : "json_check: ok\n", bad);
  return bad != 0;
}
