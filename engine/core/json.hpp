// Json: the JSON value of the HFQ metadata and the server, in place of nlohmann::ordered_json (the subset of its
// interface core/ and serve/ use, with the same semantics). Objects keep their insertion order; a key set again keeps its place.
//   Json{{"k", 1}, {"a", {1, 2}}}   brace lists: an object when every element is a [string, value] pair, else an array
//   Json::parse(text)               RFC 8259; throws Json::parse_error (bad syntax, invalid UTF-8, nesting past 512)
//   dump(indent, ' ', false, Json::error_handler_t::replace)   invalid UTF-8 in strings -> U+FFFD (strict: throws)
// Numbers are integers when written without fraction or exponent and in the int64 or uint64 range, else double;
// they compare by value. Doubles print as nlohmann does (shortest round trip, fixed between 1e-4 and 1e15).
#ifndef YAH_CORE_JSON_HPP_
#define YAH_CORE_JSON_HPP_

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace yah::core {

class Json {
 public:
  using Member = std::pair<std::string, Json>;
  using Object = std::vector<Member>;
  using Array = std::vector<Json>;
  enum class Kind : std::uint8_t { kNull, kBool, kInt, kDouble, kString, kArray, kObject };
  enum class error_handler_t { strict, replace };  // NOLINT(readability-identifier-naming): nlohmann's names
  struct parse_error : std::runtime_error {        // NOLINT(readability-identifier-naming)
    using std::runtime_error::runtime_error;
  };
  struct type_error : std::runtime_error {  // NOLINT(readability-identifier-naming)
    using std::runtime_error::runtime_error;
  };
  struct out_of_range : std::runtime_error {  // NOLINT(readability-identifier-naming)
    using std::runtime_error::runtime_error;
  };

  Json() = default;
  Json(std::nullptr_t) {}                      // NOLINT(google-explicit-constructor)
  Json(bool b) : kind_(Kind::kBool), b_(b) {}  // NOLINT(google-explicit-constructor)
  template <class T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> = 0>
  Json(T i)  // NOLINT(google-explicit-constructor)
      : kind_(Kind::kInt), u_(std::is_unsigned_v<T> && i > T(INT64_MAX)), i_(static_cast<std::int64_t>(i)) {}
  template <class T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
  Json(T d) : kind_(Kind::kDouble), d_(static_cast<double>(d)) {}  // NOLINT(google-explicit-constructor)
  Json(const char* s) : kind_(Kind::kString), s_(s) {}             // NOLINT(google-explicit-constructor)
  Json(std::string s) : kind_(Kind::kString), s_(std::move(s)) {}  // NOLINT(google-explicit-constructor)
  Json(std::string_view s) : kind_(Kind::kString), s_(s) {}        // NOLINT(google-explicit-constructor)
  template <class T>
  Json(const std::vector<T>& v) : kind_(Kind::kArray), a_(v.begin(), v.end()) {}  // NOLINT(google-explicit-constructor)
  Json(std::initializer_list<Json> list) {                                        // NOLINT(google-explicit-constructor)
    bool pairs = true;
    for (const Json& j : list) pairs = pairs && j.kind_ == Kind::kArray && j.a_.size() == 2 && j.a_[0].is_string();
    if (pairs) {
      kind_ = Kind::kObject;
      for (const Json& j : list) (*this)[j.a_[0].s_] = j.a_[1];
    } else {
      kind_ = Kind::kArray;
      a_.assign(list.begin(), list.end());
    }
  }
  static Json array(std::initializer_list<Json> list = {}) {
    Json j;
    j.kind_ = Kind::kArray;
    j.a_.assign(list.begin(), list.end());
    return j;
  }
  static Json object(std::initializer_list<Json> list = {}) {
    Json j(list);
    if (list.size() == 0) j = Json(), j.kind_ = Kind::kObject;
    if (!j.is_object()) throw type_error("json: object() of something other than [key, value] pairs");
    return j;
  }

  [[nodiscard]] Kind kind() const { return kind_; }
  [[nodiscard]] bool is_null() const { return kind_ == Kind::kNull; }
  [[nodiscard]] bool is_boolean() const { return kind_ == Kind::kBool; }
  [[nodiscard]] bool is_number() const { return kind_ == Kind::kInt || kind_ == Kind::kDouble; }
  [[nodiscard]] bool is_number_integer() const { return kind_ == Kind::kInt; }
  [[nodiscard]] bool is_string() const { return kind_ == Kind::kString; }
  [[nodiscard]] bool is_array() const { return kind_ == Kind::kArray; }
  [[nodiscard]] bool is_object() const { return kind_ == Kind::kObject; }

  template <class T>
  [[nodiscard]] T get() const {
    if constexpr (std::is_same_v<T, std::string>) {
      return get_ref<const std::string&>();
    } else if constexpr (std::is_same_v<T, bool>) {
      if (!is_boolean()) throw type_error(std::string("json: a boolean expected, not ") + Name());
      return b_;
    } else if constexpr (std::is_arithmetic_v<T>) {
      if (kind_ == Kind::kInt) return u_ ? static_cast<T>(static_cast<std::uint64_t>(i_)) : static_cast<T>(i_);
      if (kind_ == Kind::kDouble) return static_cast<T>(d_);
      throw type_error(std::string("json: a number expected, not ") + Name());
    } else {
      static_assert(std::is_same_v<T, Json>, "json: get<T> of an unsupported type");
      return *this;
    }
  }
  template <class T, std::enable_if_t<std::is_same_v<T, const std::string&>, int> = 0>
  [[nodiscard]] const std::string& get_ref() const {
    if (!is_string()) throw type_error(std::string("json: a string expected, not ") + Name());
    return s_;
  }

  // Objects (a null becomes an object on a non-const access, as in nlohmann)
  Json& operator[](const std::string& key) { return Slot(key); }
  Json& operator[](const char* key) { return Slot(key); }
  const Json& operator[](const std::string& key) const { return Get(key); }
  const Json& operator[](const char* key) const { return Get(key); }
  [[nodiscard]] const Json& at(const std::string& key) const {
    const Json* v = Find(key);
    if (!v) throw out_of_range("json: no key '" + key + "'");
    return *v;
  }
  Json& at(const std::string& key) { return const_cast<Json&>(std::as_const(*this).at(key)); }
  [[nodiscard]] bool contains(std::string_view key) const { return Find(key) != nullptr; }
  std::size_t erase(std::string_view key) {
    if (!is_object()) return 0;
    for (auto it = o_.begin(); it != o_.end(); ++it)
      if (it->first == key) return o_.erase(it), 1;
    return 0;
  }
  // The member's value, or the default (a const char* default reads a string)
  [[nodiscard]] Json value(std::string_view key, const Json& fallback) const {
    const Json* v = Find(key);
    return v ? *v : fallback;
  }
  [[nodiscard]] std::string value(std::string_view key, const char* fallback) const {
    const Json* v = Find(key);
    return v ? v->get<std::string>() : std::string(fallback);
  }
  [[nodiscard]] std::string value(std::string_view key, const std::string& fallback) const {
    return value(key, fallback.c_str());
  }
  template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
  [[nodiscard]] T value(std::string_view key, T fallback) const {
    const Json* v = Find(key);
    return v ? v->get<T>() : fallback;
  }
  // The members, for (const auto& [key, value] : j.items()); empty unless an object.
  Object& items() { return is_object() ? o_ : Empty(); }
  [[nodiscard]] const Object& items() const { return is_object() ? o_ : Empty(); }

  // Arrays (a null becomes an array on push_back and non-const indexing; indexing past the end extends with nulls)
  template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
  Json& operator[](I i) {
    if (is_null()) kind_ = Kind::kArray;
    if (!is_array()) throw type_error(std::string("json: indexing ") + Name());
    if (static_cast<std::size_t>(i) >= a_.size()) a_.resize(static_cast<std::size_t>(i) + 1);
    return a_[static_cast<std::size_t>(i)];
  }
  template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
  const Json& operator[](I i) const {
    return at(static_cast<std::size_t>(i));
  }
  [[nodiscard]] const Json& at(std::size_t i) const {
    if (!is_array()) throw type_error(std::string("json: indexing ") + Name());
    if (i >= a_.size()) throw out_of_range("json: index " + std::to_string(i) + " past the end");
    return a_[i];
  }
  Json& at(std::size_t i) { return const_cast<Json&>(std::as_const(*this).at(i)); }
  void push_back(Json v) {
    if (is_null()) kind_ = Kind::kArray;
    if (!is_array()) throw type_error(std::string("json: push_back on ") + Name());
    a_.push_back(std::move(v));
  }
  [[nodiscard]] const Json& front() const { return at(0); }
  [[nodiscard]] const Json& back() const {
    if (!is_array() || a_.empty()) throw out_of_range("json: back() of an empty or non-array value");
    return a_.back();
  }

  // Containers: the element count; null 0, other scalars 1
  [[nodiscard]] std::size_t size() const {
    return is_array() ? a_.size() : is_object() ? o_.size() : is_null() ? 0 : 1;
  }
  [[nodiscard]] bool empty() const { return size() == 0; }

  // Iteration: an array's elements or an object's values (nothing for scalars)
  template <bool kConst>
  class Iter {
    using J = std::conditional_t<kConst, const Json, Json>;
    using M = std::conditional_t<kConst, const Member, Member>;

   public:
    Iter(J* a, M* o) : a_(a), o_(o) {}
    J& operator*() const { return a_ ? *a_ : o_->second; }
    J* operator->() const { return &**this; }
    Iter& operator++() {
      if (a_)
        ++a_;
      else
        ++o_;
      return *this;
    }
    bool operator==(const Iter& b) const { return a_ == b.a_ && o_ == b.o_; }
    bool operator!=(const Iter& b) const { return !(*this == b); }
    [[nodiscard]] const std::string& key() const { return o_->first; }

   private:
    J* a_;
    M* o_;
  };
  using iterator = Iter<false>;
  using const_iterator = Iter<true>;
  iterator begin() { return is_array() ? iterator(a_.data(), nullptr) : iterator(nullptr, Members()); }
  iterator end() {
    return is_array() ? iterator(a_.data() + a_.size(), nullptr) : iterator(nullptr, Members() + MemberCount());
  }
  [[nodiscard]] const_iterator begin() const {
    return is_array() ? const_iterator(a_.data(), nullptr) : const_iterator(nullptr, Members());
  }
  [[nodiscard]] const_iterator end() const {
    return is_array() ? const_iterator(a_.data() + a_.size(), nullptr)
                      : const_iterator(nullptr, Members() + MemberCount());
  }
  [[nodiscard]] const_iterator find(std::string_view key) const {
    if (is_object())
      for (const Member& m : o_)
        if (m.first == key) return const_iterator(nullptr, &m);
    return end();
  }

  friend bool operator==(const Json& a, const Json& b) {
    if (a.is_number() && b.is_number()) {
      if (a.kind_ == Kind::kInt && b.kind_ == Kind::kInt) return a.i_ == b.i_ && a.u_ == b.u_;
      return a.get<double>() == b.get<double>();
    }
    if (a.kind_ != b.kind_) return false;
    switch (a.kind_) {
      case Kind::kNull:
        return true;
      case Kind::kBool:
        return a.b_ == b.b_;
      case Kind::kString:
        return a.s_ == b.s_;
      case Kind::kArray:
        return a.a_ == b.a_;
      case Kind::kObject:
        return a.o_ == b.o_;
      default:
        return false;
    }
  }
  friend bool operator!=(const Json& a, const Json& b) { return !(a == b); }

  // Text: compact for indent < 0, else one member or element per line indented by indent * c.
  [[nodiscard]] std::string dump(int indent = -1, char c = ' ', bool ensure_ascii = false,
                                 error_handler_t on_bad_utf8 = error_handler_t::strict) const {
    std::string out;
    Write(out, indent, c, ensure_ascii, on_bad_utf8, 0);
    return out;
  }

  static Json parse(std::string_view text) {
    Parser p{text};
    p.Space();
    Json v = p.Value(0);
    p.Space();
    if (p.at != text.size()) p.Fail("unexpected text after the value");
    return v;
  }

 private:
  [[nodiscard]] const char* Name() const {
    static constexpr const char* kNames[] = {"null", "boolean", "number", "number", "string", "array", "object"};
    return kNames[static_cast<int>(kind_)];
  }
  static Object& Empty() {
    static Object empty;
    empty.clear();
    return empty;
  }
  Member* Members() { return is_object() ? o_.data() : nullptr; }
  [[nodiscard]] const Member* Members() const { return is_object() ? o_.data() : nullptr; }
  [[nodiscard]] std::size_t MemberCount() const { return is_object() ? o_.size() : 0; }
  [[nodiscard]] const Json* Find(std::string_view key) const {
    if (is_object())
      for (const Member& m : o_)
        if (m.first == key) return &m.second;
    return nullptr;
  }
  Json& Slot(std::string_view key) {
    if (is_null()) kind_ = Kind::kObject;
    if (!is_object()) throw type_error(std::string("json: a key of ") + Name());
    if (const Json* v = Find(key)) return const_cast<Json&>(*v);
    o_.emplace_back(std::string(key), Json());
    return o_.back().second;
  }
  [[nodiscard]] const Json& Get(std::string_view key) const {
    static const Json null;
    const Json* v = Find(key);
    return v ? *v : null;
  }

  // The length of the valid UTF-8 sequence at s[i] (0: invalid); *bad: then the bytes one U+FFFD stands for (the lead
  // byte and the continuation bytes that still fit it, as nlohmann and Unicode's "maximal subpart" replace)
  static std::size_t Utf8Length(std::string_view s, std::size_t i, std::size_t* bad = nullptr) {
    std::size_t sink = 0;
    if (!bad) bad = &sink;
    *bad = 1;
    const auto b = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
    const unsigned c = b(i);
    if (c < 0x80) return 1;
    const std::size_t n = c >= 0xC2 && c <= 0xDF ? 2 : c >= 0xE0 && c <= 0xEF ? 3 : c >= 0xF0 && c <= 0xF4 ? 4 : 0;
    if (!n) return 0;
    for (std::size_t k = 1; k < n; ++k) {
      if (i + k >= s.size()) return *bad = k, 0;
      const unsigned ck = b(i + k);
      // the second byte's range excludes overlongs, surrogates and code points past U+10FFFF
      const unsigned lo = k > 1 ? 0x80 : c == 0xE0 ? 0xA0 : c == 0xF0 ? 0x90 : 0x80;
      const unsigned hi = k > 1 ? 0xBF : c == 0xED ? 0x9F : c == 0xF4 ? 0x8F : 0xBF;
      if (ck < lo || ck > hi) return *bad = k, 0;
    }
    return n;
  }
  static void Hex4(std::string& out, unsigned u) {
    char buf[8];
    std::snprintf(buf, sizeof buf, "\\u%04x", u);
    out += buf;
  }
  static void WriteString(std::string& out, std::string_view s, bool ascii, error_handler_t on_bad) {
    out += '"';
    for (std::size_t i = 0; i < s.size();) {
      const auto c = static_cast<unsigned char>(s[i]);
      switch (c) {
        case '"':
          out += "\\\"";
          ++i;
          continue;
        case '\\':
          out += "\\\\";
          ++i;
          continue;
        case '\b':
          out += "\\b";
          ++i;
          continue;
        case '\f':
          out += "\\f";
          ++i;
          continue;
        case '\n':
          out += "\\n";
          ++i;
          continue;
        case '\r':
          out += "\\r";
          ++i;
          continue;
        case '\t':
          out += "\\t";
          ++i;
          continue;
        default:
          break;
      }
      if (c < 0x20) {
        Hex4(out, c), ++i;
        continue;
      }
      std::size_t skip = 1;
      const std::size_t n = Utf8Length(s, i, &skip);
      if (!n) {
        if (on_bad == error_handler_t::strict) throw type_error("json: invalid UTF-8 in a string");
        if (ascii)
          out += "\\ufffd";
        else
          out += "\xEF\xBF\xBD";
        i += skip;
        continue;
      }
      if (ascii && n > 1) {
        const auto b = [&](std::size_t k) { return static_cast<unsigned>(static_cast<unsigned char>(s[i + k])); };
        unsigned u = n == 2   ? (b(0) & 0x1F) << 6 | (b(1) & 0x3F)
                     : n == 3 ? (b(0) & 0x0F) << 12 | (b(1) & 0x3F) << 6 | (b(2) & 0x3F)
                              : (b(0) & 0x07) << 18 | (b(1) & 0x3F) << 12 | (b(2) & 0x3F) << 6 | (b(3) & 0x3F);
        if (u >= 0x10000)
          u -= 0x10000, Hex4(out, 0xD800 + (u >> 10)), Hex4(out, 0xDC00 + (u & 0x3FF));
        else
          Hex4(out, u);
      } else {
        out.append(s.substr(i, n));
      }
      i += n;
    }
    out += '"';
  }
  static void WriteDouble(std::string& out, double d) {
    if (!std::isfinite(d)) {
      out += "null";
      return;
    }
    if (d == 0) {
      out += std::signbit(d) ? "-0.0" : "0.0";
      return;
    }
    // the shortest round-trip digits d1.d2d3...e<x>, laid out as nlohmann's dtoa does (value = 0.d1d2... * 10^n)
    char buf[40];
    const auto r = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::scientific);
    const std::string_view t(buf, static_cast<std::size_t>(r.ptr - buf));
    const std::size_t epos = t.find('e');
    std::string digits;
    std::size_t at = 0;
    if (t[0] == '-') out += '-', at = 1;
    for (std::size_t i = at; i < epos; ++i)
      if (t[i] != '.') digits += t[i];
    int x = 0;
    std::from_chars(t.data() + epos + (t[epos + 1] == '+' ? 2 : 1), t.data() + t.size(), x);
    const int k = static_cast<int>(digits.size()), n = x + 1;
    if (k <= n && n <= 15) {
      out += digits, out.append(static_cast<std::size_t>(n - k), '0'), out += ".0";
    } else if (0 < n && n <= 15) {
      out += digits.substr(0, static_cast<std::size_t>(n)), out += '.',
          out += digits.substr(static_cast<std::size_t>(n));
    } else if (-4 < n && n <= 0) {
      out += "0.", out.append(static_cast<std::size_t>(-n), '0'), out += digits;
    } else {
      out += digits[0];
      if (k > 1) out += '.', out += digits.substr(1);
      char e[16];
      std::snprintf(e, sizeof e, "e%c%02d", n - 1 < 0 ? '-' : '+', std::abs(n - 1));
      out += e;
    }
  }
  void Write(std::string& out, int indent, char c, bool ascii, error_handler_t on_bad, int depth) const {
    const auto line = [&](int d) {
      if (indent < 0) return;
      out += '\n';
      out.append(static_cast<std::size_t>(indent) * static_cast<std::size_t>(d), c);
    };
    switch (kind_) {
      case Kind::kNull:
        out += "null";
        break;
      case Kind::kBool:
        out += b_ ? "true" : "false";
        break;
      case Kind::kInt:
        out += u_ ? std::to_string(static_cast<std::uint64_t>(i_)) : std::to_string(i_);
        break;
      case Kind::kDouble:
        WriteDouble(out, d_);
        break;
      case Kind::kString:
        WriteString(out, s_, ascii, on_bad);
        break;
      case Kind::kArray:
        out += '[';
        for (std::size_t k = 0; k < a_.size(); ++k) {
          if (k) out += ',';
          line(depth + 1);
          a_[k].Write(out, indent, c, ascii, on_bad, depth + 1);
        }
        if (!a_.empty()) line(depth);
        out += ']';
        break;
      case Kind::kObject:
        out += '{';
        for (std::size_t k = 0; k < o_.size(); ++k) {
          if (k) out += ',';
          line(depth + 1);
          WriteString(out, o_[k].first, ascii, on_bad);
          out += indent < 0 ? ":" : ": ";
          o_[k].second.Write(out, indent, c, ascii, on_bad, depth + 1);
        }
        if (!o_.empty()) line(depth);
        out += '}';
        break;
    }
  }

  // A key given twice keeps its first place and its last value (a hash pass: objects may hold a whole vocabulary).
  static void Dedupe(Object& o) {
    if (o.size() < 2) return;
    std::unordered_map<std::string_view, std::size_t> first;
    first.reserve(o.size());
    std::vector<bool> drop(o.size());
    bool any = false;
    for (std::size_t i = 0; i < o.size(); ++i) {
      const auto [it, fresh] = first.emplace(o[i].first, i);
      if (!fresh) o[it->second].second = std::move(o[i].second), drop[i] = any = true;
    }
    if (!any) return;
    std::size_t w = 0;
    for (std::size_t i = 0; i < o.size(); ++i)
      if (!drop[i]) {
        if (w != i) o[w] = std::move(o[i]);
        ++w;
      }
    o.resize(w);
  }

  struct Parser {
    std::string_view s;
    std::size_t at = 0;
    static constexpr int kMaxDepth = 512;

    [[noreturn]] void Fail(const char* what) const {
      throw parse_error("parse error at byte " + std::to_string(at) + ": " + what);
    }
    void Space() {
      while (at < s.size() && (s[at] == ' ' || s[at] == '\t' || s[at] == '\n' || s[at] == '\r')) ++at;
    }
    bool Eat(char c) {
      if (at < s.size() && s[at] == c) return ++at, true;
      return false;
    }
    void Word(std::string_view w) {
      if (s.substr(at, w.size()) != w) Fail("invalid literal");
      at += w.size();
    }
    Json Value(int depth) {
      if (depth > kMaxDepth) Fail("nesting too deep");
      if (at >= s.size()) Fail("unexpected end of input");
      switch (s[at]) {
        case '{':
          return ParseObject(depth);
        case '[':
          return ParseArray(depth);
        case '"':
          return Json(String());
        case 't':
          Word("true");
          return Json(true);
        case 'f':
          Word("false");
          return Json(false);
        case 'n':
          Word("null");
          return Json();
        default:
          return Number();
      }
    }
    Json ParseObject(int depth) {
      ++at;
      Json o = Json::object();
      Space();
      if (Eat('}')) return o;
      for (;;) {
        Space();
        if (at >= s.size() || s[at] != '"') Fail("a string key expected");
        std::string key = String();
        Space();
        if (!Eat(':')) Fail("':' expected");
        Space();
        Json v = Value(depth + 1);
        o.o_.emplace_back(std::move(key), std::move(v));
        Space();
        if (Eat('}')) return Dedupe(o.o_), o;
        if (!Eat(',')) Fail("',' or '}' expected");
      }
    }
    Json ParseArray(int depth) {
      ++at;
      Json a = Json::array();
      Space();
      if (Eat(']')) return a;
      for (;;) {
        Space();
        a.a_.push_back(Value(depth + 1));
        Space();
        if (Eat(']')) return a;
        if (!Eat(',')) Fail("',' or ']' expected");
      }
    }
    unsigned Hex() {
      if (at + 4 > s.size()) Fail("a 4-digit \\u escape expected");
      unsigned u = 0;
      for (int k = 0; k < 4; ++k) {
        const char c = s[at++];
        const int d = c >= '0' && c <= '9'   ? c - '0'
                      : c >= 'a' && c <= 'f' ? c - 'a' + 10
                      : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                             : -1;
        if (d < 0) Fail("a 4-digit \\u escape expected");
        u = u * 16 + static_cast<unsigned>(d);
      }
      return u;
    }
    static void Utf8(std::string& out, unsigned u) {
      if (u < 0x80) {
        out += static_cast<char>(u);
      } else if (u < 0x800) {
        out += static_cast<char>(0xC0 | u >> 6), out += static_cast<char>(0x80 | (u & 0x3F));
      } else if (u < 0x10000) {
        out += static_cast<char>(0xE0 | u >> 12), out += static_cast<char>(0x80 | (u >> 6 & 0x3F));
        out += static_cast<char>(0x80 | (u & 0x3F));
      } else {
        out += static_cast<char>(0xF0 | u >> 18), out += static_cast<char>(0x80 | (u >> 12 & 0x3F));
        out += static_cast<char>(0x80 | (u >> 6 & 0x3F)), out += static_cast<char>(0x80 | (u & 0x3F));
      }
    }
    std::string String() {
      ++at;
      std::string out;
      for (;;) {
        if (at >= s.size()) Fail("unterminated string");
        const auto c = static_cast<unsigned char>(s[at]);
        if (c == '"') return ++at, out;
        if (c < 0x20) Fail("a control character in a string");
        if (c == '\\') {
          if (++at >= s.size()) Fail("unterminated string");
          const char e = s[at++];
          switch (e) {
            case '"':
              out += '"';
              break;
            case '\\':
              out += '\\';
              break;
            case '/':
              out += '/';
              break;
            case 'b':
              out += '\b';
              break;
            case 'f':
              out += '\f';
              break;
            case 'n':
              out += '\n';
              break;
            case 'r':
              out += '\r';
              break;
            case 't':
              out += '\t';
              break;
            case 'u': {
              unsigned u = Hex();
              if (u >= 0xD800 && u <= 0xDBFF) {
                if (s.substr(at, 2) != "\\u") Fail("a lone high surrogate");
                at += 2;
                const unsigned lo = Hex();
                if (lo < 0xDC00 || lo > 0xDFFF) Fail("a high surrogate without its low one");
                u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
              } else if (u >= 0xDC00 && u <= 0xDFFF) {
                Fail("a lone low surrogate");
              }
              Utf8(out, u);
              break;
            }
            default:
              Fail("an invalid escape");
          }
          continue;
        }
        const std::size_t n = Utf8Length(s, at);
        if (!n) Fail("invalid UTF-8");
        out.append(s.substr(at, n));
        at += n;
      }
    }
    Json Number() {
      const std::size_t from = at;
      const auto digits = [&] {
        const std::size_t d = at;
        while (at < s.size() && s[at] >= '0' && s[at] <= '9') ++at;
        return at - d;
      };
      Eat('-');
      if (at < s.size() && s[at] == '0')
        ++at;
      else if (!digits())
        Fail("invalid number");
      bool integer = true;
      if (Eat('.')) {
        integer = false;
        if (!digits()) Fail("invalid number");
      }
      if (at < s.size() && (s[at] == 'e' || s[at] == 'E')) {
        integer = false, ++at;
        if (!Eat('+')) Eat('-');
        if (!digits()) Fail("invalid number");
      }
      const char* b = s.data() + from;
      const char* e = s.data() + at;
      if (integer) {
        std::int64_t i = 0;
        if (std::from_chars(b, e, i).ec == std::errc()) return Json(i);
        std::uint64_t u = 0;
        if (*b != '-' && std::from_chars(b, e, u).ec == std::errc()) return Json(u);
      }
      double d = 0;
      if (std::from_chars(b, e, d).ec != std::errc() || !std::isfinite(d)) Fail("number out of range");
      return Json(d);
    }
  };

  Kind kind_ = Kind::kNull;
  bool b_ = false;
  bool u_ = false;  // an integer past INT64_MAX, held as its uint64 bits
  std::int64_t i_ = 0;
  double d_ = 0;
  std::string s_;
  Array a_;
  Object o_;
};

}  // namespace yah::core

#endif  // YAH_CORE_JSON_HPP_
