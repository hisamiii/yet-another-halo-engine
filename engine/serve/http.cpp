#include "serve/http.hpp"

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <thread>

namespace yah::serve::http {
namespace {

constexpr std::size_t kMaxHead = 64 * 1024;          // request line and headers
constexpr std::size_t kMaxBody = 256 * 1024 * 1024;  // a request body
constexpr int kMaxConnections = 256;
constexpr int kIdleSeconds = 60;  // a keep-alive connection waiting for its next request

std::string Lower(std::string_view s) {
  std::string out(s);
  for (char& c : out) c = static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
  return out;
}
std::string_view Trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
  return s;
}
bool HasToken(std::string_view list, std::string_view token) {  // a comma-separated header value, any case
  const std::string l = Lower(list), t = Lower(token);
  for (std::size_t at = 0; at <= l.size();) {
    const std::size_t end = std::min(l.find(',', at), l.size());
    if (Trim(std::string_view(l).substr(at, end - at)) == t) return true;
    at = end + 1;
  }
  return false;
}

bool SendAll(int fd, std::string_view data) {
  while (!data.empty()) {
    const ssize_t n = ::send(fd, data.data(), data.size(), MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    data.remove_prefix(static_cast<std::size_t>(n));
  }
  return true;
}

// Buffered reads from a socket.
class Reader {
 public:
  explicit Reader(int fd) : fd_(fd) {}
  // More bytes into the buffer; false at end of stream, an error or a timeout.
  bool Fill() {
    char tmp[16384];
    for (;;) {
      const ssize_t n = ::recv(fd_, tmp, sizeof tmp, 0);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) return false;
      buf_.append(tmp, static_cast<std::size_t>(n));
      return true;
    }
  }
  // Up to and without the next CRLF (a bare LF also ends a line); false past max bytes or at end of stream.
  bool Line(std::string* line, std::size_t max) {
    for (;;) {
      const std::size_t nl = buf_.find('\n', at_);
      if (nl != std::string::npos) {
        std::size_t end = nl;
        if (end > at_ && buf_[end - 1] == '\r') --end;
        line->assign(buf_, at_, end - at_);
        at_ = nl + 1;
        return true;
      }
      if (buf_.size() - at_ > max || !Fill()) return false;
    }
  }
  // Exactly n bytes, handed to out in pieces as they arrive; out returns false to stop.
  bool Bytes(std::size_t n, const std::function<bool(const char*, std::size_t)>& out) {
    while (n) {
      if (at_ == buf_.size()) {
        buf_.clear(), at_ = 0;
        if (!Fill()) return false;
      }
      const std::size_t k = std::min(n, buf_.size() - at_);
      if (!out(buf_.data() + at_, k)) return false;
      at_ += k, n -= k;
    }
    Compact();
    return true;
  }
  // Everything until the peer closes.
  bool Rest(const std::function<bool(const char*, std::size_t)>& out) {
    for (;;) {
      if (at_ < buf_.size() && !out(buf_.data() + at_, buf_.size() - at_)) return false;
      buf_.clear(), at_ = 0;
      if (!Fill()) return true;
    }
  }
  [[nodiscard]] bool Buffered() const { return at_ < buf_.size(); }

 private:
  void Compact() {
    if (at_ > 65536) buf_.erase(0, at_), at_ = 0;
  }
  int fd_;
  std::string buf_;
  std::size_t at_ = 0;
};

bool ParseSize(std::string_view s, std::size_t* v, int base = 10) {
  s = Trim(s);
  if (s.empty()) return false;
  const auto r = std::from_chars(s.data(), s.data() + s.size(), *v, base);
  return r.ec == std::errc() && (r.ptr == s.data() + s.size() || (base == 16 && *r.ptr == ';'));
}

// Headers until the empty line; names lower-cased. false: malformed, too long or the stream ended.
bool ReadHeaders(Reader& in, Headers* headers, std::size_t* budget) {
  std::string line;
  for (;;) {
    if (!in.Line(&line, *budget)) return false;
    if (line.size() + 2 > *budget) return false;
    *budget -= line.size() + 2;
    if (line.empty()) return true;
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos || colon == 0) return false;
    headers->emplace_back(Lower(Trim(std::string_view(line).substr(0, colon))),
                          std::string(Trim(std::string_view(line).substr(colon + 1))));
  }
}

// A chunked body (RFC 9112 7.1), trailers skipped.
bool ReadChunked(Reader& in, std::size_t max, const std::function<bool(const char*, std::size_t)>& out) {
  std::string line;
  std::size_t total = 0;
  for (;;) {
    std::size_t n = 0;
    if (!in.Line(&line, 1024) || !ParseSize(line, &n, 16)) return false;
    if (n == 0) break;
    if ((total += n) > max) return false;
    if (!in.Bytes(n, out) || !in.Line(&line, 2) || !line.empty()) return false;
  }
  Headers trailers;
  std::size_t budget = kMaxHead;
  return ReadHeaders(in, &trailers, &budget);
}

const char* Reason(int status) {
  switch (status) {
    case 100:
      return "Continue";
    case 200:
      return "OK";
    case 204:
      return "No Content";
    case 400:
      return "Bad Request";
    case 401:
      return "Unauthorized";
    case 403:
      return "Forbidden";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 408:
      return "Request Timeout";
    case 413:
      return "Content Too Large";
    case 422:
      return "Unprocessable Content";
    case 429:
      return "Too Many Requests";
    case 431:
      return "Request Header Fields Too Large";
    case 500:
      return "Internal Server Error";
    case 501:
      return "Not Implemented";
    case 503:
      return "Service Unavailable";
    default:
      return status < 400 ? "OK" : "Error";
  }
}

class ChunkSink : public Sink {
 public:
  explicit ChunkSink(int fd) : fd_(fd) {}
  bool Write(std::string_view data) override {
    if (!ok_) return false;
    if (data.empty()) return true;
    char head[24];
    const int n = std::snprintf(head, sizeof head, "%zx\r\n", data.size());
    std::string chunk(head, static_cast<std::size_t>(n));
    chunk.append(data).append("\r\n");
    return ok_ = SendAll(fd_, chunk);
  }
  bool Alive() override {
    if (!ok_) return false;
    pollfd p{fd_, POLLIN, 0};
    if (::poll(&p, 1, 0) > 0) {
      if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) return ok_ = false;
      char c;
      // readable: data (a pipelined request, still connected) or the peer's close (0 bytes)
      if ((p.revents & POLLIN) && ::recv(fd_, &c, 1, MSG_PEEK | MSG_DONTWAIT) == 0) return ok_ = false;
    }
    return true;
  }
  [[nodiscard]] bool ok() const { return ok_; }

 private:
  int fd_;
  bool ok_ = true;
};

void SetTimeout(int fd, int seconds) {
  timeval tv{seconds, 0};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
}

}  // namespace

std::string HeaderValue(const Headers& headers, std::string_view name) {
  const std::string want = Lower(name);
  for (const auto& [k, v] : headers)
    if (Lower(k) == want) return v;
  return {};
}

bool Server::Listen(const std::string& host, int port) {
  addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC, hints.ai_socktype = SOCK_STREAM, hints.ai_flags = AI_PASSIVE;
  if (::getaddrinfo(host.empty() ? nullptr : host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0)
    return false;
  int fd = -1;
  for (addrinfo* a = res; a && fd < 0; a = a->ai_next) {
    fd = ::socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, a->ai_protocol);
    if (fd < 0) continue;
    const int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (::bind(fd, a->ai_addr, a->ai_addrlen) != 0 || ::listen(fd, 128) != 0) ::close(fd), fd = -1;
  }
  ::freeaddrinfo(res);
  if (fd < 0) return false;
  listen_fd_ = fd;
  for (;;) {
    const int c = ::accept4(fd, nullptr, nullptr, SOCK_CLOEXEC);
    if (c < 0) {
      if (errno == EINTR || errno == ECONNABORTED || errno == EMFILE || errno == ENFILE) continue;
      break;  // Stop() shut the socket down
    }
    if (connections_ >= kMaxConnections) {
      ::close(c);
      continue;
    }
    ++connections_;
    std::thread([this, c] {
      Serve(c);
      ::close(c);
      --connections_;
    }).detach();
  }
  ::close(fd);
  return true;
}

void Server::Stop() {
  const int fd = listen_fd_.exchange(-1);
  if (fd >= 0) ::shutdown(fd, SHUT_RDWR);
}

void Server::Serve(int fd) {
  const int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);  // events go out as written
  Reader in(fd);
  const auto fail = [&](int status) {
    const std::string r = "HTTP/1.1 " + std::to_string(status) + " " + Reason(status) +
                          "\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    SendAll(fd, r);
  };
  for (;;) {
    SetTimeout(fd, kIdleSeconds);
    std::string line;
    do {  // (empty lines before a request are allowed)
      if (!in.Line(&line, kMaxHead)) return;
    } while (line.empty());
    SetTimeout(fd, 30);
    Request req;
    const std::size_t s1 = line.find(' '), s2 = line.rfind(' ');
    if (s1 == std::string::npos || s2 == s1) return fail(400);
    req.method = line.substr(0, s1);
    std::string target = line.substr(s1 + 1, s2 - s1 - 1);
    const std::string version = line.substr(s2 + 1);
    if (version != "HTTP/1.1" && version != "HTTP/1.0") return fail(400);
    if (target.rfind("http://", 0) == 0) {  // absolute form
      const std::size_t slash = target.find('/', 7);
      target = slash == std::string::npos ? "/" : target.substr(slash);
    }
    const std::size_t q = target.find('?');
    req.path = target.substr(0, q);
    if (q != std::string::npos) req.query = target.substr(q + 1);
    std::size_t budget = kMaxHead;
    if (!ReadHeaders(in, &req.headers, &budget)) return fail(431);
    const std::string connection = req.Header("connection");
    const bool keep_alive = version == "HTTP/1.1" ? !HasToken(connection, "close") : HasToken(connection, "keep-alive");
    // the body
    const std::string te = req.Header("transfer-encoding"), cl = req.Header("content-length");
    const bool chunked = !te.empty();
    std::size_t length = 0;
    if (chunked && !HasToken(te, "chunked")) return fail(501);
    if (!chunked && !cl.empty() && !ParseSize(cl, &length)) return fail(400);
    if (length > kMaxBody) return fail(413);
    if ((chunked || length) && HasToken(req.Header("expect"), "100-continue") &&
        !SendAll(fd, "HTTP/1.1 100 Continue\r\n\r\n"))
      return;
    const auto append = [&](const char* p, std::size_t n) {
      req.body.append(p, n);
      return true;
    };
    if (chunked ? !ReadChunked(in, kMaxBody, append) : !in.Bytes(length, append)) return fail(400);
    bool ok = true;
    Respond(fd, req, keep_alive, &ok);
    if (!ok || !keep_alive) return;
  }
}

void Server::Respond(int fd, const Request& req, bool keep_alive, bool* ok) {
  Response res;
  if (req.method == "OPTIONS") {  // a CORS preflight
    res.status = 204;
    res.headers = {{"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
                   {"Access-Control-Allow-Headers", req.Header("access-control-request-headers").empty()
                                                        ? "Content-Type, Authorization"
                                                        : req.Header("access-control-request-headers")},
                   {"Access-Control-Max-Age", "86400"}};
    res.content_type.clear();
  } else {
    const Route* route = nullptr;
    for (const Route& r : routes_)
      if (r.path == req.path && (r.method == req.method || (r.method == "GET" && req.method == "HEAD"))) route = &r;
    if (!route) {
      res.status = 404;
    } else {
      try {
        route->handler(req, res);
      } catch (const std::exception& e) {
        res = Response{}, res.status = 500;
        if (on_exception) on_exception(res, e.what());
      } catch (...) {
        res = Response{}, res.status = 500;
        if (on_exception) on_exception(res, "internal error");
      }
    }
    if (res.status >= 400 && res.body.empty() && !res.stream && on_error) on_error(res);
  }
  std::string head = "HTTP/1.1 " + std::to_string(res.status) + " " + Reason(res.status) + "\r\n";
  if (!res.content_type.empty()) head += "Content-Type: " + res.content_type + "\r\n";
  head += "Access-Control-Allow-Origin: *\r\n";
  for (const auto& [k, v] : res.headers) head += k + ": " + v + "\r\n";
  head += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
  if (res.stream && req.method != "HEAD") {
    head += "Transfer-Encoding: chunked\r\n\r\n";
    if (!SendAll(fd, head)) return void(*ok = false);
    ChunkSink sink(fd);
    try {
      res.stream(sink);
    } catch (...) {
      return void(*ok = false);  // the stream cannot carry an error status any more: drop the connection
    }
    *ok = sink.ok() && SendAll(fd, "0\r\n\r\n");
    return;
  }
  if (res.status != 204) head += "Content-Length: " + std::to_string(res.body.size()) + "\r\n";
  head += "\r\n";
  if (req.method != "HEAD" && res.status != 204) head += res.body;
  *ok = SendAll(fd, head);
}

const char* ToString(Result::Error e) {
  switch (e) {
    case Result::Error::kNone:
      return "Success";
    case Result::Error::kConnect:
      return "Could not establish connection";
    case Result::Error::kWrite:
      return "Failed to write connection";
    case Result::Error::kRead:
      return "Failed to read connection";
    case Result::Error::kCanceled:
      return "Connection handling canceled";
  }
  return "Unknown";
}

Client::Client(const std::string& url) {
  std::string_view u = url;
  if (u.rfind("http://", 0) == 0) u.remove_prefix(7);
  u = u.substr(0, u.find('/'));
  const std::size_t colon = u.rfind(':');
  if (colon != std::string_view::npos && u.find(']', colon) == std::string_view::npos) {
    host_ = std::string(u.substr(0, colon));
    port_ = std::atoi(std::string(u.substr(colon + 1)).c_str());
  } else {
    host_ = std::string(u);
  }
  if (host_.size() > 2 && host_.front() == '[' && host_.back() == ']') host_ = host_.substr(1, host_.size() - 2);
}

Result Client::Get(const std::string& path, const Headers& headers) {
  return Send("GET", path, "", "", headers, {});
}

Result Client::Post(const std::string& path, const std::string& body, const std::string& content_type,
                    const Headers& headers, const std::function<bool(const char*, std::size_t)>& receiver) {
  return Send("POST", path, body, content_type, headers, receiver);
}

Result Client::Send(const std::string& method, const std::string& path, const std::string& body,
                    const std::string& content_type, const Headers& headers,
                    const std::function<bool(const char*, std::size_t)>& receiver) {
  Result r;
  addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC, hints.ai_socktype = SOCK_STREAM;
  if (::getaddrinfo(host_.c_str(), std::to_string(port_).c_str(), &hints, &res) != 0) {
    r.error = Result::Error::kConnect;
    return r;
  }
  int fd = -1;
  for (addrinfo* a = res; a && fd < 0; a = a->ai_next) {
    fd = ::socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, a->ai_protocol);
    if (fd >= 0 && ::connect(fd, a->ai_addr, a->ai_addrlen) != 0) ::close(fd), fd = -1;
  }
  ::freeaddrinfo(res);
  if (fd < 0) {
    r.error = Result::Error::kConnect;
    return r;
  }
  struct Closer {
    int fd;
    ~Closer() { ::close(fd); }
  } closer{fd};
  SetTimeout(fd, read_timeout_);
  std::string req =
      method + " " + path + " HTTP/1.1\r\nHost: " + host_ + ":" + std::to_string(port_) + "\r\nConnection: close\r\n";
  for (const auto& [k, v] : headers) req += k + ": " + v + "\r\n";
  if (method == "POST") {
    if (!content_type.empty()) req += "Content-Type: " + content_type + "\r\n";
    req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  }
  req += "\r\n" + body;
  if (!SendAll(fd, req)) {
    r.error = Result::Error::kWrite;
    return r;
  }
  Reader in(fd);
  std::string line;
  do {  // (interim 1xx responses skipped)
    r.headers.clear();
    std::size_t budget = kMaxHead;
    if (!in.Line(&line, kMaxHead) || line.size() < 12 || line.rfind("HTTP/1.", 0) != 0 ||
        !ReadHeaders(in, &r.headers, &budget)) {
      r.error = Result::Error::kRead;
      return r;
    }
    r.status = std::atoi(line.c_str() + 9);
  } while (r.status >= 100 && r.status < 200);
  bool canceled = false;
  const auto out = [&](const char* p, std::size_t n) {
    if (!receiver) return r.body.append(p, n), true;
    canceled = !receiver(p, n);
    return !canceled;
  };
  const std::string te = r.Header("transfer-encoding"), cl = r.Header("content-length");
  std::size_t length = 0;
  bool read_ok = false;
  if (method == "HEAD" || r.status == 204 || r.status == 304)
    read_ok = true;
  else if (HasToken(te, "chunked"))
    read_ok = ReadChunked(in, SIZE_MAX, out);
  else if (!cl.empty())
    read_ok = ParseSize(cl, &length) && in.Bytes(length, out);
  else
    read_ok = in.Rest(out);
  if (canceled)
    r.error = Result::Error::kCanceled;
  else if (!read_ok)
    r.error = Result::Error::kRead;
  return r;
}

}  // namespace yah::serve::http
