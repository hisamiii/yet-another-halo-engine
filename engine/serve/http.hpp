// A small HTTP/1.1 server and client: what yah_server, yah_chat and yah_serve_test need, in place of cpp-httplib.
// Server: one thread per connection, keep-alive, Content-Length or chunked request bodies, "Expect: 100-continue",
// responses with Content-Length or streamed (chunked; server-sent events), CORS for browser clients (any origin, the
// OPTIONS preflight answered). Client: one connection per request, a streamed body handed to a receiver as it arrives.
// Plain TCP only (no TLS): the server is meant for localhost or behind a proxy.
#ifndef YAH_SERVE_HTTP_HPP_
#define YAH_SERVE_HTTP_HPP_

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace yah::serve::http {

using Headers = std::vector<std::pair<std::string, std::string>>;

// The value of header name (any case), "" when absent.
[[nodiscard]] std::string HeaderValue(const Headers& headers, std::string_view name);

struct Request {
  std::string method, path, query, body;  // path without the query; query without the '?'
  Headers headers;                        // names in lower case
  [[nodiscard]] std::string Header(std::string_view name) const { return HeaderValue(headers, name); }
};

// A streamed response body, sent as chunks.
class Sink {
 public:
  virtual ~Sink() = default;
  // false once the client is gone
  virtual bool Write(std::string_view data) = 0;
  // whether the client is still connected (a peer that closed its end reads as gone without a write)
  virtual bool Alive() = 0;
};

struct Response {
  int status = 200;
  std::string content_type = "text/plain", body;
  Headers headers;
  // When set, the body is this function's writes, streamed (status, content_type and headers still apply).
  std::function<void(Sink&)> stream;
  void Set(std::string b, std::string type) { body = std::move(b), content_type = std::move(type); }
};

using Handler = std::function<void(const Request&, Response&)>;

class Server {
 public:
  void Get(std::string path, Handler h) { routes_.push_back({"GET", std::move(path), std::move(h)}); }
  void Post(std::string path, Handler h) { routes_.push_back({"POST", std::move(path), std::move(h)}); }
  // Fills the body of an error status left without one (404: no route for the method and path).
  std::function<void(Response&)> on_error;
  // The response to a handler's exception (message: what()).
  std::function<void(Response&, const std::string& message)> on_exception;
  // Serves until Stop(); false when the address cannot be bound.
  bool Listen(const std::string& host, int port);
  void Stop();

 private:
  struct Route {
    std::string method, path;
    Handler handler;
  };
  void Serve(int fd);
  void Respond(int fd, const Request& req, bool keep_alive, bool* ok);
  std::vector<Route> routes_;
  std::atomic<int> listen_fd_{-1};
  std::atomic<int> connections_{0};
};

struct Result {
  enum class Error { kNone, kConnect, kWrite, kRead, kCanceled };
  Error error = Error::kNone;
  int status = 0;
  Headers headers;
  std::string body;  // empty when a receiver took it
  explicit operator bool() const { return error == Error::kNone; }
  [[nodiscard]] std::string Header(std::string_view name) const { return HeaderValue(headers, name); }
};
[[nodiscard]] const char* ToString(Result::Error e);

class Client {
 public:
  Client(std::string host, int port) : host_(std::move(host)), port_(port) {}
  explicit Client(const std::string& url);  // http://host[:port] (port 80 by default)
  void set_read_timeout(int seconds) { read_timeout_ = seconds; }
  Result Get(const std::string& path, const Headers& headers = {});
  // receiver: the body as it arrives (then not kept in Result::body); return false to cancel (Error::kCanceled).
  Result Post(const std::string& path, const std::string& body, const std::string& content_type,
              const Headers& headers = {}, const std::function<bool(const char*, std::size_t)>& receiver = {});

 private:
  Result Send(const std::string& method, const std::string& path, const std::string& body,
              const std::string& content_type, const Headers& headers,
              const std::function<bool(const char*, std::size_t)>& receiver);
  std::string host_;
  int port_ = 80;
  int read_timeout_ = 300;
};

}  // namespace yah::serve::http

#endif  // YAH_SERVE_HTTP_HPP_
