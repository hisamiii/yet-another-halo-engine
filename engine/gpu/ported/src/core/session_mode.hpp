#ifndef GUFO_CORE_SESSION_MODE_HPP_
#define GUFO_CORE_SESSION_MODE_HPP_

namespace gufo::core {

/// Per-session execution choice, independent of resident model capabilities.
enum class SessionMode { kAutoregressive, kSpeculative };

}  // namespace gufo::core

#endif  // GUFO_CORE_SESSION_MODE_HPP_
