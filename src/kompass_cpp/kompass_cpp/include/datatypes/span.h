#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Kompass {

/**
 * Non-owning read-only view over contiguous memory (a C++17 stand-in for
 * std::span<const T>). Constructible from std:vector. Python bindings can
 * pass raw buffer pointers for zero-copy input.
 */
template <typename T> class Span {
public:
  constexpr Span() = default;
  constexpr Span(const T *data, size_t size) : data_(data), size_(size) {}
  Span(const std::vector<T> &v) : data_(v.data()), size_(v.size()) {}

  constexpr const T *data() const { return data_; }
  constexpr size_t size() const { return size_; }
  constexpr bool empty() const { return size_ == 0; }
  constexpr const T &operator[](size_t i) const { return data_[i]; }
  // Iterators so a Span works in range-for and templated consumers
  constexpr const T *begin() const { return data_; }
  constexpr const T *end() const { return data_ + size_; }

private:
  const T *data_ = nullptr;
  size_t size_ = 0;
};

using ByteSpan = Span<uint8_t>;

} // namespace Kompass
