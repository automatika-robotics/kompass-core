#pragma once

#include "utils/logger.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <sycl/sycl.hpp>

namespace Kompass {

/**
 * @brief Per-call kernel inputs go in pinned host memory wherever the device
 * supports host USM, so kernels read them in place without a queue copy.
 * Other devices with coarse-grained SVM, keep device memory.
 * KOMPASS_STREAM_INPUT_MEMORY=device forces device memory.
 */
inline bool streamInputsInHostMemory(const sycl::device &device) {
  const char *setting = std::getenv("KOMPASS_STREAM_INPUT_MEMORY");
  if (setting && std::string(setting) == "device") {
    return false;
  }
  return device.has(sycl::aspect::usm_host_allocations);
}

/**
 * @brief A kernel input written by the host at the start of every call and
 * read by the kernels of that call, such as raw point cloud bytes or laser
 * ranges.
 *
 * The buffer records where it was allocated, so uploads use a plain CPU copy
 * for host memory and a queue copy for device memory.
 *
 * NOTE: Host copies are not ordered with the queue, the host may only write the
 * buffer while no queued work reads it. The GPU classes should guarantee this, so
 * every public call ends with a wait, a class mutex serializes callers, and a
 * call that throws drains the queue before rethrowing.
 *
 * The struct does not own its memory. The owning class calls release() from
 * its destructor, after waiting for the queue.
 */
template <typename T> struct StreamInputBuffer {
  T *data = nullptr;
  std::size_t capacity = 0; // num elements
  bool inHostMemory = false;
  // allocate pinned host memory when the device supports it, otherwise device memory
  bool preferHost = false;  // set once by the owner

  /**
   * @brief Makes room for num elements. Grow-only.
   *
   * A host allocation that returns null falls back to device memory.
   *
   * NOTE: Host memory should never be requested from a device that does not
   * report host USM support.
   */
  void reserve(std::size_t count, sycl::queue &queue) {
    if (data && capacity >= count) {
      return;
    }
    if (data) {
      // sycl::free is not ordered with the queue. Wait for any work that
      // could still read the old buffer
      queue.wait();
      release(queue);
    }
    const std::size_t elements = count > 0 ? count : 1;
    if (preferHost &&
        queue.get_device().has(sycl::aspect::usm_host_allocations)) {
      data = sycl::malloc_host<T>(elements, queue);
      inHostMemory = data != nullptr;
    }
    // fallback to device
    if (!data) {
      data = sycl::malloc_device<T>(elements, queue);
      inHostMemory = false;
    }
    // true failure
    if (!data) {
      LOG_ERROR("Could not allocate ", elements, " kernel input elements");
      throw std::bad_alloc();
    }
    capacity = elements;
  }

  /**
   * @brief Copies num elements from host memory into the buffer. A CPU copy
   * for host memory, a queue copy for device memory.
   */
  void upload(const T *source, std::size_t count, sycl::queue &queue) {
    if (inHostMemory) {
      std::memcpy(data, source, sizeof(T) * count);
    } else {
      queue.memcpy(data, source, sizeof(T) * count);
    }
  }

  /**
   * @brief Frees the buffer. The caller must make sure no queued work still
   * reads it.
   */
  void release(sycl::queue &queue) {
    if (data) {
      sycl::free(data, queue);
    }
    data = nullptr;
    capacity = 0;
    inHostMemory = false;
  }
};

} // namespace Kompass
