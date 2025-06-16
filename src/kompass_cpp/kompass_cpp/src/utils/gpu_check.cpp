#include <string>

#ifdef GPU
#include <sycl/sycl.hpp>
#endif

std::string getAvailableAccelerators() {
#ifdef GPU
  std::string result;
  auto platforms = sycl::platform::get_platforms();

  for (const auto &platform : platforms) {
    for (const auto &device : platform.get_devices()) {
      result += device.get_info<sycl::info::device::name>() + "\n";
    }
  }

  return result;
#else
  return "";
#endif
}
