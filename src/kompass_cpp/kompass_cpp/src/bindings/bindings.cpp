#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/logger.h"

namespace py = pybind11;

void bindings_types(py::module_ &);
void bindings_config(py::module_ &);
void bindings_control(py::module_ &);
void bindings_mapping(py::module_ &);

using namespace Kompass;

// Define a variant type to hold different parameter value types
using ParamValue = std::variant<double, int, bool, std::string>;

PYBIND11_MODULE(kompass_cpp, m) {
  m.doc() = "Algorithms for robot path tracking and control";

  bindings_types(m);
  bindings_config(m);
  bindings_control(m);
  bindings_mapping(m);

  // Utils bindings submodule
  py::enum_<LogLevel>(m, "LogLevel")
      .value("DEBUG", LogLevel::DEBUG)
      .value("INFO", LogLevel::INFO)
      .value("WARNING", LogLevel::WARNING)
      .value("ERROR", LogLevel::ERROR)
      .export_values();
  m.def("set_log_level", &setLogLevel, "Set the log level");
  m.def("set_log_file", &setLogFile, "Set the log file");
}
