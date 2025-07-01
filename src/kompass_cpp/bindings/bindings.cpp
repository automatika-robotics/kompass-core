#include <Eigen/Dense>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <variant>

#include "utils/gpu_check.h"
#include "utils/logger.h"

namespace py = nanobind;

void bindings_types(py::module_ &);
void bindings_config(py::module_ &);
void bindings_control(py::module_ &);
void bindings_mapping(py::module_ &);
void bindings_utils(py::module_ &);
void bindings_planning(py::module_ &);

using namespace Kompass;

// Define a variant type to hold different parameter value types
using ParamValue = std::variant<double, int, bool, std::string>;

NB_MODULE(kompass_cpp, m) {
  m.doc() = "Algorithms for robot path tracking and control";

  bindings_types(m);
  bindings_utils(m);
  bindings_config(m);
  bindings_control(m);
  bindings_mapping(m);
  bindings_planning(m);

  // Utils bindings submodule
  py::enum_<LogLevel>(m, "LogLevel")
      .value("DEBUG", LogLevel::DEBUG)
      .value("INFO", LogLevel::INFO)
      .value("WARNING", LogLevel::WARNING)
      .value("WARN", LogLevel::WARNING)
      .value("ERROR", LogLevel::ERROR)
      .export_values();

  m.def("set_log_level", &setLogLevel, "Set the log level");
  m.def("set_log_file", &setLogFile, "Set the log file");
  m.def("get_available_accelerators", &getAvailableAccelerators,
        "Get available accelerators");
}
