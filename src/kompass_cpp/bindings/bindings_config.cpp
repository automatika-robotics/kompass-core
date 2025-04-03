#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "datatypes/parameter.h"

namespace py = pybind11;

// Method to set parameter values based on dict instance
void set_parameters_from_dict(Parameters &params,
                              const py::object &attrs_instance) {
  auto attrs_dict = attrs_instance.cast<py::dict>();
  for (const auto &item : attrs_dict) {
    const std::string &name = py::str(item.first);
    py::handle value = item.second;
    try {
      auto it = params.parameters.find(name);
      if (it != params.parameters.end()) {
        if (py::isinstance<py::bool_>(value)) {
          it->second.setValue(value.cast<bool>());
        } else if (py::isinstance<py::float_>(value)) {
          it->second.setValue(value.cast<double>());
        } else if (py::isinstance<py::str>(value)) {
          it->second.setValue(py::str(value).cast<std::string>());
        } else if (py::isinstance<py::int_>(value)) {
          it->second.setValue(value.cast<int>());
        }
      }
    } catch (const std::exception &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      throw py::error_already_set();
    }
  }
}

// Config parameters bindings submodule
void bindings_config(py::module_ &m) {
  auto m_config = m.def_submodule("configure", "Configuration classes");

  py::class_<Parameter>(m_config, "ConfigParameter")
      .def(py::init<>())
      .def(py::init<int, int, int>())
      .def(py::init<double, double, double>())
      .def(py::init<std::string>())
      .def(py::init<bool>())
      .def("set_value", (void(Parameter::*)(int)) & Parameter::setValue<int>)
      .def("set_value",
           (void(Parameter::*)(double)) & Parameter::setValue<double>)
      .def("set_value",
           (void(Parameter::*)(std::string)) & Parameter::setValue<std::string>)
      .def("get_value_int", &Parameter::getValue<int>)
      .def("get_value_double", &Parameter::getValue<double>)
      .def("get_value_string", &Parameter::getValue<std::string>);

  py::class_<Parameters>(m_config, "ConfigParameters")
      .def(py::init<>())
      .def("add_parameter",
           (void(Parameters::*)(const std::string &, const Parameter &)) &
               Parameters::addParameter)
      .def("add_parameter", (void(Parameters::*)(const std::string &, bool)) &
                                Parameters::addParameter)
      .def("add_parameter",
           (void(Parameters::*)(const std::string &, const char *)) &
               Parameters::addParameter)
      .def("add_parameter",
           (void(Parameters::*)(const std::string &, int, int, int)) &
               Parameters::addParameter)
      .def("add_parameter",
           (void(Parameters::*)(const std::string &, double, double, double)) &
               Parameters::addParameter)
      .def("from_dict", &set_parameters_from_dict);
}
