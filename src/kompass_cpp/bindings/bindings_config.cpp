#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "datatypes/parameter.h"

namespace py = nanobind;

// Method to set parameter values based on dict instance
void set_parameters_from_dict(Parameters &params,
                              const py::object &attrs_instance) {
    auto attrs_dict = py::cast<py::dict>(attrs_instance);
    for (const auto &item : attrs_dict) {
        const std::string name = py::cast<std::string>(item.first);
        py::handle value = item.second;
        try {
            auto it = params.parameters.find(name);
            if (it != params.parameters.end()) {
                if (py::isinstance<py::bool_>(value)) {
                    it->second.setValue(py::cast<bool>(value));
                } else if (py::isinstance<py::float_>(value)) {
                    it->second.setValue(py::cast<double>(value));
                } else if (py::isinstance<py::str>(value)) {
                    it->second.setValue(py::cast<std::string>(py::str(value)));
                } else if (py::isinstance<py::int_>(value)) {
                    it->second.setValue(py::cast<int>(value));
                }
            }
        } catch (const std::exception &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            throw py::python_error();
        }
    }
}

// Config parameters bindings submodule
void bindings_config(py::module_ &m) {
  auto m_config = m.def_submodule("configure", "Configuration classes");

  py::class_<Parameter>(m_config, "ConfigParameter")
      .def("set_value", (void(Parameter::*)(int)) & Parameter::setValue<int>)
      .def("set_value", (void(Parameter::*)(bool)) & Parameter::setValue<bool>)
      .def("set_value",
           (void(Parameter::*)(double)) & Parameter::setValue<double>)
      .def("set_value",
           (void(Parameter::*)(std::string)) & Parameter::setValue<std::string>)
      .def("get_value_int", &Parameter::getValue<int>)
      .def("get_value_double", &Parameter::getValue<double>)
      .def("get_value_string", &Parameter::getValue<std::string>)
      .def("get_value_bool", &Parameter::getValue<bool>);

  py::class_<Parameters>(m_config, "ConfigParameters")
      .def(py::init<>())
      .def("from_dict", &set_parameters_from_dict);
}
