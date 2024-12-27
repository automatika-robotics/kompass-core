#pragma once

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>

/**
 * @brief Class used to define a configuration parameter of type [integer,
 * double, string, boolean] A Parameter is defined with a default value, max and
 * min value (if applicable). A string description of the parameter can be
 * optionally added.
 */
class Parameter {
public:
  /**
   * @brief Possible Parameter types
   *
   */
  using ValueType = std::variant<int, double, std::string, bool>;

  Parameter()
      : defaultValue(0), minValue(0), maxValue(0), currentValue(0),
        description("Parameter") {}

  /**
   * @brief Construct a new Parameter object with default, min and max values -
   * Used for integer and double
   *
   * @tparam T
   * @param defaultValue
   * @param minValue
   * @param maxValue
   * @param description
   */
  template <typename T>
  Parameter(T defaultValue, T minValue, T maxValue,
            std::string description = "Parameter")
      : defaultValue(defaultValue), minValue(minValue), maxValue(maxValue),
        currentValue(defaultValue), description(description) {}

  /**
   * @brief Construct a new Parameter object with a default value - Used for
   * string/bool, can also be used for integer and double
   *
   * @tparam T
   * @param defaultValue
   * @param description
   */
  template <typename T>
  Parameter(T defaultValue, std::string description = "Parameter")
      : defaultValue(defaultValue), minValue(0), maxValue(0),
        currentValue(defaultValue), description(description) {}

  /**
   * @brief Construct a new Parameter object using another parameter object
   *
   * @param other
   */
  Parameter(const Parameter &other)
      : defaultValue(other.defaultValue), minValue(other.minValue),
        maxValue(other.maxValue), currentValue(other.currentValue),
        description(other.description) {}

  /**
   * @brief Assignment operator
   *
   * @param other
   * @return Parameter&
   */
  Parameter &operator=(const Parameter &other) {
    if (this != &other) {
      defaultValue = other.defaultValue;
      minValue = other.minValue;
      maxValue = other.maxValue;
      currentValue = other.currentValue;
    }
    return *this;
  }

  /**
   * @brief Friend function for the overloaded << operator
   *
   * @param os
   * @param obj
   * @return std::ostream&
   */
  friend std::ostream &operator<<(std::ostream &os, const Parameter &obj) {
    os << "Parameter:" << obj.description << "\n";
    return os;
  }

  /**
   * @brief Set the Parameter value object
   *
   * @tparam T
   * @param value
   */
  template <typename T> void setValue(T value) {
    if (std::holds_alternative<T>(defaultValue)) {
      if (minValue.index() == defaultValue.index() &&
          maxValue.index() == defaultValue.index()) {
        if (value < std::get<T>(minValue) || value > std::get<T>(maxValue)) {
          throw std::out_of_range("Parameter value is out of range");
        }
      }
      currentValue = value;
    } else {
      throw std::invalid_argument("Parameter value type mismatch");
    }
  }

  /**
   * @brief Get the Info object
   *
   * @return std::string
   */
  std::string getInfo() const { return description; }

  /**
   * @brief Get the Value object
   *
   * @return auto
   */
  auto getValue() const { return currentValue; }

  /**
   * @brief Get the Value object
   *
   * @tparam T
   * @return T
   */
  template <typename T> T getValue() const {
    if (std::holds_alternative<T>(currentValue)) {
      return std::get<T>(currentValue);
    } else {
      throw std::invalid_argument("Type mismatch");
    }
  }

private:
  ValueType defaultValue;
  ValueType minValue;
  ValueType maxValue;
  ValueType currentValue;
  std::string description;
};

/**
 * @brief Class used to define a set of configuration parameters
 */
class Parameters {
public:
  /**
   * @brief Add a new Parameter
   *
   * @param name
   * @param parameter
   */
  void addParameter(const std::string &name, const Parameter &parameter) {
    parameters[name] = parameter;
  }

  /**
   * @brief  Add a new Parameter from given value
   *
   * @tparam T
   * @param name
   * @param defaultValue
   */
  template <typename T>
  void addParameter(const std::string &name, T defaultValue) {
    parameters[name] = Parameter(defaultValue);
  }

  /**
   * @brief Add a new Parameter from given value, min and max
   *
   * @tparam T
   * @param name
   * @param defaultValue
   * @param minValue
   * @param maxValue
   */
  template <typename T>
  void addParameter(const std::string &name, T defaultValue, T minValue,
                    T maxValue) {
    parameters[name] = Parameter(defaultValue, minValue, maxValue);
  }

  // Overloads for specific types
  void addParameter(const std::string &name, const char *defaultValue) {
    parameters[name] = Parameter(std::string(defaultValue));
  }

  void addParameter(const std::string &name, bool defaultValue) {
    parameters[name] = Parameter(defaultValue);
  }

  void addParameter(const std::string &name, int defaultValue, int minValue,
                    int maxValue) {
    parameters[name] = Parameter(defaultValue, minValue, maxValue);
  }

  void addParameter(const std::string &name, double defaultValue,
                    double minValue, double maxValue) {
    parameters[name] = Parameter(defaultValue, minValue, maxValue);
  }

  /**
   * @brief Set the Parameter object
   *
   * @tparam T
   * @param name
   * @param value
   */
  template <typename T> void setParameter(const std::string &name, T value) {
    if (parameters.find(name) != parameters.end()) {
      parameters[name].setValue(value);
    } else {
      throw std::invalid_argument("Parameter {" + name + "} not found");
    }
  }

  /**
   * @brief Get the Parameter object
   *
   * @tparam T
   * @param name
   * @return T
   */
  template <typename T> T getParameter(const std::string &name) const {
    if (parameters.find(name) != parameters.end()) {
      return parameters.at(name).getValue<T>();
    } else {
      throw std::invalid_argument("Parameter {" + name + "} not found");
    }
  }

  /**
   * @brief Proxy class to provide parameter access using the [] operator
   *
   */
  class ParameterProxy {
  public:
    ParameterProxy(Parameters &parameterList, const std::string &name)
        : parameterList(parameterList), name(name) {}

    template <typename T> operator T() const {
      return parameterList.getParameter<T>(name);
    }

    template <typename T> ParameterProxy &operator=(T value) {
      parameterList.setParameter(name, value);
      return *this;
    }

  private:
    Parameters &parameterList;
    std::string name;
  };

  ParameterProxy operator[](const std::string &name) {
    return ParameterProxy(*this, name);
  }

  // Assignment operator for Parameters
  Parameters &operator=(const Parameters &other) {
    if (this != &other) {
      parameters = other.parameters;
    }
    return *this;
  }

  std::map<std::string, Parameter> parameters;
};
