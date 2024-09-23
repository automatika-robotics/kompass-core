#pragma once

#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace Kompass {

enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

class Logger {
public:
  static Logger &getInstance() {
    static Logger instance;
    return instance;
  }

  void setLogLevel(LogLevel level) { logLevel = level; }

  void log(LogLevel level, const std::string &message) {
    if (level >= logLevel) {
      std::lock_guard<std::mutex> guard(logMutex);
      std::ostringstream logStream;
      logStream << "[" << getLogLevelString(level) << "] " << message
                << std::endl;
      std::cout << logStream.str();
      if (logFile.is_open()) {
        logFile << logStream.str();
      }
    }
  }

  template <typename T> void log(LogLevel level, const T &message) {
    std::ostringstream ss;
    ss << message;
    log(level, ss.str());
  }

  template <typename T, typename... Args>
  void log(LogLevel level, const T &first, const Args &...args) {
    std::ostringstream ss;
    logImpl(ss, first, args...);
    log(level, ss.str());
  }

  void setLogFile(const std::string &filename) {
    std::lock_guard<std::mutex> guard(logMutex);
    if (logFile.is_open()) {
      logFile.close();
    }
    logFile.open(filename, std::ios::out | std::ios::app);
    if (!logFile) {
      std::cerr << "Failed to open log file: " << filename << std::endl;
    }
  }

private:
  Logger() : logLevel(LogLevel::INFO) {}

  // Delete copy constructor and assignment operator
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  std::string getLogLevelString(LogLevel level) const {
    switch (level) {
    case LogLevel::DEBUG:
      return "DEBUG";
    case LogLevel::INFO:
      return "INFO";
    case LogLevel::WARNING:
      return "WARNING";
    case LogLevel::ERROR:
      return "ERROR";
    default:
      return "UNKNOWN";
    }
  }

  template <typename T, typename... Args>
  void logImpl(std::ostringstream &ss, const T &first, const Args &...args) {
    ss << first;
    if constexpr (sizeof...(args) > 0) {
      ss << " ";
      logImpl(ss, args...);
    }
  }

  LogLevel logLevel;
  std::ofstream logFile;
  std::mutex logMutex;
};

/**
 * @brief Set the Log Level object for the entire package
 *
 * @param level
 */
inline void setLogLevel(LogLevel level) {
  Logger::getInstance().setLogLevel(level);
}

/**
 * @brief Set the Log File object for the entire package
 *
 * @param filename
 */
inline void setLogFile(const std::string &filename) {
  Logger::getInstance().setLogFile(filename);
}

#define LOG_DEBUG(...)                                                         \
  Kompass::Logger::getInstance().log(Kompass::LogLevel::DEBUG, __VA_ARGS__)
#define LOG_INFO(...)                                                          \
  Kompass::Logger::getInstance().log(Kompass::LogLevel::INFO, __VA_ARGS__)
#define LOG_WARNING(...)                                                       \
  Kompass::Logger::getInstance().log(Kompass::LogLevel::WARNING, __VA_ARGS__)
#define LOG_ERROR(...)                                                         \
  Kompass::Logger::getInstance().log(Kompass::LogLevel::ERROR, __VA_ARGS__)

} // namespace Kompass
