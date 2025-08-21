#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

using namespace Kompass;

// Convert Point to JSON
inline void to_json(json &j, const Path::Point &p) {
  j = json{{"x", p.x()}, {"y", p.y()}};
}

// Convert JSON to Point
inline void from_json(const json &j, Path::Point &p) {
  p.x() = j.at("x");
  p.y() = j.at("y");
}

// Convert Path to JSON
inline void to_json(json &j, const Path::Path &p) {
  j["points"] = json::array(); // Initialize as a JSON array
  for (const auto &point : p) {
    j["points"].push_back(
        json{{"x", point.x()}, {"y", point.y()}}); // Serialize each Point
  }
}

// Convert TrajectoryPath to JSON
inline void to_json(json &j, const Control::TrajectoryPath &p) {
  j["points"] = json::array(); // Initialize as a JSON array
  for (const auto &point : p) {
    j["points"].push_back(
        json{{"x", point.x()}, {"y", point.y()}}); // Serialize each Point
  }
}

// Convert JSON to Path
inline void from_json(const json &j, Path::Path &p) {
  std::vector<Path::Point> points;
  for (const auto &item : j.at("points")) {
    Path::Point point;
    point.x() = item.at("x");
    point.y() = item.at("y");
    points.push_back(point); // Deserialize each Point
  }
  p = Path::Path(points, points.size());
}

// Convert Velocity to JSON
inline void to_json(json &j, const Control::Velocity2D &v) {
  j = json{{"vx", v.vx()},
           {"vy", v.vy()},
           {"omega", v.omega()},
           {"steer_ang", v.steer_ang()}};
}

// Convert JSON to Velocity
inline void from_json(const json &j, Control::Velocity2D &v) {
  v.setVx(j.at("vx"));
  v.setVy(j.at("vy"));
  v.setOmega(j.at("omega"));
  v.setSteerAng(j.at("steer_ang"));
}

// Convert Trajectory to JSON
inline void to_json(json &j, const Control::TrajectorySamples2D &samples) {
  j["paths"] = json::array(); // Initialize as a JSON array
  for (const auto &traj : samples) {
    json j_p;
    to_json(j_p, traj.path);
    j["paths"].push_back(j_p); // Serialize each Point
  }
}

// Convert Trajectory & Costs to JSON
inline void to_json(json &j, const Control::TrajectorySamples2D &samples,
             const std::vector<double> &costs) {
  j["paths"] = json::array(); // Initialize as a JSON array
  int idx{0};
  for (const auto &traj : samples) {
    json j_p;
    to_json(j_p, traj.path);
    j["paths"].push_back(
        json{{"path", j_p}, {"cost", costs[idx]}}); // Serialize each Point
    idx++;
  }
}

// Convert LaserScan to JSON
inline void to_json(json &j, const std::vector<double>& ranges,
                    const std::vector<double>& angles) {
  j = json{{"ranges", ranges}, {"angles", angles}};
}

// Save trajectories to a JSON file
void saveTrajectoriesToJson(const Control::TrajectorySamples2D &trajectories,
                            const std::string &filename) {
  json j;
  to_json(j, trajectories);
  std::ofstream file(filename);
  if (file.is_open()) {
    file << j.dump(4); // Pretty print with 4 spaces indentation
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filename << std::endl;
  }
}

void saveTrajectoryToJson(const Control::Trajectory2D &trajectory,
                          const std::string &filename) {
  json j;
  j["paths"] = json::array(); // Initialize as a JSON array
  json j_p;
  to_json(j_p, trajectory.path);
  j["paths"].push_back(j_p); // Serialize each Point
  std::ofstream file(filename);
  if (file.is_open()) {
    file << j.dump(4); // Pretty print with 4 spaces indentation
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filename << std::endl;
  }
}

// Save one path to a JSON file
inline void savePathToJson(const Path::Path &path, const std::string &filename) {
  json j;
  to_json(j, path);
  std::ofstream file(filename);
  if (file.is_open()) {
    file << j.dump(4); // Pretty print with 4 spaces indentation
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filename << std::endl;
  }
}

// Save LaserScan data (ranges and angles) to JSON
inline void saveScanToJson(const std::vector<double>& ranges,
                           const std::vector<double>& angles,
                           const std::string &filename) {
  json j;
  to_json(j, ranges, angles);
  std::ofstream file(filename);
  if (file.is_open()) {
    file << j.dump(4); // Pretty print with 4 spaces indentation
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filename << std::endl;
  }
}
