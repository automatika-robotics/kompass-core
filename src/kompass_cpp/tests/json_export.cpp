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
void to_json(json &j, const Path::Point &p) {
  j = json{{"x", p.x}, {"y", p.y}};
}

// Convert JSON to Point
void from_json(const json &j, Path::Point &p) {
  j.at("x").get_to(p.x);
  j.at("y").get_to(p.y);
}

// Convert Path to JSON
void to_json(json &j, const Path::Path &p) {
  j["points"] = json::array(); // Initialize as a JSON array
  for (const auto &point : p.points) {
    j["points"].push_back(
        json{{"x", point.x}, {"y", point.y}}); // Serialize each Point
  }
}

// Convert JSON to Path
void from_json(const json &j, Path::Path &p) {
  p.points.clear(); // Clear existing points
  for (const auto &item : j.at("points")) {
    Path::Point point;
    item.at("x").get_to(point.x);
    item.at("y").get_to(point.y);
    p.points.push_back(point); // Deserialize each Point
  }
}

// Convert Velocity to JSON
void to_json(json &j, const Control::Velocity &v) {
  j = json{{"vx", v.vx},
           {"vy", v.vy},
           {"omega", v.omega},
           {"steer_ang", v.steer_ang}};
}

// Convert JSON to Velocity
void from_json(const json &j, Control::Velocity &v) {
  j.at("vx").get_to(v.vx);
  j.at("vy").get_to(v.vy);
  j.at("omega").get_to(v.omega);
  j.at("steer_ang").get_to(v.steer_ang);
}

// Convert Trajectory to JSON
void to_json(json &j, const std::vector<Control::Trajectory> &samples) {
  j["paths"] = json::array(); // Initialize as a JSON array
  for (const auto &traj : samples) {
    json j_p;
    to_json(j_p, traj.path);
    j["paths"].push_back(j_p); // Serialize each Point
  }
}

// Convert Trajectory & Costs to JSON
void to_json(json &j, const std::vector<Control::Trajectory> &samples,
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

// Convert JSON to Trajectory
void from_json(const json &j, std::vector<Control::Trajectory> &samples) {
  for (auto traj : samples) {
    traj.path.points.clear();
    from_json(j, traj.path);
  }
}

// Save trajectories to a JSON file
void saveTrajectoriesToJson(
    const std::vector<Control::Trajectory> &trajectories,
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

// Save one path to a JSON file
void savePathToJson(const Path::Path &path, const std::string &filename) {
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
