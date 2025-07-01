#include "controllers/controller.h"
#include "controllers/follower.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "json_export.h"
#include "test.h"
#include "utils/logger.h"
#include "utils/trajectory_sampler.h"
#include <Eigen/Dense>
#include <boost/dll/runtime_symbol_info.hpp> // for program_location
#include <boost/filesystem.hpp>
#include <string>
#include <vector>

using namespace Kompass;

void testTrajSampler() {

  Logger::getInstance().setLogLevel(LogLevel::DEBUG);

  // Create a test path
  // ------------------------------------------------------------------------------
  std::vector<Path::Point> points{Path::Point(0.0, 0.0, 0.0),
                                  Path::Point(1.0, 0.0, 0.0),
                                  Path::Point(2.0, 0.0, 0.0)};
  Path::Path raw_path(points, 500);

  // Generic follower to use for raw path interpolation and segmentation
  Control::Follower *follower = new Control::Follower();
  follower->setCurrentPath(raw_path);
  // Get the interpolated/segmented path
  Path::Path path = follower->getCurrentPath();

  // delete follower
  delete follower;
  // -----------------------------------------------------------------------------------------------------

  // Sampling configuration
  // ------------------------------------------------------------------------------
  double timeStep = 0.1;
  double predictionHorizon = 1.0;
  double controlHorizon = 0.2;
  int maxLinearSamples = 20;
  int maxAngularSamples = 20;
  int numThreads = 10;

  // Octomap resolution
  double octreeRes = 0.1;
  // -------------------------------------------------------------------------------------------------------

  // -------------------------------------------------------------------------------------------------------

  // Robot configuration
  // -----------------------------------------------------------------------------------
  Control::LinearVelocityControlParams x_params(1, 3, 5);
  Control::LinearVelocityControlParams y_params(1, 3, 5);
  Control::AngularVelocityControlParams angular_params(3.14, 3, 5, 8);
  Control::ControlLimitsParams controlLimits(x_params, y_params,
                                             angular_params);
  auto robotShapeType = Kompass::CollisionChecker::ShapeType::BOX;
  std::vector<float> robotDimensions{0.3, 0.3, 1.0};
  // std::array<float, 3> sensorPositionWRTbody {0.0, 0.0, 1.0};
  const Eigen::Vector3f sensor_position_body{0.0, 0.0, 0.5};
  const Eigen::Quaternionf sensor_rotation_body{0, 0, 0, 1};

  // Robot start state (pose)
  Path::State robotState(0.0, 0.0, 0.0, 0.0);

  // Robot laserscan value (empty)
  Control::LaserScan robotScan({20.0, 10.0, 10.0}, {0, 0.1, 0.2});

  std::array<Control::ControlType, 3> robot_types = {
      Control::ControlType::ACKERMANN, Control::ControlType::DIFFERENTIAL_DRIVE,
      Control::ControlType::OMNI};

  // -------------------------------------------------------------------------------------------------------

  // RUN TEST FOR EACH ROBOT TYPE
  // ------------------------------------------------------------------------
  for (size_t j = 0; j < robot_types.size(); j++) {
    Control::TrajectorySampler trajSampler(
        controlLimits, robot_types[j], timeStep, predictionHorizon,
        controlHorizon, maxLinearSamples, maxAngularSamples, robotShapeType,
        robotDimensions, sensor_position_body, sensor_rotation_body, octreeRes,
        numThreads);

    // Robot initial velocity control
    Control::Velocity2D robotControl;
    std::unique_ptr<Control::TrajectorySamples2D> samples;

    LOG_INFO("TESTING ", Control::controlTypeToString(robot_types[j]));

    {
      Timer time;
      samples =
          trajSampler.generateTrajectories(robotControl, robotState, robotScan);
    }

    // Plot the trajectories (Save to json then run python script for plotting)
    boost::filesystem::path executablePath = boost::dll::program_location();
    std::string file_location = executablePath.parent_path().string();

    std::string trajectories_filename =
        file_location + "/trajectories_" +
        Control::controlTypeToString(robot_types[j]);
    std::string ref_path_filename = file_location + "/ref_path";

    saveTrajectoriesToJson(*samples, trajectories_filename + ".json");
    savePathToJson(path, ref_path_filename + ".json");

    std::string command =
        "python3 " + file_location + "/trajectory_sampler_plt.py --samples \"" +
        trajectories_filename + "\" --reference \"" + ref_path_filename + "\"";

    // Execute the Python script
    int res = system(command.c_str());
    if (res != 0)
      throw std::system_error(res, std::generic_category(),
                              "Python script failed with error code");
  }
}

int main() { testTrajSampler(); }
