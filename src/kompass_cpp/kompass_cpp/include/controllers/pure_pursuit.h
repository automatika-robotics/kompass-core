#pragma once

#include "controllers/follower.h"
#include <cmath>

namespace Kompass {
namespace Control {

/**
 * @brief Basic Pure Pursuit Path Follower
 * Algorithm details from PURDUS SIGBOTS
 * https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit
 *
 */
class PurePursuit : public Follower {
public:
  class PurePursuitConfig : public Follower::FollowerParameters {
  public:
    PurePursuitConfig() : Follower::FollowerParameters() {
      addParameter("wheel_base", Parameter(0.34, 0.0, 100.0));
      addParameter(
          "lookahead_gain_forward",
          Parameter(0.8, 0.001, 10.0,
                    "Factor to scale lookahead distance by velocity (k * v)"));
    }
  };

  // Destructor
  virtual ~PurePursuit() = default;

  /**
   * @brief Construct a new Pure Pursuit object
   *
   * @param robotCtrlType
   * @param ctrlLimits
   * @param cfg
   */
  PurePursuit(const ControlType &robotCtrlType,
              const ControlLimitsParams &ctrlLimits,
              const PurePursuitConfig &cfg = PurePursuitConfig());

  /**
   * @brief Executes one Pure Pursuit control step
   *
   * @param currentPosition
   * @param deltaTime
   * @return Controller::Result
   */
  Controller::Result execute(Path::State currentPosition, double deltaTime);

  private:
    double wheel_base{0.0};
    double lookahead_gain_forward{0.0};

    size_t last_found_index_ = 0;

    Path::Point findLookaheadPoint(double radius);
};

} // namespace Control
} // namespace Kompass
