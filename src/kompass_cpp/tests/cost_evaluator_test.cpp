#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "test.h"
#include "utils/cost_evaluator.h"
#include <memory>
#define BOOST_TEST_MODULE KOMPASS COSTS TESTS
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <vector>

using namespace Kompass::Control;

// ==========================================================================
//  Correctness-focused cost tests
//  ------------------------------------------------------------------------
//  Each BOOST_AUTO_TEST_CASE below exercises one specific behaviour of one
//  cost function with a hand-computable expected value and a tolerance, so
//  the test fails with a numerical mismatch (not an index mismatch) when the
//  cost formula changes.
// ==========================================================================

namespace tt = boost::test_tools;

namespace {

// Build a straight reference path along +X from (0,0,0) to (length,0,0),
// interpolated at `interp` and segmented at `seg`. Returns the interpolated
// and segmented path.
Path::Path makeInterpolatedStraightPath(float length, float interp, float seg) {
  std::vector<Path::Point> pts{Path::Point(0.0f, 0.0f, 0.0f),
                               Path::Point(length, 0.0f, 0.0f)};
  Path::Path p(pts);
  p.interpolate(interp, Path::InterpolationType::LINEAR);
  p.segment(seg, 10000);
  return p;
}

// Build a 3/4 CCW circular reference path, radius R, centered at origin,
// starting at (R, 0) and ending near (0, -R). Interpolated at `interp` and
// segmented at `seg` (pass seg larger than the total arc length to get a
// single segment spanning the whole arc).
Path::Path makeInterpolated34Circle(float R, size_t input_pts, float interp,
                                    float seg) {
  std::vector<Path::Point> pts;
  pts.reserve(input_pts);
  const double max_theta = 3.0 * M_PI / 2.0;
  for (size_t i = 0; i < input_pts; ++i) {
    const double theta =
        (static_cast<double>(i) / static_cast<double>(input_pts - 1)) *
        max_theta;
    pts.emplace_back(R * std::cos(theta), R * std::sin(theta), 0.0);
  }
  Path::Path p(pts);
  p.interpolate(interp, Path::InterpolationType::LINEAR);
  p.segment(seg, 10000);
  return p;
}

// A single-sample TrajectorySamples2D whose path has every point equal to
// `endpoint` (constant-stationary path). Velocities are zero. Intended for
// goal-cost tests that only care about the trajectory endpoint.
std::unique_ptr<TrajectorySamples2D>
makeSingleSampleAtEndpoint(size_t num_points, Path::Point endpoint) {
  auto s = std::make_unique<TrajectorySamples2D>(1, num_points);
  TrajectoryPath path(num_points);
  TrajectoryVelocities2D vel(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    path.add(i, endpoint);
    if (i < num_points - 1) {
      vel.add(i, Velocity2D(0.0, 0.0, 0.0));
    }
  }
  s->push_back(vel, path);
  return s;
}

// Dummy control limits — only accLimits_ are read downstream, and only by
// smoothness/jerk (which we don't enable unless testing those costs).
ControlLimitsParams dummyControlLimits() {
  LinearVelocityControlParams x(1.0, 1.0, 1.0);
  LinearVelocityControlParams y(1.0, 1.0, 1.0);
  AngularVelocityControlParams a(1.0, 1.0, 1.0, 1.0);
  return ControlLimitsParams(x, y, a);
}

// Weights with every cost zero except `name`, which is set to `value`.
CostEvaluator::TrajectoryCostsWeights soloWeight(const char *name,
                                                 double value = 1.0) {
  CostEvaluator::TrajectoryCostsWeights w;
  w.setParameter("reference_path_distance_weight", 0.0);
  w.setParameter("goal_distance_weight", 0.0);
  w.setParameter("obstacles_distance_weight", 0.0);
  w.setParameter("smoothness_weight", 0.0);
  w.setParameter("jerk_weight", 0.0);
  w.setParameter(name, value);
  return w;
}

// Single-sample TrajectorySamples2D with the given path points (size defines
// num_points) and zero velocities. For tests that exercise path-related
// costs.
std::unique_ptr<TrajectorySamples2D>
makeSingleSampleWithPath(const std::vector<Path::Point> &path_pts) {
  const size_t n = path_pts.size();
  auto s = std::make_unique<TrajectorySamples2D>(1, n);
  TrajectoryPath path(n);
  TrajectoryVelocities2D vel(n);
  for (size_t i = 0; i < n; ++i) {
    path.add(i, path_pts[i]);
    if (i < n - 1) {
      vel.add(i, Velocity2D(0.0, 0.0, 0.0));
    }
  }
  s->push_back(vel, path);
  return s;
}

// Single-sample TrajectorySamples2D with the given path points AND explicit
// velocities. `velocities` must have size path_pts.size() - 1.
std::unique_ptr<TrajectorySamples2D>
makeSingleSampleWithVelocities(const std::vector<Path::Point> &path_pts,
                               const std::vector<Velocity2D> &velocities) {
  const size_t n = path_pts.size();
  BOOST_TEST_REQUIRE(velocities.size() == n - 1,
                     "velocities must have size path_pts.size() - 1");
  auto s = std::make_unique<TrajectorySamples2D>(1, n);
  TrajectoryPath path(n);
  TrajectoryVelocities2D vel(n);
  for (size_t i = 0; i < n; ++i) {
    path.add(i, path_pts[i]);
    if (i < n - 1) {
      vel.add(i, velocities[i]);
    }
  }
  s->push_back(vel, path);
  return s;
}

// Generic evaluator wrapper. If `obstacles` is non-empty, setPointScan is
// called with those points (Cartesian, zero pose, max_sensor_range = 30 so
// maxObstaclesDist = 10 via the default /3 multiple).
float evalCost(CostEvaluator::TrajectoryCostsWeights weights,
               const Path::Path &ref, size_t seg_idx,
               std::unique_ptr<TrajectorySamples2D> samples,
               const std::vector<Path::Point> &obstacles = {}) {
  auto limits = dummyControlLimits();
  CostEvaluator ev(weights, limits, samples->size(),
                   samples->numPointsPerTrajectory_, ref.getSize());
  if (!obstacles.empty()) {
    ev.setPointScan(obstacles, Path::State(), /*max_sensor_range=*/30.0f,
                    /*max_obstacle_cost_range_multiple=*/3.0f);
  }
  auto result = ev.getMinTrajectoryCost(samples, &ref, ref.getSegment(seg_idx));
  BOOST_TEST_REQUIRE(result.isTrajFound,
                     "CostEvaluator did not find a trajectory");
  return result.trajCost;
}

} // namespace

// -------- Goal cost -------------------------------------------------
// Sanity: endpoint sitting exactly on a known point of a straight tracked
// segment. arc_remaining = total_length - prefix_at_closest_idx, tie-breaker
// = 0. Cost is purely the normalized arc-remaining.
BOOST_AUTO_TEST_CASE(goal_cost_on_straight_path) {
  LOG_INFO(BOLD(FMAG("Running goal_cost_on_straight_path")));
  Timer t;
  // 10 m path, interp 1 m → 11 points at X = 0, 1, ..., 10.
  // Segment length 5 m → segment 0 spans indices [0, 5),
  //   i.e. points at X = 0, 1, 2, 3, 4.
  Path::Path ref = makeInterpolatedStraightPath(10.0f, 1.0f, 5.0f);

  // Endpoint exactly on interpolated point idx 4: closest_abs_idx = 4,
  // prefix = 4.0, arc_remaining_normalized = (10 - 4)/10 = 0.6,
  // tie-breaker = 0 / 10 = 0.
  auto samples = makeSingleSampleAtEndpoint(5, Path::Point(4.0f, 0.0f, 0.0f));
  const float cost = evalCost(soloWeight("goal_distance_weight"), ref,
                              /*seg_idx=*/0, std::move(samples));

  BOOST_TEST(cost == 0.6f, tt::tolerance(1e-4f));
}

// Regression for issue #40: on a 3/4 circle, a chord-cutting endpoint that
// falls outside the arc's angular sweep must receive a HIGHER goal cost
// than an arc-following endpoint at the same travel distance. Under the old
// euclidean goal cost the chord-cutter wins, which pulls DWA off the arc.
BOOST_AUTO_TEST_CASE(goal_cost_arc_remaining_on_curved_path) {
  LOG_INFO(BOLD(FMAG("Running goal_cost_arc_remaining_on_curved_path")));
  Timer t;
  const float R = 2.0f;
  // Segment > total arc length so the tracked segment spans the whole arc
  // (both endpoints below are therefore reachable from segment 0).
  Path::Path ref = makeInterpolated34Circle(R, /*input_pts=*/60,
                                            /*interp=*/0.05f, /*seg=*/20.0f);
  const float total = ref.totalPathLength();

  // Arc-follower: endpoint sits on the arc at θ = 0.5 rad, arc prefix R*0.5.
  const double theta_follow = 0.5;
  const Path::Point follow_pt(R * std::cos(theta_follow),
                              R * std::sin(theta_follow), 0.0);
  const float follow_cost = evalCost(soloWeight("goal_distance_weight"), ref, 0,
                                     makeSingleSampleAtEndpoint(5, follow_pt));

  // Chord-cutter: (1.5, -0.5) lies in the 4th quadrant at angle ≈ -18° — i.e.
  // outside the arc's [0, 3π/2] angular sweep — and is closer to the arc
  // start (R, 0) than to the arc end (0, -R). The kernel's closest-point
  // search therefore snaps to idx 0, giving arc_remaining_normalized = 1.
  const Path::Point chord_pt(1.5f, -0.5f, 0.0f);
  const float chord_cost = evalCost(soloWeight("goal_distance_weight"), ref, 0,
                                    makeSingleSampleAtEndpoint(5, chord_pt));

  // Expected values (loose tolerance to absorb interpolation rounding).
  const float expected_follow = (total - R * 0.5f) / total; // tie-breaker ≈ 0
  BOOST_TEST(follow_cost == expected_follow, tt::tolerance(0.02f));

  // Chord-cutter: closest-arc-point = (R, 0), euclidean dist = sqrt(0.5).
  const float expected_chord =
      1.0f + static_cast<float>(std::sqrt(0.5)) / total;
  BOOST_TEST(chord_cost == expected_chord, tt::tolerance(0.02f));

  // Core regression assertion.
  BOOST_TEST(follow_cost < chord_cost,
             "Arc follower must beat chord cutter under arc-remaining goal "
             "cost (issue #40).");
}

// Tie-breaker: two samples with the same closest_abs_idx but different
// lateral offsets should differ in cost by exactly the normalized euclidean
// distance delta.
BOOST_AUTO_TEST_CASE(goal_cost_tie_breaker) {
  LOG_INFO(BOLD(FMAG("Running goal_cost_tie_breaker")));
  Timer t;
  Path::Path ref = makeInterpolatedStraightPath(10.0f, 1.0f, 5.0f);

  // Both endpoints have X = 4 → closest_abs_idx = 4 on the straight path.
  // arc_remaining_normalized = 0.6 for both; only tie-breaker differs.
  auto sampleA = makeSingleSampleAtEndpoint(5, Path::Point(4.0f, 0.1f, 0.0f));
  auto sampleB = makeSingleSampleAtEndpoint(5, Path::Point(4.0f, 0.5f, 0.0f));

  const float cost_A =
      evalCost(soloWeight("goal_distance_weight"), ref, 0, std::move(sampleA));
  const float cost_B =
      evalCost(soloWeight("goal_distance_weight"), ref, 0, std::move(sampleB));

  // Expected: 0.6 + 0.1/10 = 0.61, and 0.6 + 0.5/10 = 0.65.
  BOOST_TEST(cost_A == 0.61f, tt::tolerance(1e-4f));
  BOOST_TEST(cost_B == 0.65f, tt::tolerance(1e-4f));
  BOOST_TEST(cost_A < cost_B);
}

// -------- Path cost -------------------------------------------------------

// Sample whose every point sits exactly on the tracked segment. Average
// cross-track error = 0; end-point euclidean distance from the segment end
// = 0. Cost = (0 + 0)/2 = 0.
BOOST_AUTO_TEST_CASE(path_cost_centered_sample) {
  LOG_INFO(BOLD(FMAG("Running path_cost_centered_sample")));
  Timer t;
  // Segment 0: points at X = 0, 1, 2, 3, 4 (5 points, total seg length 4 m).
  Path::Path ref = makeInterpolatedStraightPath(10.0f, 1.0f, 5.0f);
  const std::vector<Path::Point> on_path{
      Path::Point(0.0f, 0.0f, 0.0f), Path::Point(1.0f, 0.0f, 0.0f),
      Path::Point(2.0f, 0.0f, 0.0f), Path::Point(3.0f, 0.0f, 0.0f),
      Path::Point(4.0f, 0.0f, 0.0f)};
  auto samples = makeSingleSampleWithPath(on_path);
  const float cost = evalCost(soloWeight("reference_path_distance_weight"), ref,
                              0, std::move(samples));
  BOOST_TEST(cost == 0.0f, tt::tolerance(1e-4f));
}

// Sample running parallel to the segment at constant lateral offset d.
// Each trajectory point's closest segment point is directly opposite it, so
// min_dist = d for every point → avg cross-track = d. The trajectory end
// point is offset by d from the segment end → end_dist = d, normalized =
// d / seg_length. Cost = (d + d/seg_length) / 2.
BOOST_AUTO_TEST_CASE(path_cost_constant_lateral_offset) {
  LOG_INFO(BOLD(FMAG("Running path_cost_constant_lateral_offset")));
  Timer t;
  Path::Path ref = makeInterpolatedStraightPath(10.0f, 1.0f, 5.0f);
  // Segment length is sum of distances between the 5 segment points (X =
  // 0..4, step 1) = 4 m.
  const float seg_len = 4.0f;
  const float d = 0.5f;
  const std::vector<Path::Point> offset_path{
      Path::Point(0.0f, d, 0.0f), Path::Point(1.0f, d, 0.0f),
      Path::Point(2.0f, d, 0.0f), Path::Point(3.0f, d, 0.0f),
      Path::Point(4.0f, d, 0.0f)};
  auto samples = makeSingleSampleWithPath(offset_path);
  const float cost = evalCost(soloWeight("reference_path_distance_weight"), ref,
                              0, std::move(samples));
  const float expected = (d + d / seg_len) / 2.0f;
  BOOST_TEST(cost == expected, tt::tolerance(1e-4f));
}

// -------- Smoothness cost -------------------------------------------------

// Constant velocity → zero velocity deltas → zero smoothness cost.
BOOST_AUTO_TEST_CASE(smoothness_cost_constant_velocity) {
  LOG_INFO(BOLD(FMAG("Running smoothness_cost_constant_velocity")));
  Timer t;
  Path::Path ref = makeInterpolatedStraightPath(10.0f, 1.0f, 5.0f);
  // 5 path points, 4 velocities, all constant at (1, 0, 0).
  const std::vector<Path::Point> pts(5, Path::Point(0.0f, 0.0f, 0.0f));
  const std::vector<Velocity2D> vels(4, Velocity2D(1.0, 0.0, 0.0));
  auto samples = makeSingleSampleWithVelocities(pts, vels);
  const float cost =
      evalCost(soloWeight("smoothness_weight"), ref, 0, std::move(samples));
  BOOST_TEST(cost == 0.0f, tt::tolerance(1e-4f));
}

// Single step change in vx: velocities = [0, 1, 1, 1]. Loop iterates i in
// {1,2,3}. Only i=1 contributes a nonzero delta (1). Squared delta = 1.
// Divisor = accLimits_[0] (1.0 with dummy limits) × 3 × vx.size() (=4) = 12.
// Cost = 1 / 12.
BOOST_AUTO_TEST_CASE(smoothness_cost_single_step_change) {
  LOG_INFO(BOLD(FMAG("Running smoothness_cost_single_step_change")));
  Timer t;
  Path::Path ref = makeInterpolatedStraightPath(10.0f, 1.0f, 5.0f);
  const std::vector<Path::Point> pts(5, Path::Point(0.0f, 0.0f, 0.0f));
  const std::vector<Velocity2D> vels{
      Velocity2D(0.0, 0.0, 0.0), Velocity2D(1.0, 0.0, 0.0),
      Velocity2D(1.0, 0.0, 0.0), Velocity2D(1.0, 0.0, 0.0)};
  auto samples = makeSingleSampleWithVelocities(pts, vels);
  const float cost =
      evalCost(soloWeight("smoothness_weight"), ref, 0, std::move(samples));
  BOOST_TEST(cost == 1.0f / 12.0f, tt::tolerance(1e-4f));
}

// -------- Jerk cost -------------------------------------------------------

// Constant acceleration → zero second difference of velocity → zero jerk.
BOOST_AUTO_TEST_CASE(jerk_cost_constant_acceleration) {
  LOG_INFO(BOLD(FMAG("Running jerk_cost_constant_acceleration")));
  Timer t;
  Path::Path ref = makeInterpolatedStraightPath(10.0f, 1.0f, 5.0f);
  // Linear vx ramp — second differences are zero.
  const std::vector<Path::Point> pts(5, Path::Point(0.0f, 0.0f, 0.0f));
  const std::vector<Velocity2D> vels{
      Velocity2D(0.1, 0.0, 0.0), Velocity2D(0.2, 0.0, 0.0),
      Velocity2D(0.3, 0.0, 0.0), Velocity2D(0.4, 0.0, 0.0)};
  auto samples = makeSingleSampleWithVelocities(pts, vels);
  const float cost =
      evalCost(soloWeight("jerk_weight"), ref, 0, std::move(samples));
  BOOST_TEST(cost == 0.0f, tt::tolerance(1e-4f));
}

// Known second difference in vx: velocities = [0, 1, 3, 6].
//   i=2: v[2] - 2*v[1] + v[0] = 3 - 2 + 0 = 1   → sq = 1
//   i=3: v[3] - 2*v[2] + v[1] = 6 - 6 + 1 = 1   → sq = 1
// Sum = 2. Divisor = accLimits_[0] (1.0) × 3 × vx.size() (=4) = 12.
// Cost = 2 / 12 = 1/6.
BOOST_AUTO_TEST_CASE(jerk_cost_known_second_diff) {
  LOG_INFO(BOLD(FMAG("Running jerk_cost_known_second_diff")));
  Timer t;
  Path::Path ref = makeInterpolatedStraightPath(10.0f, 1.0f, 5.0f);
  const std::vector<Path::Point> pts(5, Path::Point(0.0f, 0.0f, 0.0f));
  const std::vector<Velocity2D> vels{
      Velocity2D(0.0, 0.0, 0.0), Velocity2D(1.0, 0.0, 0.0),
      Velocity2D(3.0, 0.0, 0.0), Velocity2D(6.0, 0.0, 0.0)};
  auto samples = makeSingleSampleWithVelocities(pts, vels);
  const float cost =
      evalCost(soloWeight("jerk_weight"), ref, 0, std::move(samples));
  BOOST_TEST(cost == 2.0f / 12.0f, tt::tolerance(1e-4f));
}

// -------- Obstacles cost --------------------------------------------------
// With the default scan setup (max_sensor_range = 30, /3 multiple) →
// maxObstaclesDist = 10. Formula: max(maxObstaclesDist - min_dist, 0) /
// maxObstaclesDist. min_dist is euclidean from any trajectory point to any
// obstacle.

// Obstacle placed far enough away that min_dist >= maxObstaclesDist → cost
// clamps to 0.
BOOST_AUTO_TEST_CASE(obstacles_cost_at_max_range) {
  LOG_INFO(BOLD(FMAG("Running obstacles_cost_at_max_range")));
  Timer t;
  Path::Path ref = makeInterpolatedStraightPath(10.0f, 1.0f, 5.0f);
  // All trajectory points at origin. Obstacle at (20, 0) → min_dist = 20 >
  // maxObstaclesDist (10) → cost = 0.
  auto samples = makeSingleSampleAtEndpoint(5, Path::Point(0.0f, 0.0f, 0.0f));
  const std::vector<Path::Point> obstacles{Path::Point(20.0f, 0.0f, 0.0f)};
  const float cost = evalCost(soloWeight("obstacles_distance_weight"), ref, 0,
                              std::move(samples), obstacles);
  BOOST_TEST(cost == 0.0f, tt::tolerance(1e-4f));
}

// Obstacle exactly on the trajectory → min_dist = 0 → cost = 1.
BOOST_AUTO_TEST_CASE(obstacles_cost_at_zero_distance) {
  LOG_INFO(BOLD(FMAG("Running obstacles_cost_at_zero_distance")));
  Timer t;
  Path::Path ref = makeInterpolatedStraightPath(10.0f, 1.0f, 5.0f);
  auto samples = makeSingleSampleAtEndpoint(5, Path::Point(0.0f, 0.0f, 0.0f));
  const std::vector<Path::Point> obstacles{Path::Point(0.0f, 0.0f, 0.0f)};
  const float cost = evalCost(soloWeight("obstacles_distance_weight"), ref, 0,
                              std::move(samples), obstacles);
  BOOST_TEST(cost == 1.0f, tt::tolerance(1e-4f));
}

// Obstacle at half of maxObstaclesDist → cost = (10 - 5) / 10 = 0.5.
BOOST_AUTO_TEST_CASE(obstacles_cost_at_half_range) {
  LOG_INFO(BOLD(FMAG("Running obstacles_cost_at_half_range")));
  Timer t;
  Path::Path ref = makeInterpolatedStraightPath(10.0f, 1.0f, 5.0f);
  auto samples = makeSingleSampleAtEndpoint(5, Path::Point(0.0f, 0.0f, 0.0f));
  const std::vector<Path::Point> obstacles{Path::Point(5.0f, 0.0f, 0.0f)};
  const float cost = evalCost(soloWeight("obstacles_distance_weight"), ref, 0,
                              std::move(samples), obstacles);
  BOOST_TEST(cost == 0.5f, tt::tolerance(1e-4f));
}
