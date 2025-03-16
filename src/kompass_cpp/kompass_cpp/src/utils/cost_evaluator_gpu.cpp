#ifdef GPU
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "utils/cost_evaluator.h"
#include "utils/logger.h"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Transform.h>
#include <cstddef>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include <vector>

namespace Kompass {

namespace Control {
CostEvaluator::CostEvaluator(TrajectoryCostsWeights costsWeights,
                             ControlType controlType,
                             ControlLimitsParams ctrLimits, double timeStep,
                             double timeHorizon, size_t maxLinearSamples,
                             size_t maxAngularSamples, size_t maxPathLength) {

  this->costWeights = costsWeights;

  numPointsPerTrajectory_ = getNumPointsPerTrajectory(timeStep, timeHorizon);
  numTrajectories_ =
      getNumTrajectories(ctrType, maxLinearSamples, maxAngularSamples);

  accLimits_ = {ctrLimits.velXParams.maxAcceleration,
                ctrLimits.velYParams.maxAcceleration,
                ctrLimits.omegaParams.maxAcceleration};
  m_q =
      sycl::queue{sycl::default_selector_v, sycl::property::queue::in_order{}};
  auto dev = m_q.get_device();
  LOG_INFO("Running on :", dev.get_info<sycl::info::device::name>());
  m_devicePtrPathsX = sycl::malloc_device<double>(
      numTrajectories_ * numPointsPerTrajectory_, m_q);
  m_devicePtrPathsY = sycl::malloc_device<double>(
      numTrajectories_ * numPointsPerTrajectory_, m_q);
  m_devicePtrVelocitiesVx = sycl::malloc_device<double>(
      numTrajectories_ * numPointsPerTrajectory_ - 1, m_q);
  m_devicePtrVelocitiesVy = sycl::malloc_device<double>(
      numTrajectories_ * numPointsPerTrajectory_ - 1, m_q);
  m_devicePtrVelocitiesOmega = sycl::malloc_device<double>(
      numTrajectories_ * numPointsPerTrajectory_ - 1, m_q);
  m_devicePtrCosts = sycl::malloc_device<double>(numTrajectories_, m_q);
  if (costWeights.getParameter<double>("reference_path_distance_weight") >
      0.0) {
    m_devicePtrReferencePath = sycl::malloc_device<double>(maxPathLength, m_q);
    m_minCost = sycl::malloc_shared<LowestCost>(1, m_q);
  };
}

CostEvaluator::CostEvaluator(TrajectoryCostsWeights costsWeights,
                             const std::array<float, 3> &sensor_position_body,
                             const std::array<float, 4> &sensor_rotation_body,
                             ControlType controlType,
                             ControlLimitsParams ctrLimits, double timeStep,
                             double timeHorizon, size_t maxLinearSamples,
                             size_t maxAngularSamples, size_t maxPathLength) {

  sensor_tf_body_ =
      getTransformation(Eigen::Quaternionf(sensor_rotation_body.data()),
                        Eigen::Vector3f(sensor_position_body.data()));
  this->costWeights = costsWeights;

  accLimits_ = {ctrLimits.velXParams.maxAcceleration,
                ctrLimits.velYParams.maxAcceleration,
                ctrLimits.omegaParams.maxAcceleration};

  numPointsPerTrajectory_ = getNumPointsPerTrajectory(timeStep, timeHorizon);
  numTrajectories_ =
      getNumTrajectories(ctrType, maxLinearSamples, maxAngularSamples);

  m_q =
      sycl::queue{sycl::default_selector_v, sycl::property::queue::in_order{}};
  auto dev = m_q.get_device();
  LOG_INFO("Running on :", dev.get_info<sycl::info::device::name>());
  m_devicePtrPathsX = sycl::malloc_device<double>(
      numTrajectories_ * numPointsPerTrajectory_, m_q);
  m_devicePtrPathsY = sycl::malloc_device<double>(
      numTrajectories_ * numPointsPerTrajectory_, m_q);
  m_devicePtrVelocitiesVx = sycl::malloc_device<double>(
      numTrajectories_ * numPointsPerTrajectory_ - 1, m_q);
  m_devicePtrVelocitiesVy = sycl::malloc_device<double>(
      numTrajectories_ * numPointsPerTrajectory_ - 1, m_q);
  m_devicePtrVelocitiesOmega = sycl::malloc_device<double>(
      numTrajectories_ * numPointsPerTrajectory_ - 1, m_q);
  m_devicePtrCosts = sycl::malloc_device<double>(numTrajectories_, m_q);
  if (costWeights.getParameter<double>("reference_path_distance_weight") >
      0.0) {
    m_devicePtrReferencePath = sycl::malloc_device<double>(maxPathLength, m_q);
  };
  m_minCost = sycl::malloc_shared<LowestCost>(1, m_q);
}

CostEvaluator::~CostEvaluator() {

  // delete and clear custom cost pointers
  for (auto ptr : customTrajCostsPtrs_) {
    delete ptr;
  }
  customTrajCostsPtrs_.clear();

  // unallocate device memory
  if (m_devicePtrPathsX) {
    sycl::free(m_devicePtrPathsX, m_q);
  }
  if (m_devicePtrPathsY) {
    sycl::free(m_devicePtrPathsY, m_q);
  }
  if (m_devicePtrVelocitiesVx) {
    sycl::free(m_devicePtrVelocitiesVx, m_q);
  }
  if (m_devicePtrVelocitiesVy) {
    sycl::free(m_devicePtrVelocitiesVy, m_q);
  }
  if (m_devicePtrVelocitiesOmega) {
    sycl::free(m_devicePtrVelocitiesOmega, m_q);
  }
  if (m_devicePtrCosts) {
    sycl::free(m_devicePtrCosts, m_q);
  }
  if (costWeights.getParameter<double>("reference_path_distance_weight") >
      0.0) {
    sycl::free(m_devicePtrReferencePath, m_q);
  };
  if (m_minCost) {
    sycl::free(m_minCost, m_q);
  }
};

TrajSearchResult CostEvaluator::getMinTrajectoryCost(
    const TrajectorySamples2D &trajs, const Path::Path &reference_path,
    const Path::Path &tracked_segment, const size_t closest_segment_index) {

  try {
    double weight;
    double total_cost;

    m_q.fill(m_devicePtrCosts, 0.0, trajs.size());

    m_q.memcpy(m_devicePtrPathsX, trajs.paths.x.data(),
               sizeof(double) * numTrajectories_ * numPointsPerTrajectory_);
    m_q.memcpy(m_devicePtrPathsY, trajs.paths.y.data(),
               sizeof(double) * numTrajectories_ * numPointsPerTrajectory_);
    m_q.memcpy(m_devicePtrVelocitiesVx, trajs.velocities.vx.data(),
               sizeof(double) * numTrajectories_ * numPointsPerTrajectory_ - 1);
    m_q.memcpy(m_devicePtrVelocitiesVy, trajs.velocities.vy.data(),
               sizeof(double) * numTrajectories_ * numPointsPerTrajectory_ - 1);
    m_q.memcpy(m_devicePtrVelocitiesOmega, trajs.velocities.omega.data(),
               sizeof(double) * numTrajectories_ * numPointsPerTrajectory_ - 1);
    *m_minCost = LowestCost();

    /*for (const auto &traj : trajs) {*/
    /*  total_cost = 0.0;*/
    /*  if (reference_path.totalPathLength() > 0.0) {*/
    /*    if ((weight = costWeights.getParameter<double>("goal_distance_weight")
     * >*/
    /*                  0.0)) {*/
    /*      double goalCost = goalCostFunc(traj, reference_path);*/
    /*      total_cost += weight * goalCost;*/
    /*    }*/
    /*    if ((weight = costWeights.getParameter<double>(*/
    /*                      "reference_path_distance_weight") > 0.0)) {*/
    /*      double refPathCost = pathCostFunc(traj, reference_path);*/
    /*      total_cost += weight * refPathCost;*/
    /*    }*/
    /*  }*/
    /**/
    /*  if (obstaclePoints.size() > 0 and*/
    /*      (weight = costWeights.getParameter<double>(*/
    /*           "obstacles_distance_weight")) > 0.0) {*/
    /**/
    /*    double objCost = obstaclesDistCostFunc(traj, obstaclePoints);*/
    /*    total_cost += weight * objCost;*/
    /*  }*/
    /**/
    /*  for (const auto &custom_cost : customTrajCostsPtrs_) {*/
    /*    // custom cost functions takes in the trajectory and the reference
     * path*/
    /*    total_cost +=*/
    /*        custom_cost->weight * custom_cost->evaluator_(traj,
     * reference_path);*/
    /*  }*/
    /**/
    /*  if (total_cost < minCost) {*/
    /*    minCost = total_cost;*/
    /*    minCostTraj = traj;*/
    /*    traj_found = true;*/
    /*  }*/

    if ((weight =
             costWeights.getParameter<double>("smoothness_weight") > 0.0)) {
      smoothnessCostFunc(trajs.size(), weight);
    }
    if ((weight = costWeights.getParameter<double>("jerk_weight") > 0.0)) {
      jerkCostFunc(trajs.size(), weight);
    }

    m_q.wait_and_throw();
    // Perform reduction to find the minimum value and its index
    // Command scope
    m_q.submit([&](sycl::handler &h) {
      auto costs = m_devicePtrCosts;
      auto minCost = m_minCost;
      auto reduction = sycl::reduction(minCost, sycl::plus<LowestCost>());
      // Kernel scope
      h.parallel_for(sycl::range<1>(trajs.size()), reduction,
                     [=](sycl::id<1> idx, auto &minVal) {
                       minVal.combine(LowestCost(costs[idx], idx));
                     });
    });

    m_q.wait_and_throw();

    return {trajs.getIndex(m_minCost->sampleIndex), true, m_minCost->cost};
  } catch (const sycl::exception &e) {
    LOG_ERROR("Exception caught: ", e.what());
  }
  return TrajSearchResult();
}

/*std::vector<double>*/
/*CostEvaluator::pathCostFunc(const size_t trajs_size,*/
/*                            const Path::Path &reference_path,*/
/*                            const double cost_weight) {*/
/**/
/*  std::vector<double> costVec;*/
/*  costVec.reserve(trajs_size);*/
/*  for (size_t t = 0; t < trajs_size, ++t) {*/
/*    double total_cost = 0.0;*/
/*    double distError, dist;*/
/*    for (size_t i = 0; i < trajectories[t].path.x.size(); ++i) {*/
/*      // Set min distance between trajectory sample point i and the reference
 * to*/
/*      // infinity*/
/*      distError = std::numeric_limits<double>::max();*/
/*            // Get minimum distance to the reference*/
/*      for (size_t j = 0; j < reference_path.points.size(); ++j) {*/
/*        dist = Path::Path::distance(reference_path.points[j],*/
/*                                    trajectories[t].path.getIndex(i));*/
/*        if (dist < distError) {*/
/*          distError = dist;*/
/*        }*/
/*      }*/
/*      // Total min distance to each point*/
/*      total_cost += distError;*/
/*    }*/
/**/
/*    // end point distance*/
/*    double end_dist_error =
 * Path::Path::distance(trajectories[t].path.getEnd(),*/
/*                                                 reference_path.getEnd());*/
/*    // Divide by number of points to get average distance*/
/*    total_cost = total_cost / trajectories[t].path.x.size() +
 * end_dist_error;*/
/*  }*/
/*  return cost_array;*/
/*}*/
/**/
// Compute the cost of a trajectory based on distance to a given reference
// path
/*double CostEvaluator::goalCostFunc(const Trajectory2D &trajectory,*/
/*                                   const Path::Path &reference_path) {*/
/*    // end point distance*/
/*    return Path::Path::distance(trajectory.path.getEnd(),*/
/*                                reference_path.getEnd()) /*/
/*           reference_path.totalPathLength();*/
/*}*/

/*double CostEvaluator::obstaclesDistCostFunc(*/
/*    const Trajectory2D &trajectory,*/
/*    const std::vector<Path::Point> &obstaclePoints) {*/
/*    return trajectory.path.minDist2D(obstaclePoints);*/
/*}*/

// Compute the cost of trajectory based on smoothness in velocity commands
void CostEvaluator::smoothnessCostFunc(const size_t trajs_size,
                                       const double weight) {

  // -----------------------------------------------------
  //  Parallelize over trajectories and velocity indices.
  //  Each valid inner index (i >= 1 and i < numPointsPerTrajectory_) computes
  //  a cost contribution which is atomically added to the trajectory’s cost.
  // -----------------------------------------------------

  // command scope
  m_q.submit([&](sycl::handler &h) {
    // local copies of class members to be used inside the kernel
    auto velocitiesVx = m_devicePtrVelocitiesVy;
    auto velocitiesVy = m_devicePtrVelocitiesVy;
    auto velocitiesOmega = m_devicePtrVelocitiesOmega;
    auto costs = m_devicePtrCosts;
    double costWeight = weight;
    size_t velocitiesSize = numPointsPerTrajectory_ - 1;
    sycl::vec<double, 3> accLimits = {accLimits_[0], accLimits_[1],
                                      accLimits_[2]};
    sycl::accessor<double, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        trajCost(sycl::range<1>(velocitiesSize), h);
    auto global_size = sycl::range<1>(trajs_size * velocitiesSize);
    auto workgroup_size = sycl::range<1>(velocitiesSize);
    m_q.parallel_for(
        sycl::nd_range<1>(global_size, workgroup_size),
        [=](sycl::nd_item<1> item) {
          const int traj = item.get_group().get_group_id();
          const int vel_item = item.get_local_id();

          // Initialize local memory once per work-group in the first thread
          if (vel_item == 0) {
            for (size_t i = 0; i < velocitiesSize; ++i) {
              trajCost[i] = 0;
            }
          }
          // Synchronize to make sure initialization is complete
          item.barrier(sycl::access::fence_space::local_space);

          // Process only if i is valid (skip i==0 since we need a previous
          // sample)
          if (vel_item >= 1 && vel_item < velocitiesSize) {
            double cost_contrib = 0.0;
            // Get the velocity samples for the current trajectory
            if (accLimits[0] > 0) {
              double delta_vx = velocitiesVx[traj * velocitiesSize + vel_item] -
                                velocitiesVx[traj * velocitiesSize + (vel_item - 1)];
              cost_contrib += (delta_vx * delta_vx) / accLimits[0];
            }
            if (accLimits[1] > 0) {
              double delta_vy = velocitiesVy[traj * velocitiesSize + vel_item] -
                                velocitiesVy[traj * velocitiesSize + (vel_item - 1)];
              cost_contrib += (delta_vy * delta_vy) / accLimits[1];
            }
            if (accLimits[2] > 0) {
              double delta_omega =
                  velocitiesOmega[traj * velocitiesSize + vel_item] -
                  velocitiesOmega[traj * velocitiesSize + (vel_item - 1)];
              cost_contrib += (delta_omega * delta_omega) / accLimits[2];
            }

            // Each work-item performs an atomic update for its trajectory
            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                atomicLocal(trajCost[traj]);

            // atomically add cost_contrib to trajectory cost
            atomicLocal.fetch_add(1);

            // Synchronize again so that all atomic updates are finished before
            // further processing
            item.barrier(sycl::access::fence_space::local_space);

            // normalize all costs using the first thread of the group and add
            // them to global cost
            if (vel_item == 0) {
              trajCost[traj] =
                  costWeight * (trajCost[traj] / (3.0 * velocitiesSize));

              // Atomically add the computed contribution to the global cost for
              // this trajectory
              sycl::atomic_ref<double, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  atomic_cost(costs[traj]);
              atomic_cost.fetch_add(cost_contrib);
            }
          }
        });
  });
}

// Compute the cost of trajectory based on jerk in velocity commands
void CostEvaluator::jerkCostFunc(const size_t trajs_size, const double weight) {
  // -----------------------------------------------------
  //  Parallelize over trajectories and velocity indices.
  //  Each valid inner index (i >= 1 and i < numPointsPerTrajectory_) computes
  //  a cost contribution which is atomically added to the trajectory’s cost.
  // -----------------------------------------------------

  // command scope
  m_q.submit([&](sycl::handler &h) {
    // local copies of class members to be used inside the kernel
    auto velocitiesVx = m_devicePtrVelocitiesVy;
    auto velocitiesVy = m_devicePtrVelocitiesVy;
    auto velocitiesOmega = m_devicePtrVelocitiesOmega;
    auto costs = m_devicePtrCosts;
    double costWeight = weight;
    size_t velocitiesSize = numPointsPerTrajectory_ - 1;
    sycl::vec<double, 3> accLimits = {accLimits_[0], accLimits_[1],
                                      accLimits_[2]};
    sycl::accessor<double, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        trajCost(sycl::range<1>(velocitiesSize), h);
    auto global_size = sycl::range<1>(trajs_size * velocitiesSize);
    auto workgroup_size = sycl::range<1>(velocitiesSize);
    m_q.parallel_for(
        sycl::nd_range<1>(global_size, workgroup_size),
        [=](sycl::nd_item<1> item) {
          const int traj = item.get_group().get_group_id();
          const int vel_point = item.get_local_id();

          // Initialize local memory once per work-group in the first thread
          if (vel_point == 0) {
            for (size_t i = 0; i < velocitiesSize; ++i) {
              trajCost[i] = 0;
            }
          }
          // Synchronize to make sure initialization is complete
          item.barrier(sycl::access::fence_space::local_space);

          // Process only if i is valid (skip i==0 since we need a previous
          // sample)
          if (vel_point >= 2 && vel_point < velocitiesSize) {
            double cost_contrib = 0.0;
            // Get the velocity samples for the current trajectory.
            if (accLimits[0] > 0) {
              double delta_vx =
                  velocitiesVx[traj * velocitiesSize + vel_point] -
                  2 * velocitiesVx[traj * velocitiesSize + (vel_point - 1)] +
                  velocitiesVx[traj * velocitiesSize + (vel_point - 2)];
              cost_contrib += (delta_vx * delta_vx) / accLimits[0];
            }
            if (accLimits[1] > 0) {
              double delta_vy =
                  velocitiesVy[traj * velocitiesSize + vel_point] -
                  2 * velocitiesVy[traj * velocitiesSize + (vel_point - 1)] +
                  velocitiesVy[traj * velocitiesSize + (vel_point - 2)];
              cost_contrib += (delta_vy * delta_vy) / accLimits[1];
            }
            if (accLimits[2] > 0) {
              double delta_omega =
                  velocitiesOmega[traj * velocitiesSize + vel_point] -
                  2 * velocitiesOmega[traj * velocitiesSize + (vel_point - 1)] +
                  velocitiesOmega[traj * velocitiesSize + (vel_point - 2)];
              cost_contrib += (delta_omega * delta_omega) / accLimits[2];
            }

            // Each work-item performs an atomic update for its trajectory
            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                atomicLocal(trajCost[traj]);

            // atomically add cost_contrib to trajectory cost
            atomicLocal.fetch_add(1);

            // Synchronize again so that all atomic updates are finished before
            // further processing
            item.barrier(sycl::access::fence_space::local_space);

            // normalize all costs using the first thread of the group and add
            // them to global cost
            if (vel_point == 0) {
              trajCost[traj] =
                  costWeight * (trajCost[traj] / (3.0 * velocitiesSize));

              // Atomically add the computed contribution to the global cost for
              // this trajectory
              sycl::atomic_ref<double, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  atomic_cost(costs[traj]);
              atomic_cost.fetch_add(cost_contrib);
            }
          }
        });
  });
}

}; // namespace Control
} // namespace Kompass
#endif // !GPU
