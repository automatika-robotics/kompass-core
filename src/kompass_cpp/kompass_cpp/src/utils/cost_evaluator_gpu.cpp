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
                             size_t maxAngularSamples, size_t maxRefPathSize) {

  this->costWeights = costsWeights;

  numPointsPerTrajectory_ = getNumPointsPerTrajectory(timeStep, timeHorizon);
  numTrajectories_ =
      getNumTrajectories(ctrType, maxLinearSamples, maxAngularSamples);

  accLimits_ = {static_cast<float>(ctrLimits.velXParams.maxAcceleration),
                static_cast<float>(ctrLimits.velYParams.maxAcceleration),
                static_cast<float>(ctrLimits.omegaParams.maxAcceleration)};
  maxRefPathSize_ = maxRefPathSize;
  initializeGPUMemory();
}

CostEvaluator::CostEvaluator(TrajectoryCostsWeights costsWeights,
                             const std::array<float, 3> &sensor_position_body,
                             const std::array<float, 4> &sensor_rotation_body,
                             ControlType controlType,
                             ControlLimitsParams ctrLimits, double timeStep,
                             double timeHorizon, size_t maxLinearSamples,
                             size_t maxAngularSamples, size_t maxRefPathSize) {

  sensor_tf_body_ =
      getTransformation(Eigen::Quaternionf(sensor_rotation_body.data()),
                        Eigen::Vector3f(sensor_position_body.data()));
  this->costWeights = costsWeights;

  accLimits_ = {static_cast<float>(ctrLimits.velXParams.maxAcceleration),
                static_cast<float>(ctrLimits.velYParams.maxAcceleration),
                static_cast<float>(ctrLimits.omegaParams.maxAcceleration)};

  numPointsPerTrajectory_ = getNumPointsPerTrajectory(timeStep, timeHorizon);
  numTrajectories_ =
      getNumTrajectories(ctrType, maxLinearSamples, maxAngularSamples);
  maxRefPathSize_ = maxRefPathSize;
  initializeGPUMemory();
}

void CostEvaluator::initializeGPUMemory() {
  m_q =
      sycl::queue{sycl::default_selector_v, sycl::property::queue::in_order{}};
  auto dev = m_q.get_device();
  LOG_INFO("Running on :", dev.get_info<sycl::info::device::name>());
  if (dev.has(sycl::aspect::atomic64)) {
    LOG_INFO("Device supports 64-bit atomic operations.\n");
  } else {
    LOG_INFO("Device does NOT support 64-bit atomic operations.\n");
  }

  // Data memory allocation
  m_devicePtrPathsX = sycl::malloc_device<float>(
      numTrajectories_ * numPointsPerTrajectory_, m_q);
  m_devicePtrPathsY = sycl::malloc_device<float>(
      numTrajectories_ * numPointsPerTrajectory_, m_q);
  m_devicePtrVelocitiesVx = sycl::malloc_device<float>(
      numTrajectories_ * (numPointsPerTrajectory_ - 1), m_q);
  m_devicePtrVelocitiesVy = sycl::malloc_device<float>(
      numTrajectories_ * (numPointsPerTrajectory_ - 1), m_q);
  m_devicePtrVelocitiesOmega = sycl::malloc_device<float>(
      numTrajectories_ * (numPointsPerTrajectory_ - 1), m_q);
  m_devicePtrCosts = sycl::malloc_device<float>(numTrajectories_, m_q);

  // Cost specific memory allocation
  if (costWeights.getParameter<double>("reference_path_distance_weight") >
      0.0) {
    m_devicePtrReferencePathX =
        sycl::malloc_device<float>(maxRefPathSize_, m_q);
    m_devicePtrReferencePathY =
        sycl::malloc_device<float>(maxRefPathSize_, m_q);
  };
  if (costWeights.getParameter<double>("obstacles_distance_weight") > 0.0 ||
      customTrajCostsPtrs_.size() > 0) {
    m_devicePtrTempCosts = sycl::malloc_shared<float>(numTrajectories_, m_q);
  }

  // Result struct memory allocation
  m_minCost = sycl::malloc_shared<LowestCost>(1, m_q);
}

void CostEvaluator::updateCostWeights(TrajectoryCostsWeights costsWeights) {
  this->costWeights = costsWeights;
  // Do Cost specific memory allocation if not already done
  if (costWeights.getParameter<double>("reference_path_distance_weight") >
          0.0 &&
      !m_devicePtrReferencePathX && !m_devicePtrReferencePathY) {
    m_devicePtrReferencePathX =
        sycl::malloc_device<float>(maxRefPathSize_, m_q);
    m_devicePtrReferencePathY =
        sycl::malloc_device<float>(maxRefPathSize_, m_q);
  };
  if ((costWeights.getParameter<double>("obstacles_distance_weight") > 0.0 ||
       customTrajCostsPtrs_.size() > 0) &&
      !m_devicePtrTempCosts) {
    m_devicePtrTempCosts = sycl::malloc_shared<float>(numTrajectories_, m_q);
  }
};

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
  if (m_devicePtrReferencePathX) {
    sycl::free(m_devicePtrReferencePathX, m_q);
  }
  if (m_devicePtrReferencePathY) {
    sycl::free(m_devicePtrReferencePathY, m_q);
  }
  if (m_devicePtrTempCosts) {
    sycl::free(m_devicePtrTempCosts, m_q);
  }
  if (m_minCost) {
    if (m_minCost) {
      sycl::free(m_minCost, m_q);
    }
  }
};

TrajSearchResult CostEvaluator::getMinTrajectoryCost(
    const TrajectorySamples2D &trajs, const Path::Path &reference_path,
    const Path::Path &tracked_segment, const size_t closest_segment_index) {

  try {
    double weight;
    float ref_path_length;
    size_t trajs_size = trajs.size();

    m_q.fill(m_devicePtrCosts, 0.0, trajs.size()).wait();

    m_q.memcpy(m_devicePtrPathsX, trajs.paths.x.data(),
               sizeof(float) * trajs_size * numPointsPerTrajectory_);
    m_q.memcpy(m_devicePtrPathsY, trajs.paths.y.data(),
               sizeof(float) * trajs_size * numPointsPerTrajectory_);
    m_q.memcpy(m_devicePtrVelocitiesVx, trajs.velocities.vx.data(),
               sizeof(float) * trajs_size * (numPointsPerTrajectory_ - 1))
        .wait();
    m_q.memcpy(m_devicePtrVelocitiesVy, trajs.velocities.vy.data(),
               sizeof(float) * trajs_size * (numPointsPerTrajectory_ - 1))
        .wait();
    m_q.memcpy(m_devicePtrVelocitiesOmega, trajs.velocities.omega.data(),
               sizeof(float) * trajs_size * (numPointsPerTrajectory_ - 1))
        .wait();
    *m_minCost = LowestCost();

    // wait for all data to be transferred
    m_q.wait();

    if ((costWeights.getParameter<double>("reference_path_distance_weight") >
             0.0 ||
         costWeights.getParameter<double>("goal_distance_weight") > 0.0) &&
        (ref_path_length = reference_path.totalPathLength()) > 0.0) {
      if ((weight = costWeights.getParameter<double>("goal_distance_weight")) >
          0.0) {
        goalCostFunc(trajs_size, reference_path.getEnd(), ref_path_length,
                     weight);
      }
      if ((weight = costWeights.getParameter<double>(
               "reference_path_distance_weight")) > 0.0) {
        m_q.memcpy(m_devicePtrReferencePathX, reference_path.getX().data(),
                   sizeof(float) * reference_path.points.size())
            .wait();
        m_q.memcpy(m_devicePtrReferencePathY, reference_path.getY().data(),
                   sizeof(float) * reference_path.points.size())
            .wait();
        pathCostFunc(trajs_size, reference_path.points.size(), weight);
      }
    }
    if ((weight = costWeights.getParameter<double>("smoothness_weight")) >
        0.0) {
      std::cout << "Going in \n";
      smoothnessCostFunc(trajs_size, weight);
    }
    if ((weight = costWeights.getParameter<double>("jerk_weight")) > 0.0) {
      jerkCostFunc(trajs_size, weight);
    }

    // wait for all costs to be calculated
    m_q.wait();

    // calculate costs on the CPU
    if (((weight = costWeights.getParameter<double>(
              "obstacles_distance_weight")) > 0.0) ||
        customTrajCostsPtrs_.size() > 0) {

      m_q.fill(m_devicePtrTempCosts, 0.0, trajs.size());
      float total_cost;
      size_t idx = 0;
      for (const auto traj : trajs) {
        if (weight > 0.0 && obstaclePoints.size() > 0) {
          total_cost += weight * obstaclesDistCostFunc(traj, obstaclePoints);
        }
        for (const auto &custom_cost : customTrajCostsPtrs_) {
          // custom cost functions takes in the trajectory and the reference
          // path
          total_cost += custom_cost->weight *
                        custom_cost->evaluator_(traj, reference_path);
        }
        // add cost to the shared temp cost array
        m_devicePtrTempCosts[idx] = total_cost;
        idx += 1;
      }

      // Add temp costs to global costs
      // Command scope
      m_q.submit([&](sycl::handler &h) {
           auto tempCosts = m_devicePtrTempCosts;
           auto costs = m_devicePtrCosts;
           // Kernel scope
           h.parallel_for(sycl::range<1>(trajs.size()),
                          [=](sycl::id<1> id) { costs[id] += tempCosts[id]; });
         })
          .wait();
    }

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
       })
        .wait();

    return {trajs.getIndex(m_minCost->sampleIndex), true, m_minCost->cost};
  } catch (const sycl::exception &e) {
    LOG_ERROR("Exception caught: ", e.what());
  }
  return TrajSearchResult();
}

// Compute the cost of a trajectory based on distance to a given reference
// path
void CostEvaluator::pathCostFunc(const size_t trajs_size,
                                 const size_t ref_path_size,
                                 const double cost_weight) {

  // -----------------------------------------------------
  //  Parallelize over trajectories and path indices.
  //  Calculate distance of each point with each point in reference path.
  //  Atomically add lowest distance to the trajectory’s cost.
  // -----------------------------------------------------

  // command scope
  m_q.submit([&](sycl::handler &h) {
       // local copies of class members to be used inside the kernel
       auto X = m_devicePtrPathsX;
       auto Y = m_devicePtrPathsY;
       auto ref_X = m_devicePtrReferencePathX;
       auto ref_Y = m_devicePtrReferencePathY;
       auto costs = m_devicePtrCosts;
       const float costWeight = cost_weight;
       const size_t trajsSize = trajs_size;
       const size_t pathSize = numPointsPerTrajectory_;
       const size_t refPathSize = ref_path_size;
       // local memory for storing lowest per point cost
       sycl::local_accessor<float, 1> pointCost(sycl::range<1>(pathSize), h);
       // local memory for storing per trajectory average cost
       sycl::local_accessor<float, 1> trajCost(sycl::range<1>(trajs_size), h);
       auto global_size = sycl::range<2>(trajs_size * pathSize, 1);
       auto workgroup_size = sycl::range<2>(pathSize, refPathSize);
       // Kernel scope
       h.parallel_for(
           sycl::nd_range<2>(global_size, workgroup_size),
           [=](sycl::nd_item<2> item) {
             const size_t traj = item.get_group().get_group_id()[0];
             const size_t path_index = item.get_local_id()[0];
             const size_t ref_path_index = item.get_local_id()[1];

             // Initialize local memory once per work-group in the first thread
             if (path_index == 0 && ref_path_index == 0) {
               for (size_t i = 0; i < pathSize; ++i) {
                 pointCost[i] = 0;
               }
               for (size_t i = 0; i < trajsSize; ++i) {
                 trajCost[i] = 0;
               }
             }

             // Synchronize to make sure initialization is complete
             item.barrier(sycl::access::fence_space::local_space);

             sycl::vec<float, 2> point = {X[traj * pathSize + path_index],
                                          Y[traj * pathSize + path_index]};
             sycl::vec<float, 2> ref_point = {ref_X[ref_path_index],
                                              ref_Y[ref_path_index]};

             float distance = sycl::distance(point, ref_point);

             // Each work-item performs an atomic update for its path point
             sycl::atomic_ref<float, sycl::memory_order::relaxed,
                              sycl::memory_scope::work_group,
                              sycl::access::address_space::local_space>
                 atomicMin(pointCost[path_index]);

             // atomically get minimum cost per point
             atomicMin.fetch_min(distance);

             // Synchronize again so that all atomic updates are finished before
             // further processing
             item.barrier(sycl::access::fence_space::local_space);

             // Atomically add the computed min costs to the local cost for
             // this trajectory, only once per ref_path_index (last thread)
             if (ref_path_index == refPathSize - 1) {
               // Atomically add the computed min costs to the local cost for
               // this trajectory
               sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::local_space>
                   atomic_cost(trajCost[traj]);
               atomic_cost.fetch_add(pointCost[path_index]);

               // Synchronize again so that all atomic updates are finished
               // before further processing
               item.barrier(sycl::access::fence_space::local_space);

               // normalize the trajectory cost and add end point distance to it
               if (ref_path_index == refPathSize - 1 &&
                   path_index == pathSize - 1) {

                 sycl::vec<float, 2> last_path_point = {
                     X[traj * pathSize + path_index],
                     Y[traj * pathSize + path_index]};
                 sycl::vec<float, 2> last_ref_point = {ref_X[ref_path_index],
                                                       ref_Y[ref_path_index]};

                 // get distance between two last points
                 float end_point_distance =
                     sycl::distance(last_path_point, last_ref_point);
                 trajCost[traj] = costWeight * (trajCost[traj] / pathSize +
                                                end_point_distance);

                 // Atomically add the computed contribution to the global cost
                 // for this trajectory
                 sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                     atomic_cost(costs[traj]);
                 atomic_cost.fetch_add(trajCost[traj]);
               }
             }
           });
     })
      .wait();
}

// Compute the cost of trajectory based on distance to the goal point
void CostEvaluator::goalCostFunc(const size_t trajs_size,
                                 const Path::Point &last_ref_point,
                                 const float path_length,
                                 const double cost_weight) {
  // -----------------------------------------------------
  //  Parallelize over trajectories.
  //  Calculate distance of the last point in trajectory and reference path.
  //  Atomically add to global trajectory’s cost.
  // -----------------------------------------------------

  // command scope
  m_q.submit([&](sycl::handler &h) {
       // local copies of class members to be used inside the kernel
       auto X = m_devicePtrPathsX;
       auto Y = m_devicePtrPathsY;
       auto costs = m_devicePtrCosts;
       const float costWeight = cost_weight;
       const size_t pathSize = numPointsPerTrajectory_;
       const float pathLength = path_length;
       auto global_size = sycl::range<1>(trajs_size);
       // Last point of reference path
       sycl::vec<float, 2> lastRefPoint = {last_ref_point.x(),
                                           last_ref_point.y()};
       // Kernel scope
       h.parallel_for(sycl::range<1>(global_size), [=](sycl::id<1> id) {
         // get last point of the trajectory path
         sycl::vec<float, 2> last_path_point = {
             X[id * pathSize + pathSize - 1], Y[id * pathSize + pathSize - 1]};

         // end point distance normalized by path length
         float distance =
             sycl::distance(last_path_point, lastRefPoint) / pathLength;

         // Atomically add the computed contribution to the global cost
         // for this trajectory
         sycl::atomic_ref<float, sycl::memory_order::relaxed,
                          sycl::memory_scope::device,
                          sycl::access::address_space::global_space>
             atomic_cost(costs[id]);
         atomic_cost.fetch_add(costWeight * distance);
       });
     })
      .wait();
}

// Compute the cost of trajectory based on smoothness in velocity commands
void CostEvaluator::smoothnessCostFunc(const size_t trajs_size,
                                       const double cost_weight) {
  // -----------------------------------------------------
  //  Parallelize over trajectories and velocity indices.
  //  Each valid inner index (i >= 1 and i < numPointsPerTrajectory_)
  //  computes a cost contribution which is atomically added to the
  //  trajectory’s cost.
  // -----------------------------------------------------

  // command scope
  m_q.submit([&](sycl::handler &h) {
       // local copies of class members to be used inside the kernel
       auto velocitiesVx = m_devicePtrVelocitiesVx;
       auto velocitiesVy = m_devicePtrVelocitiesVy;
       auto velocitiesOmega = m_devicePtrVelocitiesOmega;
       auto costs = m_devicePtrCosts;
       const float costWeight = cost_weight;
       const size_t trajsSize = trajs_size;
       const size_t velocitiesSize = numPointsPerTrajectory_ - 1;
       const sycl::vec<float, 3> accLimits = {accLimits_[0], accLimits_[1],
                                              accLimits_[2]};
       sycl::local_accessor<float, 1> trajCost(sycl::range<1>(trajs_size), h);
       auto global_size = sycl::range<1>(trajs_size * velocitiesSize);
       auto workgroup_size = sycl::range<1>(velocitiesSize);
       h.parallel_for(
           sycl::nd_range<1>(global_size, workgroup_size),
           [=](sycl::nd_item<1> item) {
             const size_t traj = item.get_group().get_group_id();
             const size_t vel_item = item.get_local_id();

             // Initialize local memory once per work-group in the first thread
             if (vel_item == 0) {
               for (size_t i = 0; i < trajsSize; ++i) {
                 trajCost[i] = 0;
               }
             }
             // Synchronize to make sure initialization is complete
             item.barrier(sycl::access::fence_space::local_space);

             // Process only if i is valid (skip i==0 since we need a previous
             // sample)
             if (vel_item >= 1 && vel_item < velocitiesSize) {
               float cost_contrib = 0.0;
               // Get the cost contribution for each point
               if (accLimits[0] > 0) {
                 float delta_vx =
                     velocitiesVx[traj * velocitiesSize + vel_item] -
                     velocitiesVx[traj * velocitiesSize + (vel_item - 1)];
                 cost_contrib += (delta_vx * delta_vx) / accLimits[0];
               }
               if (accLimits[1] > 0) {
                 float delta_vy =
                     velocitiesVy[traj * velocitiesSize + vel_item] -
                     velocitiesVy[traj * velocitiesSize + (vel_item - 1)];
                 cost_contrib += (delta_vy * delta_vy) / accLimits[1];
               }
               if (accLimits[2] > 0) {
                 float delta_omega =
                     velocitiesOmega[traj * velocitiesSize + vel_item] -
                     velocitiesOmega[traj * velocitiesSize + (vel_item - 1)];
                 cost_contrib += (delta_omega * delta_omega) / accLimits[2];
               }

               // Each work-item performs an atomic update for its trajectory
               sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                sycl::memory_scope::work_group,
                                sycl::access::address_space::local_space>
                   atomicLocal(trajCost[traj]);

               // atomically add cost_contrib to trajectory cost
               atomicLocal.fetch_add(cost_contrib);
             }

             // Synchronize again so that all atomic updates are finished
             // before further processing
             item.barrier(sycl::access::fence_space::local_space);

             // normalize all costs using the first thread of the group and
             // add them to global cost
             if (vel_item == 0) {
               trajCost[traj] =
                   costWeight * (trajCost[traj] / (3.0 * velocitiesSize));

               // Atomically add the computed contribution to the global cost
               // for this trajectory
               sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                   atomic_cost(costs[traj]);
               atomic_cost.fetch_add(trajCost[traj]);
             }
           });
     })
      .wait();
}

// Compute the cost of trajectory based on jerk in velocity commands
void CostEvaluator::jerkCostFunc(const size_t trajs_size,
                                 const double cost_weight) {
  // -----------------------------------------------------
  //  Parallelize over trajectories and velocity indices.
  //  Each valid inner index (i >= 1 and i < numPointsPerTrajectory_)
  //  computes a cost contribution which is atomically added to the
  //  trajectory’s cost.
  // -----------------------------------------------------

  // command scope
  m_q.submit([&](sycl::handler &h) {
       // local copies of class members to be used inside the kernel
       auto velocitiesVx = m_devicePtrVelocitiesVx;
       auto velocitiesVy = m_devicePtrVelocitiesVy;
       auto velocitiesOmega = m_devicePtrVelocitiesOmega;
       auto costs = m_devicePtrCosts;
       const float costWeight = cost_weight;
       const size_t trajsSize = trajs_size;
       const size_t velocitiesSize = numPointsPerTrajectory_ - 1;
       const sycl::vec<float, 3> accLimits = {accLimits_[0], accLimits_[1],
                                              accLimits_[2]};
       sycl::local_accessor<float, 1> trajCost(sycl::range<1>(trajs_size), h);
       auto global_size = sycl::range<1>(trajs_size * velocitiesSize);
       auto workgroup_size = sycl::range<1>(velocitiesSize);
       h.parallel_for(
           sycl::nd_range<1>(global_size, workgroup_size),
           [=](sycl::nd_item<1> item) {
             const size_t traj = item.get_group().get_group_id();
             const size_t vel_point = item.get_local_id();

             // Initialize local memory once per work-group in the first thread
             if (vel_point == 0) {
               for (size_t i = 0; i < trajsSize; ++i) {
                 trajCost[i] = 0;
               }
             }
             // Synchronize to make sure initialization is complete
             item.barrier(sycl::access::fence_space::local_space);

             // Process only if i is valid (skip i==0 since we need a previous
             // sample)
             if (vel_point >= 2 && vel_point < velocitiesSize) {
               float cost_contrib = 0.0;
               // Get the cost contribution for each point
               if (accLimits[0] > 0) {
                 float delta_vx =
                     velocitiesVx[traj * velocitiesSize + vel_point] -
                     2 * velocitiesVx[traj * velocitiesSize + (vel_point - 1)] +
                     velocitiesVx[traj * velocitiesSize + (vel_point - 2)];
                 cost_contrib += (delta_vx * delta_vx) / accLimits[0];
               }
               if (accLimits[1] > 0) {
                 float delta_vy =
                     velocitiesVy[traj * velocitiesSize + vel_point] -
                     2 * velocitiesVy[traj * velocitiesSize + (vel_point - 1)] +
                     velocitiesVy[traj * velocitiesSize + (vel_point - 2)];
                 cost_contrib += (delta_vy * delta_vy) / accLimits[1];
               }
               if (accLimits[2] > 0) {
                 float delta_omega =
                     velocitiesOmega[traj * velocitiesSize + vel_point] -
                     2 * velocitiesOmega[traj * velocitiesSize +
                                         (vel_point - 1)] +
                     velocitiesOmega[traj * velocitiesSize + (vel_point - 2)];
                 cost_contrib += (delta_omega * delta_omega) / accLimits[2];
               }

               // Each work-item performs an atomic update for its trajectory
               sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                sycl::memory_scope::work_group,
                                sycl::access::address_space::local_space>
                   atomicLocal(trajCost[traj]);

               // atomically add cost_contrib to trajectory cost
               atomicLocal.fetch_add(cost_contrib);
             }

             // Synchronize again so that all atomic updates are finished
             // before further processing
             item.barrier(sycl::access::fence_space::local_space);

             // normalize all costs using the first thread of the group and
             // add them to global cost
             if (vel_point == 0) {
               trajCost[traj] =
                   costWeight * (trajCost[traj] / (3.0 * velocitiesSize));

               // Atomically add the computed trajectory cost to the global
               // cost for this trajectory
               sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                   atomic_cost(costs[traj]);
               atomic_cost.fetch_add(trajCost[traj]);
             }
           });
     })
      .wait();
}

// Calculate obstacle distance cost per trajectory (CPU)
float CostEvaluator::obstaclesDistCostFunc(
    const Trajectory2D &trajectory,
    const std::vector<Path::Point> &obstaclePoints) {
  return trajectory.path.minDist2D(obstaclePoints);
}
}; // namespace Control
} // namespace Kompass
#endif // !GPU
