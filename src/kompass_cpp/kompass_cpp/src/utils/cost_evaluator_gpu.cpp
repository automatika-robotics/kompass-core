#ifdef GPU
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "utils/cost_evaluator.h"
#include "utils/logger.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <sycl/sycl.hpp>
#include <vector>

namespace Kompass {

namespace Control {
CostEvaluator::CostEvaluator(TrajectoryCostsWeights &costsWeights,
                             ControlLimitsParams ctrLimits,
                             size_t maxNumTrajectories,
                             size_t numPointsPerTrajectory,
                             size_t maxRefPathSegmentSize) {

  this->costWeights = std::make_unique<TrajectoryCostsWeights>(costsWeights);

  numTrajectories_ = maxNumTrajectories;
  numPointsPerTrajectory_ = numPointsPerTrajectory;

  accLimits_ = {static_cast<float>(ctrLimits.velXParams.maxAcceleration),
                static_cast<float>(ctrLimits.velYParams.maxAcceleration),
                static_cast<float>(ctrLimits.omegaParams.maxAcceleration)};
  maxRefPathSegmentSize_ = maxRefPathSegmentSize;
  initializeGPUMemory();
}

CostEvaluator::CostEvaluator(TrajectoryCostsWeights &costsWeights,
                             const Eigen::Vector3f &sensor_position_body,
                             const Eigen::Quaternionf &sensor_rotation_body,
                             ControlLimitsParams ctrLimits,
                             size_t maxNumTrajectories,
                             size_t numPointsPerTrajectory,
                             size_t maxRefPathSegmentSize) {

  sensor_tf_body_ =
      getTransformation(sensor_rotation_body, sensor_position_body);
  this->costWeights = std::make_unique<TrajectoryCostsWeights>(costsWeights);

  accLimits_ = {static_cast<float>(ctrLimits.velXParams.maxAcceleration),
                static_cast<float>(ctrLimits.velYParams.maxAcceleration),
                static_cast<float>(ctrLimits.omegaParams.maxAcceleration)};

  numTrajectories_ = maxNumTrajectories;
  numPointsPerTrajectory_ = numPointsPerTrajectory;
  maxRefPathSegmentSize_ = maxRefPathSegmentSize;
  initializeGPUMemory();
}

void CostEvaluator::initializeGPUMemory() {
  m_q = sycl::queue{sycl::default_selector_v};
  auto dev = m_q.get_device();
  LOG_INFO("Running on :", dev.get_info<sycl::info::device::name>());

  if (!dev.has(sycl::aspect::atomic64)) {
    LOG_WARNING("Device does NOT support 64-bit atomic operations. Some kernel "
                "calculations might be slow.\n");
  }

  // Query maximum work-group size
  max_wg_size_ = dev.get_info<sycl::info::device::max_work_group_size>();
  if (numPointsPerTrajectory_ > max_wg_size_) {
    LOG_WARNING("Number of points per sample trajectory should be less than:",
                max_wg_size_,
                "for your device. Please try to modify the control time step "
                "and prediction horizon such that "
                "prediction_horizon/control_time_step is less than",
                max_wg_size_);
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
  if (costWeights->getParameter<double>("reference_path_distance_weight") >
      0.0) {
    m_devicePtrTrackedSegmentX =
        sycl::malloc_device<float>(maxRefPathSegmentSize_, m_q);
    m_devicePtrTrackedSegmentY =
        sycl::malloc_device<float>(maxRefPathSegmentSize_, m_q);
  };
  if (costWeights->getParameter<double>("obstacles_distance_weight") > 0.0) {
    m_devicePtrObstaclesX = sycl::malloc_device<float>(max_wg_size_, m_q);
    m_devicePtrObstaclesY = sycl::malloc_device<float>(max_wg_size_, m_q);
  }
  if (customTrajCostsPtrs_.size() > 0) {
    m_devicePtrTempCosts = sycl::malloc_shared<float>(numTrajectories_, m_q);
  }

  // Result struct memory allocation and init
  m_minCost = sycl::malloc_shared<LowestCost>(1, m_q);
  *m_minCost = LowestCost();
}

void CostEvaluator::updateCostWeights(TrajectoryCostsWeights &newCostsWeights) {
  this->costWeights = std::make_unique<TrajectoryCostsWeights>(newCostsWeights);
  // Do Cost specific memory allocation if not already done
  if (costWeights->getParameter<double>("reference_path_distance_weight") >
          0.0 &&
      !m_devicePtrTrackedSegmentX && !m_devicePtrTrackedSegmentY) {
    if (m_devicePtrTrackedSegmentX) {
      sycl::free(m_devicePtrTrackedSegmentX, m_q);
    }
    if (m_devicePtrTrackedSegmentY) {
      sycl::free(m_devicePtrTrackedSegmentY, m_q);
    }
    m_devicePtrTrackedSegmentX =
        sycl::malloc_device<float>(maxRefPathSegmentSize_, m_q);
    m_devicePtrTrackedSegmentY =
        sycl::malloc_device<float>(maxRefPathSegmentSize_, m_q);
  };
  if (costWeights->getParameter<double>("obstacles_distance_weight") > 0.0 &&
      !m_devicePtrObstaclesX && !m_devicePtrObstaclesY) {
    if (m_devicePtrObstaclesX) {
      sycl::free(m_devicePtrObstaclesX, m_q);
    }
    if (m_devicePtrObstaclesY) {
      sycl::free(m_devicePtrObstaclesY, m_q);
    }
    m_devicePtrObstaclesX = sycl::malloc_device<float>(max_wg_size_, m_q);
    m_devicePtrObstaclesY = sycl::malloc_device<float>(max_wg_size_, m_q);
  }
  if (customTrajCostsPtrs_.size() > 0 && !m_devicePtrTempCosts) {
    if (m_devicePtrTempCosts) {
      sycl::free(m_devicePtrTempCosts, m_q);
    }
    // Allocate shared memory for temporary costs
    m_devicePtrTempCosts = sycl::malloc_shared<float>(numTrajectories_, m_q);
  }
};

CostEvaluator::~CostEvaluator() {

  // Clear custom cost pointers
  customTrajCostsPtrs_.clear();

  // unallocate device memory
  if (m_devicePtrCosts) {
    sycl::free(m_devicePtrCosts, m_q);
  }
  if (m_minCost) {
    sycl::free(m_minCost, m_q);
  }
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
  if (m_devicePtrTrackedSegmentX) {
    sycl::free(m_devicePtrTrackedSegmentX, m_q);
  }
  if (m_devicePtrTrackedSegmentY) {
    sycl::free(m_devicePtrTrackedSegmentY, m_q);
  }
  if (m_devicePtrObstaclesX) {
    sycl::free(m_devicePtrObstaclesX, m_q);
  }
  if (m_devicePtrObstaclesY) {
    sycl::free(m_devicePtrObstaclesY, m_q);
  }
  if (m_devicePtrTempCosts) {
    sycl::free(m_devicePtrTempCosts, m_q);
  }
};

TrajSearchResult CostEvaluator::getMinTrajectoryCost(
    const std::unique_ptr<TrajectorySamples2D> &trajs,
    const Path::Path *reference_path, const Path::Path &tracked_segment) {

  try {
    double weight;
    float ref_path_length;
    size_t obs_size;
    size_t trajs_size = trajs->size();
    std::vector<sycl::event> events;

    // TODO: Investigate error in AdaptiveCPP JIT compilation for filling
    // float memory without wait
    m_q.fill(m_devicePtrCosts, 0.0f, trajs_size).wait();

    m_q.memcpy(m_devicePtrPathsX, trajs->paths.x.data(),
               sizeof(float) * trajs_size * numPointsPerTrajectory_);
    m_q.memcpy(m_devicePtrPathsY, trajs->paths.y.data(),
               sizeof(float) * trajs_size * numPointsPerTrajectory_);
    m_q.memcpy(m_devicePtrVelocitiesVx, trajs->velocities.vx.data(),
               sizeof(float) * trajs_size * (numPointsPerTrajectory_ - 1));
    m_q.memcpy(m_devicePtrVelocitiesVy, trajs->velocities.vy.data(),
               sizeof(float) * trajs_size * (numPointsPerTrajectory_ - 1));
    m_q.memcpy(m_devicePtrVelocitiesOmega, trajs->velocities.omega.data(),
               sizeof(float) * trajs_size * (numPointsPerTrajectory_ - 1));

    m_minCost->cost = DEFAULT_MIN_DIST;
    m_minCost->sampleIndex = 0;

    // wait for all data to be transferred
    m_q.wait();

    if ((costWeights->getParameter<double>("reference_path_distance_weight") >
             0.0 ||
         costWeights->getParameter<double>("goal_distance_weight") > 0.0) &&
        (ref_path_length = reference_path->totalPathLength()) > 0.0) {
      if ((weight = costWeights->getParameter<double>("goal_distance_weight")) >
          0.0) {
        auto last_point = reference_path->getEnd();
        m_deviceRefPathEnd =
            sycl::vec(last_point.x(), last_point.y(), last_point.z());
        events.push_back(goalCostFunc(trajs_size, ref_path_length, weight));
      }
      if ((weight = costWeights->getParameter<double>(
               "reference_path_distance_weight")) > 0.0) {
        size_t tracked_segment_size = tracked_segment.getSize();
        m_q.memcpy(m_devicePtrTrackedSegmentX, tracked_segment.getX().data(),
                   sizeof(float) * tracked_segment_size)
            .wait();
        m_q.memcpy(m_devicePtrTrackedSegmentY, tracked_segment.getY().data(),
                   sizeof(float) * tracked_segment_size)
            .wait();
        events.push_back(pathCostFunc(trajs_size, tracked_segment_size,
                                      tracked_segment.totalPathLength(),
                                      weight));
      }
    }
    if ((weight = costWeights->getParameter<double>("smoothness_weight")) >
        0.0) {
      events.push_back(smoothnessCostFunc(trajs_size, weight));
    }
    if ((weight = costWeights->getParameter<double>("jerk_weight")) > 0.0) {
      events.push_back(jerkCostFunc(trajs_size, weight));
    }
    if ((weight = costWeights->getParameter<double>(
             "obstacles_distance_weight")) > 0.0 &&
        (obs_size = obstaclePointsX.size()) > 0) {
      if (obs_size > max_wg_size_) {
        LOG_WARNING("The number of obstacles registered(", obs_size,
                    ") are more than the maximum workgroup size(", max_wg_size_,
                    "). Some obstacles will be dropped.");
        obs_size = max_wg_size_;
      }
      m_q.memcpy(m_devicePtrObstaclesX, obstaclePointsX.data(),
                 sizeof(float) * obs_size)
          .wait();
      m_q.memcpy(m_devicePtrObstaclesY, obstaclePointsY.data(),
                 sizeof(float) * obs_size)
          .wait();
      events.push_back(obstaclesDistCostFunc(trajs_size, weight));
    }

    // calculate costs on the CPU
    if (customTrajCostsPtrs_.size() > 0) {
      size_t idx = 0;
      for (const auto traj : *trajs) {
        m_devicePtrTempCosts[idx] = 0.0;
        for (const auto &custom_cost : customTrajCostsPtrs_) {
          // custom cost functions takes in the trajectory and the reference
          // path
          m_devicePtrTempCosts[idx] +=
              custom_cost->weight *
              custom_cost->evaluator_(traj, *reference_path);
        }
        idx += 1;
      }

      // Add temp costs to global costs
      // Command scope
      m_q.submit([&](sycl::handler &h) {
           h.depends_on(events);
           auto tempCosts = m_devicePtrTempCosts;
           auto costs = m_devicePtrCosts;
           // Kernel scope
           h.parallel_for<class customCostAdditionKernel>(
               sycl::range<1>(trajs_size),
               [=](sycl::id<1> id) { costs[id] += tempCosts[id]; });
         })
          .wait();
    }

    // Perform reduction to find the minimum value and its index
    // Command scope
    m_q.submit([&](sycl::handler &h) {
         h.depends_on(events);
         auto costs = m_devicePtrCosts;
         auto reduction = sycl::reduction(m_minCost, sycl::plus<LowestCost>());
         // Kernel scope
         h.parallel_for<class minimumCostReduction>(
             sycl::range<1>(trajs_size), reduction,
             [=](sycl::id<1> idx, auto &minVal) {
               minVal.combine(LowestCost(costs[idx], idx));
             });
       })
        .wait();

    return {trajs->getIndex(m_minCost->sampleIndex), true, m_minCost->cost};
  } catch (const sycl::exception &e) {
    LOG_ERROR("Exception caught: ", e.what());
    throw;
  }
  return TrajSearchResult();
}

// Compute the cost of a trajectory based on distance to a given reference
// path
sycl::event CostEvaluator::pathCostFunc(const size_t trajs_size,
                                        const size_t tracked_segment_size,
                                        const float tracked_segment_length,
                                        const double cost_weight) {

  // -----------------------------------------------------
  //  Parallelize over trajectories and path indices.
  //  Calculate distance of each point with each point in reference path.
  //  Atomically add lowest distance to the trajectory’s cost.
  // -----------------------------------------------------

  // command scope
  return m_q.submit([&](sycl::handler &h) {
    // local copies of class members to be used inside the kernel
    auto X = m_devicePtrPathsX;
    auto Y = m_devicePtrPathsY;
    auto tracked_X = m_devicePtrTrackedSegmentX;
    auto tracked_Y = m_devicePtrTrackedSegmentY;
    auto costs = m_devicePtrCosts;
    const float costWeight = static_cast<float>(cost_weight);
    const size_t trajsSize = trajs_size;
    const size_t pathSize = numPointsPerTrajectory_;
    const size_t trackedSegmentSize = tracked_segment_size;
    const float trackedSegmentLength = tracked_segment_length;
    // local memory for storing per trajectory average cost
    sycl::local_accessor<float, 1> trajCost(sycl::range<1>(trajs_size), h);
    auto global_size = sycl::range<1>(trajs_size * pathSize);
    auto workgroup_size = sycl::range<1>(pathSize);
    // Kernel scope
    h.parallel_for<class refPathCostKernel>(
        sycl::nd_range<1>(global_size, workgroup_size),
        [=](sycl::nd_item<1> item) {
          const size_t traj = item.get_group().get_group_id(0);
          const size_t path_index = item.get_local_id();

          // Initialize local memory once per work-group in the first thread
          if (path_index == 0) {
            for (size_t i = 0; i < trajsSize; ++i) {
              trajCost[i] = 0.0f;
            }
          }
          // Synchronize to make sure initialization is complete
          item.barrier(sycl::access::fence_space::local_space);

          sycl::vec<float, 2> point;
          sycl::vec<float, 2> ref_point;
          float minDist = DEFAULT_MIN_DIST;
          float distance;

          for (size_t i = 0; i < trackedSegmentSize; ++i) {
            point = {X[traj * pathSize + path_index],
                     Y[traj * pathSize + path_index]};
            ref_point = {tracked_X[i], tracked_Y[i]};

            distance = sycl::distance(point, ref_point);
            if (distance < minDist) {
              minDist = distance;
            }
          }

          // Atomically add the computed min costs to the local cost for
          // this trajectory
          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::local_space>
              atomic_cost(trajCost[traj]);
          atomic_cost.fetch_add(minDist);

          // Synchronize again so that all atomic updates are finished
          // before further processing
          item.barrier(sycl::access::fence_space::local_space);

          // normalize the trajectory cost and add end point distance to it
          if (path_index == pathSize - 1) {
            sycl::vec<float, 2> last_path_point = {
                X[traj * pathSize + path_index],
                Y[traj * pathSize + path_index]};
            sycl::vec<float, 2> last_ref_point = {
                tracked_X[trackedSegmentSize - 1],
                tracked_Y[trackedSegmentSize - 1]};

            // get distance between two last points
            float end_point_distance =
                sycl::distance(last_path_point, last_ref_point) /
                trackedSegmentLength;
            trajCost[traj] =
                costWeight *
                ((trajCost[traj] / pathSize + end_point_distance) / 2);

            // Atomically add the computed trajectory costs to the global cost
            // for this trajectory
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_cost(costs[traj]);
            atomic_cost.fetch_add(trajCost[traj]);
          }
        });
  });
}

// Compute the cost of trajectory based on distance to the goal point
sycl::event CostEvaluator::goalCostFunc(const size_t trajs_size,
                                        const float ref_path_length,
                                        const double cost_weight) {
  // -----------------------------------------------------
  //  Parallelize over trajectories.
  //  Calculate distance of the last point in trajectory and reference path.
  //  Atomically add to global trajectory’s cost.
  // -----------------------------------------------------

  // command scope
  return m_q.submit([&](sycl::handler &h) {
    // local copies of class members to be used inside the kernel
    auto X = m_devicePtrPathsX;
    auto Y = m_devicePtrPathsY;
    auto costs = m_devicePtrCosts;
    const float costWeight = static_cast<float>(cost_weight);
    const size_t pathSize = numPointsPerTrajectory_;
    const float pathLength = ref_path_length;
    auto global_size = sycl::range<1>(trajs_size);
    // Last point of reference path
    sycl::vec<float, 2> lastRefPoint = {m_deviceRefPathEnd[0],
                                        m_deviceRefPathEnd[1]};
    // Kernel scope
    h.parallel_for<class goalCostKernel>(
        sycl::range<1>(global_size), [=](sycl::id<1> id) {
          // get last point of the trajectory path
          sycl::vec<float, 2> last_path_point = {
              X[id * pathSize + pathSize - 1], Y[id * pathSize + pathSize - 1]};

          // end point distance normalized by path length
          float distance =
              sycl::distance(last_path_point, lastRefPoint) / pathLength;

          // Atomically add the computed trajectory cost to the global cost
          // for this trajectory
          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
              atomic_cost(costs[id]);
          atomic_cost.fetch_add(costWeight * distance);
        });
  });
}

// Compute the cost of trajectory based on smoothness in velocity commands
sycl::event CostEvaluator::smoothnessCostFunc(const size_t trajs_size,
                                              const double cost_weight) {
  // -----------------------------------------------------
  //  Parallelize over trajectories and velocity indices.
  //  Each valid inner index (i >= 1 and i < numPointsPerTrajectory_)
  //  computes a cost contribution which is atomically added to the
  //  trajectory’s cost.
  // -----------------------------------------------------

  // command scope
  return m_q.submit([&](sycl::handler &h) {
    // local copies of class members to be used inside the kernel
    auto velocitiesVx = m_devicePtrVelocitiesVx;
    auto velocitiesVy = m_devicePtrVelocitiesVy;
    auto velocitiesOmega = m_devicePtrVelocitiesOmega;
    auto costs = m_devicePtrCosts;
    const float costWeight = static_cast<float>(cost_weight);
    const size_t trajsSize = trajs_size;
    const size_t velocitiesSize = numPointsPerTrajectory_ - 1;
    const sycl::vec<float, 3> accLimits = {accLimits_[0], accLimits_[1],
                                           accLimits_[2]};
    sycl::local_accessor<float, 1> trajCost(sycl::range<1>(trajs_size), h);
    auto global_size = sycl::range<1>(trajs_size * velocitiesSize);
    auto workgroup_size = sycl::range<1>(velocitiesSize);
    h.parallel_for<class smoothnessCostKernel>(
        sycl::nd_range<1>(global_size, workgroup_size),
        [=](sycl::nd_item<1> item) {
          const size_t traj = item.get_group().get_group_id();
          const size_t vel_item = item.get_local_id();

          // Initialize local memory once per work-group in the first thread
          if (vel_item == 0) {
            for (size_t i = 0; i < trajsSize; ++i) {
              trajCost[i] = 0.0f;
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

            // Atomically add the computed trajectory cost to the global cost
            // for this trajectory
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_cost(costs[traj]);
            atomic_cost.fetch_add(trajCost[traj]);
          }
        });
  });
}

// Compute the cost of trajectory based on jerk in velocity commands
sycl::event CostEvaluator::jerkCostFunc(const size_t trajs_size,
                                        const double cost_weight) {
  // -----------------------------------------------------
  //  Parallelize over trajectories and velocity indices.
  //  Each valid inner index (i >= 1 and i < numPointsPerTrajectory_)
  //  computes a cost contribution which is atomically added to the
  //  trajectory’s cost.
  // -----------------------------------------------------

  // command scope
  return m_q.submit([&](sycl::handler &h) {
    // local copies of class members to be used inside the kernel
    auto velocitiesVx = m_devicePtrVelocitiesVx;
    auto velocitiesVy = m_devicePtrVelocitiesVy;
    auto velocitiesOmega = m_devicePtrVelocitiesOmega;
    auto costs = m_devicePtrCosts;
    const float costWeight = static_cast<float>(cost_weight);
    const size_t trajsSize = trajs_size;
    const size_t velocitiesSize = numPointsPerTrajectory_ - 1;
    const sycl::vec<float, 3> accLimits = {accLimits_[0], accLimits_[1],
                                           accLimits_[2]};
    sycl::local_accessor<float, 1> trajCost(sycl::range<1>(trajs_size), h);
    auto global_size = sycl::range<1>(trajs_size * velocitiesSize);
    auto workgroup_size = sycl::range<1>(velocitiesSize);
    h.parallel_for<class jerkCostKernel>(
        sycl::nd_range<1>(global_size, workgroup_size),
        [=](sycl::nd_item<1> item) {
          const size_t traj = item.get_group().get_group_id();
          const size_t vel_point = item.get_local_id();

          // Initialize local memory once per work-group in the first thread
          if (vel_point == 0) {
            for (size_t i = 0; i < trajsSize; ++i) {
              trajCost[i] = 0.0f;
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
                  2 * velocitiesOmega[traj * velocitiesSize + (vel_point - 1)] +
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
  });
}

// Calculate obstacle distance cost per trajectory
sycl::event CostEvaluator::obstaclesDistCostFunc(const size_t trajs_size,
                                                 const double cost_weight) {
  // -----------------------------------------------------
  //  Parallelize over trajectories and path indices.
  //  Calculate distance of each point with each obstacle point.
  //  Atomically add lowest distance to the trajectory’s cost.
  // -----------------------------------------------------

  // command scope
  return m_q.submit([&](sycl::handler &h) {
    // local copies of class members to be used inside the kernel
    auto X = m_devicePtrPathsX;
    auto Y = m_devicePtrPathsY;
    auto obs_X = m_devicePtrObstaclesX;
    auto obs_Y = m_devicePtrObstaclesY;
    auto costs = m_devicePtrCosts;
    const float costWeight = static_cast<float>(cost_weight);
    const size_t trajsSize = trajs_size;
    const size_t pathSize = numPointsPerTrajectory_;
    const size_t obsSize = obstaclePointsX.size();
    const float maxObstacleDistance = maxObstaclesDist;
    // local memory for storing per trajectory average cost
    sycl::local_accessor<float, 1> trajCost(sycl::range<1>(trajs_size), h);
    auto global_size = sycl::range<1>(trajs_size * pathSize);
    auto workgroup_size = sycl::range<1>(pathSize);
    // Kernel scope
    h.parallel_for<class obstaclesDistCostKernel>(
        sycl::nd_range<1>(global_size, workgroup_size),
        [=](sycl::nd_item<1> item) {
          const size_t traj = item.get_group().get_group_id(0);
          const size_t path_index = item.get_local_id();

          // Initialize local memory once per work-group in the first thread
          if (path_index == 0) {
            for (size_t i = 0; i < trajsSize; ++i) {
              trajCost[i] = DEFAULT_MIN_DIST;
            }
          }
          // Synchronize to make sure initialization is complete
          item.barrier(sycl::access::fence_space::local_space);

          sycl::vec<float, 2> point;
          sycl::vec<float, 2> ref_point;
          float distance;

          for (size_t i = 0; i < obsSize; ++i) {
            point = {X[traj * pathSize + path_index],
                     Y[traj * pathSize + path_index]};
            ref_point = {obs_X[i], obs_Y[i]};

            distance = sycl::distance(point, ref_point);
            // Atomically add the computed min costs to the local cost for
            // this trajectory
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::local_space>
                atomic_min(trajCost[traj]);
            atomic_min.fetch_min(distance);
          }

          // Synchronize again so that all atomic updates are finished
          // before further processing
          item.barrier(sycl::access::fence_space::local_space);

          // normalize the trajectory cost and add end point distance to it
          if (path_index == 0) {
            // Atomically add the computed trajectory cost to the global cost
            // for this trajectory. Before adding normalize the cost to [0, 1]
            // based on the robot max local range for the obstacles. Minimum
            // cost is assigned at distance value maxObstaclesDist
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_cost(costs[traj]);
            atomic_cost.fetch_add(
                costWeight *
                (sycl::max((maxObstacleDistance - trajCost[traj]), 0.0f) /
                 maxObstacleDistance));
          }
        });
  });
}
}; // namespace Control
} // namespace Kompass
#endif // !GPU
