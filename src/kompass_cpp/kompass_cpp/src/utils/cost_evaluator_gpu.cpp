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
    LOG_WARNING(
        "Number of points per sample trajectory are more than", max_wg_size_,
        "for your device. This will make the cost kernels run in "
        "strides. For fastest execution try to modify the control time step "
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

    // The input arrays have size (N-1) relative to points
    const size_t velocitiesCount = numPointsPerTrajectory_ - 1;

    // Pre-calculate Inverse Limits to convert Division -> Multiplication
    // Avoid division by zero
    const float invLimX = (accLimits_[0] > 0) ? 1.0f / accLimits_[0] : 0.0f;
    const float invLimY = (accLimits_[1] > 0) ? 1.0f / accLimits_[1] : 0.0f;
    const float invLimOmega = (accLimits_[2] > 0) ? 1.0f / accLimits_[2] : 0.0f;

    // Configure kernel based on device work-group size
    const size_t WG_SIZE = max_wg_size_;

    // Global Size = Trajectories * Threads per Trajectory
    auto global_size = sycl::range<1>(trajs_size * WG_SIZE);
    auto local_size = sycl::range<1>(WG_SIZE);
    h.parallel_for<class smoothnessCostKernel>(
        sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
          const size_t traj_idx = item.get_group(0);
          const size_t local_id = item.get_local_id(0);

          // Each thread calculates a partial sum for its assigned points.
          float local_cost_contrib = 0.0f;

          // -----------------------------------------------------------
          // STRIDE LOOP
          // -----------------------------------------------------------
          // We strictly skip i=0 inside the loop because we need (i-1).
          for (size_t i = local_id; i < velocitiesCount; i += WG_SIZE) {

            // Skip index 0 (cannot compute delta)
            if (i == 0)
              continue;

            // Compute index
            size_t curr_idx = traj_idx * velocitiesCount + i;
            size_t prev_idx = curr_idx - 1;

            // Load data (Expecting L2 cache hits for efficiency)
            float vx_curr = velocitiesVx[curr_idx];
            float vx_prev = velocitiesVx[prev_idx];
            float dv_x = vx_curr - vx_prev;
            local_cost_contrib += (dv_x * dv_x) * invLimX;

            float vy_curr = velocitiesVy[curr_idx];
            float vy_prev = velocitiesVy[prev_idx];
            float dv_y = vy_curr - vy_prev;
            local_cost_contrib += (dv_y * dv_y) * invLimY;

            float w_curr = velocitiesOmega[curr_idx];
            float w_prev = velocitiesOmega[prev_idx];
            float dv_w = w_curr - w_prev;
            local_cost_contrib += (dv_w * dv_w) * invLimOmega;
          }

          // Sum up the partial results from all threads in the work-group
          float traj_total_cost = sycl::reduce_over_group(
              item.get_group(), local_cost_contrib, sycl::plus<float>());

          // normalize average cost using the first thread of the group and
          // add them to global cost
          if (local_id == 0) {
            float final_val =
                costWeight * (traj_total_cost / (3.0f * velocitiesCount));
            // Atomically add the computed trajectory cost to the global cost
            // for this trajectory
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_cost(costs[traj_idx]);
            atomic_cost.fetch_add(final_val);
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
    const size_t pathSize = numPointsPerTrajectory_;
    const size_t obsSize = obstaclePointsX.size();
    const float maxObstacleDistance = maxObstaclesDist;

    // Configure kernel workgroup size using Device Limits
    const size_t WG_SIZE = max_wg_size_;

    // Define Local Memory (Shared Memory) Accessors
    // Size is dynamic based on the workgroup size to match 1-to-1 loading
    sycl::local_accessor<float, 1> tile_obs_X(sycl::range<1>(WG_SIZE), h);
    sycl::local_accessor<float, 1> tile_obs_Y(sycl::range<1>(WG_SIZE), h);

    // Launch Config:
    // Global Size = Number of Trajectories * WorkGroup Size
    // Local Size  = WorkGroup Size
    auto global_size = sycl::range<1>(trajs_size * WG_SIZE);
    auto local_size = sycl::range<1>(WG_SIZE);
    h.parallel_for<class obstaclesDistCostKernel>(
        sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
          const size_t traj_idx = item.get_group(0);
          const size_t local_id = item.get_local_id(0);

          // Initialize per-thread minimum distance to max
          float min_dist_for_point = DEFAULT_MIN_DIST;

          // ---------------------------------------------------------
          // TILING LOOP: Iterate over obstacles in chunks
          // ---------------------------------------------------------
          for (size_t tile_base = 0; tile_base < obsSize;
               tile_base += WG_SIZE) {

            // Threads load one obstacle each into Local Memory
            size_t obs_idx = tile_base + local_id;

            // Guard against reading past the end of the obstacle list
            if (obs_idx < obsSize) {
              tile_obs_X[local_id] = obs_X[obs_idx];
              tile_obs_Y[local_id] = obs_Y[obs_idx];
            }

            // Wait for the entire tile to be loaded
            item.barrier(sycl::access::fence_space::local_space);

            // Compare trajectory points against this tile
            // Calculate valid obstacles in this current tile
            size_t current_tile_count = sycl::min(WG_SIZE, obsSize - tile_base);

            // STRIDE LOOP: Handles pathSize > WG_SIZE
            // Threads iterate in steps of WG_SIZE to cover the whole path.
            for (size_t k = local_id; k < pathSize; k += WG_SIZE) {

              // Load the specific point for this trajectory
              float px = X[traj_idx * pathSize + k];
              float py = Y[traj_idx * pathSize + k];

              // Check distance against all cached obstacles in this tile
              for (size_t j = 0; j < current_tile_count; ++j) {
                float ox = tile_obs_X[j];
                float oy = tile_obs_Y[j];

                float dx = px - ox;
                float dy = py - oy;

                // Euclidean distance
                float d = sycl::sqrt(dx * dx + dy * dy);
                min_dist_for_point = sycl::fmin(min_dist_for_point, d);
              }
            }
            // Synchronization: Wait for computation to finish before
            // overwriting tile
            item.barrier(sycl::access::fence_space::local_space);
          }

          // Reduce the minimum across the entire trajectory
          // Finds the single closest obstacle to ANY point in this trajectory.
          float traj_min_dist = sycl::reduce_over_group(
              item.get_group(), min_dist_for_point, sycl::minimum<float>());

          // normalize the cost and add to global cost of the trajectory
          if (local_id == 0) {
            // Atomically add the computed trajectory cost to the global cost
            // for this trajectory. Before adding normalize the cost to [0, 1]
            // based on the robot max local range for the obstacles. Minimum
            // cost is assigned at distance value maxObstaclesDist
            float normalized_cost =
                sycl::max((maxObstacleDistance - traj_min_dist), 0.0f) /
                maxObstacleDistance;

            // Apply weight and update global costs
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_cost(costs[traj_idx]);

            atomic_cost.fetch_add(costWeight * normalized_cost);
          }
        });
  });
}
}; // namespace Control
} // namespace Kompass
#endif // !GPU
