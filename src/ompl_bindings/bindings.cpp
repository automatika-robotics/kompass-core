#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/Path.h>
#include <ompl/base/Planner.h>
#include <ompl/base/PlannerData.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/State.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/objectives/MaximizeMinClearanceObjective.h>
#include <ompl/base/objectives/MechanicalWorkOptimizationObjective.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/est/BiEST.h>
#include <ompl/geometric/planners/est/EST.h>
#include <ompl/geometric/planners/est/ProjEST.h>
#include <ompl/geometric/planners/fmt/BFMT.h>
#include <ompl/geometric/planners/fmt/FMT.h>
#include <ompl/geometric/planners/informedtrees/ABITstar.h>
#include <ompl/geometric/planners/informedtrees/AITstar.h>
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/geometric/planners/kpiece/BKPIECE1.h>
#include <ompl/geometric/planners/kpiece/KPIECE1.h>
#include <ompl/geometric/planners/kpiece/LBKPIECE1.h>
#include <ompl/geometric/planners/pdst/PDST.h>
#include <ompl/geometric/planners/prm/LazyPRM.h>
#include <ompl/geometric/planners/prm/LazyPRMstar.h>
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/geometric/planners/rrt/LBTRRT.h>
#include <ompl/geometric/planners/rrt/LazyLBTRRT.h>
#include <ompl/geometric/planners/rrt/LazyRRT.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRTXstatic.h>
#include <ompl/geometric/planners/rrt/RRTsharp.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/TRRT.h>
#include <ompl/geometric/planners/sbl/SBL.h>
#include <ompl/geometric/planners/sst/SST.h>
#include <ompl/geometric/planners/stride/STRIDE.h>
#include <ompl/util/Console.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace om = ompl::msg;

namespace py = nanobind;

std::vector<std::string> py_getParamNames(ob::ParamSet &paramset) {
  std::vector<std::string> params;
  paramset.getParamNames(params);
  return params;
}

NB_MODULE(omplpy, m) {
  m.doc() = "Python Bindings for OMPL";

  auto m_base = m.def_submodule("base", "ompl.base");
  m_base.doc() = "OMPL base module containing all base classes for States, "
                 "State Spaces, Parameters and Planners";

  auto m_geometric = m.def_submodule("geometric", "ompl.geometric");
  m_geometric.doc() = "OMPL geometric module containing all the geometric "
                      "planners classes in 'ompl.geometric.planners'";

  auto m_util = m.def_submodule("util", "ompl.util");
  m_util.doc() = "OMPL utilities module used to configure OMPL logging level";

  // States and spaces
  py::class_<ob::ScopedState<>>(m_base, "ScopedState")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .def(py::init<ob::StateSpacePtr>())
      .def(py::init<const ob::ScopedState<> &>())
      .def("getSpace", &ob::ScopedState<>::getSpace)
      .def("random", &ob::ScopedState<>::random)
      .def("enforceBounds", &ob::ScopedState<>::enforceBounds)
      .def("satisfiesBounds", &ob::ScopedState<>::satisfiesBounds)
      .def("reals", &ob::ScopedState<>::reals)
      .def("print", &ob::ScopedState<>::print)
      .def("get", py::overload_cast<>(&ob::ScopedState<>::get),
           py::rv_policy::reference_internal);

  py::class_<ob::State>(m_base, "State");
  py::class_<ob::CompoundState, ob::State>(m_base, "CompoundState");
  py::class_<ob::StateSpace>(m_base, "StateSpace");
  py::class_<ob::CompoundStateSpace, ob::StateSpace>(m_base,
                                                     "CompoundStateSpace");
  py::class_<ob::SpaceInformation>(m_base, "SpaceInformation")
      .def(py::init<ob::StateSpacePtr>())
      .def("satisfiesBounds", &ob::SpaceInformation::satisfiesBounds);

  py::class_<ob::SE2StateSpace::StateType, ob::CompoundState>(m_base,
                                                              "SE2StateType")
      .def(py::init<>())
      .def("getX", &ob::SE2StateSpace::StateType::getX)
      .def("getY", &ob::SE2StateSpace::StateType::getY)
      .def("getYaw", &ob::SE2StateSpace::StateType::getYaw)
      .def("setX", &ob::SE2StateSpace::StateType::setX)
      .def("setY", &ob::SE2StateSpace::StateType::setY)
      .def("setYaw", &ob::SE2StateSpace::StateType::setYaw)
      .doc() = "A state in the SE2StateSpace (2D space) SE(2): (x, y, yaw)";

  py::class_<ob::SE2StateSpace, ob::CompoundStateSpace>(m_base, "SE2StateSpace")
      .def(py::init<>())
      .def("setBounds", &ob::SE2StateSpace::setBounds)
      .def("allocState", &ob::SE2StateSpace::allocState)
      .def("freeState", &ob::SE2StateSpace::freeState)
      .doc() =
      "2D space SE(2) used for planning 2D navigation in the (x,y) plane";

  // Simple Setup
  py::class_<og::SimpleSetup>(m_geometric, "SimpleSetup")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .def(py::init<const ob::StateSpacePtr &>())
      .def("setup", &og::SimpleSetup::setup)
      .def("clear", &og::SimpleSetup::clear)
      .def("setStartAndGoalStates", &og::SimpleSetup::setStartAndGoalStates,
           py::arg("start"), py::arg("goal"), py::arg("threshold") = 0.01)
      .def("setGoalState", &og::SimpleSetup::setGoalState)
      .def("setGoal", &og::SimpleSetup::setGoal)
      .def("solve",
           py::overload_cast<const ob::PlannerTerminationCondition &>(
               &og::SimpleSetup::solve),
           "Solve using PlannerTerminationCondition")
      .def("solve", py::overload_cast<double>(&og::SimpleSetup::solve),
           "Solve using time")
      .def("simplifySolution",
           py::overload_cast<const ob::PlannerTerminationCondition &>(
               &og::SimpleSetup::simplifySolution))
      .def("simplifySolution",
           py::overload_cast<double>(&og::SimpleSetup::simplifySolution))
      .def("getSolutionPlannerName", &og::SimpleSetup::getSolutionPlannerName)
      .def("getSolutionPath", &og::SimpleSetup::getSolutionPath)
      .def("getPlannerData", &og::SimpleSetup::getPlannerData)
      .def("setStateValidityChecker",
           py::overload_cast<const ob::StateValidityCheckerPtr &>(
               &og::SimpleSetup::setStateValidityChecker))
      .def("setStateValidityChecker", // State Validy Checker Function - Takes
                                      // of form Callable(State) -> bool
           py::overload_cast<const ob::StateValidityCheckerFn &>(
               &og::SimpleSetup::setStateValidityChecker))
      .def("print", &og::SimpleSetup::print)
      .def("getSpaceInformation", &og::SimpleSetup::getSpaceInformation,
           py::rv_policy::copy)
      .def("setPlanner", &og::SimpleSetup::setPlanner)
      .def("setOptimizationObjective",
           &og::SimpleSetup::setOptimizationObjective)
      .doc() = "Main class used to create the set of objects typically needed "
               "to solve a geometric planning problem";

  // Cost
  py::class_<ob::Cost>(m_base, "Cost")
      .def(py::init<double>())
      .def("value", &ob::Cost::value);

  // Path Objects
  py::class_<ob::Path>(m_base, "Path");
  py::class_<og::PathGeometric, ob::Path>(m_geometric, "PathGeometric")
      .def("cost", &og::PathGeometric::cost)
      .def("length", &og::PathGeometric::length)
      .def("smoothness", &og::PathGeometric::smoothness)
      .def("clearance", &og::PathGeometric::clearance)
      .def("interpolate",
           py::overload_cast<unsigned int>(&og::PathGeometric::interpolate))
      .def("interpolate", py::overload_cast<>(&og::PathGeometric::interpolate))
      .def("subdivide", &og::PathGeometric::subdivide)
      .def("reverse", &og::PathGeometric::reverse)
      .def("checkAndRepair", &og::PathGeometric::checkAndRepair)
      .def("append",
           py::overload_cast<const ob::State *>(&og::PathGeometric::append))
      .def("append", py::overload_cast<const og::PathGeometric &>(
                         &og::PathGeometric::append))
      .def("prepend", &og::PathGeometric::prepend)
      .def("keepAfter", &og::PathGeometric::keepAfter)
      .def("keepBefore", &og::PathGeometric::keepBefore)
      .def("getClosestIndex", &og::PathGeometric::getClosestIndex)
      .def("getStates", &og::PathGeometric::getStates,
           py::rv_policy::reference_internal)
      .def("getState",
           py::overload_cast<unsigned int>(&og::PathGeometric::getState),
           py::rv_policy::reference_internal)
      .def("getStateCount", &og::PathGeometric::getStateCount)
      .def("clear", &og::PathGeometric::clear)
      .doc() = "Definition of a geometric path along with possible methods "
               "applied to the geomteric path";

  py::class_<ob::RealVectorBounds>(m_base, "RealVectorBounds")
      .def(py::init<unsigned int>(), py::arg("dim"))
      .def("setLow", py::overload_cast<double>(&ob::RealVectorBounds::setLow),
           py::arg("value"))
      .def("setLow",
           py::overload_cast<unsigned int, double>(
               &ob::RealVectorBounds::setLow),
           py::arg("index"), py::arg("value"))
      .def("setHigh", py::overload_cast<double>(&ob::RealVectorBounds::setHigh),
           py::arg("value"))
      .def("setHigh",
           py::overload_cast<unsigned int, double>(
               &ob::RealVectorBounds::setHigh),
           py::arg("index"), py::arg("value"))
      .def("resize", &ob::RealVectorBounds::resize, py::arg("size"))
      .def("getVolume", &ob::RealVectorBounds::getVolume)
      .def("getDifference", &ob::RealVectorBounds::getDifference)
      .def("check", &ob::RealVectorBounds::check)
      .def_rw("low", &ob::RealVectorBounds::low)
      .def_rw("high", &ob::RealVectorBounds::high)
      .doc() = "The lower and upper bounds for an R^n space, used to set the "
               "bounds on the used StateSpace in the planning problem";

  // Params and Paramset
  py::class_<ob::GenericParam>(m_base, "GenericParam")
      .def("getName", &ob::GenericParam::getName)
      .def("setName", &ob::GenericParam::setName)
      .def("setValue", &ob::GenericParam::setValue)
      .def("getValue", &ob::GenericParam::getValue)
      .def("setRangeSuggestion", &ob::GenericParam::setRangeSuggestion)
      .def("getRangeSuggestion", &ob::GenericParam::getRangeSuggestion)
      .doc() =
      "Motion planning algorithms often employ parameters to guide their "
      "exploration process. (e.g., goal biasing). Motion planners (and some of "
      "their components) use this class to declare what the parameters are, in "
      "a generic way, so that they can be set externally.";

  py::class_<ob::ParamSet>(m_base, "ParamSet")
      .def(py::init<>())
      .def("setParam", &ob::ParamSet::setParam)
      .def("setParams", &ob::ParamSet::setParams, py::arg("kv"),
           py::arg("ignoreUnknown") = false)
      // custom method to get param names
      .def("getParamNames", &py_getParamNames)
      .def("getParamValues", &ob::ParamSet::getParamValues)
      .def("hasParam", &ob::ParamSet::hasParam)
      .def("__getitem__", &ob::ParamSet::operator[],
           py::rv_policy::reference_internal)
      .doc() = "A set of GenericParam used for a Motion planning algorithm";

  // Exposed Planners
  py::class_<ob::Planner>(m_base, "Planner")
      .def("params", py::overload_cast<>(&ob::Planner::params),
           py::rv_policy::reference_internal)
      .def("params", py::overload_cast<>(&ob::Planner::params, py::const_),
           py::rv_policy::reference_internal)
      .doc() = "Generic parent class for any motion planner";

  py::class_<og::EST, ob::Planner>(m_geometric, "EST")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Expansive Space Trees (EST) Planner";

  py::class_<og::BiEST, ob::Planner>(m_geometric, "BiEST")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Bi-directional Expansive Space Trees (BiEST) Planner";

  py::class_<og::FMT, ob::Planner>(m_geometric, "FMT")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Asymptotically Optimal Fast Marching Tree algorithm developed "
               "by L. Janson and M. Pavone";

  py::class_<og::BFMT, ob::Planner>(m_geometric, "BFMT")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Bidirectional Asymptotically Optimal Fast Marching Tree "
               "algorithm developed by J. Starek, J.V. Gomez, et al.";

  py::class_<og::ABITstar, ob::Planner>(m_geometric, "ABITstar")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Advanced Batch Informed Trees (ABIT*) Planner";

  py::class_<og::AITstar, ob::Planner>(m_geometric, "AITstar")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Adaptively Informed Trees (AIT*) Planner";

  py::class_<og::BITstar, ob::Planner>(m_geometric, "BITstar")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Batch Informed Trees (BIT*) Planner";

  py::class_<og::BKPIECE1, ob::Planner>(m_geometric, "BKPIECE1")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Bi-directional KPIECE with one level of discretization";

  py::class_<og::InformedRRTstar, ob::Planner>(m_geometric, "InformedRRTstar")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Informed Optimal Rapidly-exploring Random Trees RRT*";

  py::class_<og::KPIECE1, ob::Planner>(m_geometric, "KPIECE1")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Kinematic Planning by Interior-Exterior Cell Exploration";

  py::class_<og::LBKPIECE1, ob::Planner>(m_geometric, "LBKPIECE1")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Lazy Bi-directional KPIECE with one level of discretization";

  py::class_<og::LBTRRT, ob::Planner>(m_geometric, "LBTRRT")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Lower Bound Tree Rapidly-exploring Random Trees";

  py::class_<og::LazyLBTRRT, ob::Planner>(m_geometric, "LazyLBTRRT")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Rapidly-exploring Random Trees";

  py::class_<og::LazyPRM, ob::Planner>(m_geometric, "LazyPRM")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Lazy Probabilistic RoadMap planner";

  py::class_<og::LazyPRMstar, ob::Planner>(m_geometric, "LazyPRMstar")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Probabilistic RoadMap Star (PRM*) planner";

  py::class_<og::LazyRRT, ob::Planner>(m_geometric, "LazyRRT")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Lazy Rapidly-exploring Random Trees";

  py::class_<og::PDST, ob::Planner>(m_geometric, "PDST")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Path-Directed Subdivision Tree";

  py::class_<og::PRM, ob::Planner>(m_geometric, "PRM")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Probabilistic RoadMap planner";

  py::class_<og::PRMstar, ob::Planner>(m_geometric, "PRMstar")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Probabilistic RoadMap planner Start";

  py::class_<og::ProjEST, ob::Planner>(m_geometric, "ProjEST")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Expansive Space Trees";

  py::class_<og::RRT, ob::Planner>(m_geometric, "RRT")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Rapidly-exploring Random Trees";

  py::class_<og::RRTConnect, ob::Planner>(m_geometric, "RRTConnect")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "RRTConnect";

  py::class_<og::RRTXstatic, ob::Planner>(m_geometric, "RRTXstatic")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Optimal Rapidly-exploring Random Trees Maintaining A Pseudo "
               "Optimal Tree";

  py::class_<og::RRTsharp, ob::Planner>(m_geometric, "RRTsharp")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() =
      "Optimal Rapidly-exploring Random Trees Maintaining An Optimal Tree";

  py::class_<og::RRTstar, ob::Planner>(m_geometric, "RRTstar")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Optimal Rapidly-exploring Random Trees";

  py::class_<og::TRRT, ob::Planner>(m_geometric, "TRRT")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Transition-based Rapidly-exploring Random Trees";

  py::class_<og::SST, ob::Planner>(m_geometric, "SST")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Sparse Sampling Tree";

  py::class_<og::SBL, ob::Planner>(m_geometric, "SBL")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "Single-Query Bi-Directional Probabilistic Roadmap Planner with "
               "Lazy Collision Checking";

  py::class_<og::STRIDE, ob::Planner>(m_geometric, "STRIDE")
      .def(py::init<const ob::SpaceInformationPtr &, bool, unsigned int,
                    unsigned int, unsigned int, unsigned int, double>(),
           py::arg("si"), py::arg("useProjectedDistance") = false,
           py::arg("degree") = 16, py::arg("minDegree") = 12,
           py::arg("maxDegree") = 18, py::arg("maxNumPtsPerLeaf") = 6,
           py::arg("estimatedDimension") = 0.0)
      .doc() = "Search Tree with Resolution Independent Density Estimation";

  // Planner Status
  py::enum_<ob::PlannerStatus::StatusType>(m_base, "StatusType")
      .value("UNKNOWN", ob::PlannerStatus::UNKNOWN)
      .value("INVALID_START", ob::PlannerStatus::INVALID_START)
      .value("INVALID_GOAL", ob::PlannerStatus::INVALID_GOAL)
      .value("UNRECOGNIZED_GOAL_TYPE",
             ob::PlannerStatus::UNRECOGNIZED_GOAL_TYPE)
      .value("TIMEOUT", ob::PlannerStatus::TIMEOUT)
      .value("APPROXIMATE_SOLUTION", ob::PlannerStatus::APPROXIMATE_SOLUTION)
      .value("EXACT_SOLUTION", ob::PlannerStatus::EXACT_SOLUTION)
      .value("CRASH", ob::PlannerStatus::CRASH)
      .value("ABORT", ob::PlannerStatus::ABORT)
      .value("TYPE_COUNT", ob::PlannerStatus::TYPE_COUNT)
      .export_values();

  py::class_<ob::PlannerStatus>(m_base, "PlannerStatus")
      .def(py::init<>())
      .def(py::init<ob::PlannerStatus::StatusType>())
      .def(py::init<bool, bool>())
      .def("asString", &ob::PlannerStatus::asString)
      .def("__bool__",
           [](const ob::PlannerStatus &status) {
             return static_cast<bool>(status);
           })
      .def("__repr__",
           [](const ob::PlannerStatus &status) {
             return "<PlannerStatus: " + status.asString() + ">";
           })
      .doc() = "A class to store the result (exit status) of Planner.solve()";

  // Optimization Objectives
  py::class_<ob::OptimizationObjective>(m_base, "OptimizationObjective");

  py::class_<ob::PathLengthOptimizationObjective, ob::OptimizationObjective>(
      m_base, "PathLengthOptimizationObjective")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() =
      "An optimization objective which corresponds to optimizing path length";

  py::class_<ob::MechanicalWorkOptimizationObjective,
             ob::OptimizationObjective>(m_base,
                                        "MechanicalWorkOptimizationObjective")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() = "An optimization objective which defines path cost using the "
               "idea of mechanical work. To be used in conjunction with TRRT";

  py::class_<ob::MaximizeMinClearanceObjective, ob::OptimizationObjective>(
      m_base, "MaximizeMinClearanceObjective")
      .def(py::init<const ob::SpaceInformationPtr &>())
      .doc() =
      "Objective for attempting to maximize the minimum clearance along a path";

  // logging
  py::enum_<om::LogLevel>(m_util, "LogLevel")
      .value("LOG_DEV2", om::LogLevel::LOG_DEV2)
      .value("LOG_DEV1", om::LogLevel::LOG_DEV1)
      .value("LOG_DEBUG", om::LogLevel::LOG_DEBUG)
      .value("LOG_INFO", om::LogLevel::LOG_INFO)
      .value("LOG_WARN", om::LogLevel::LOG_WARN)
      .value("LOG_ERROR", om::LogLevel::LOG_ERROR)
      .value("LOG_NONE", om::LogLevel::LOG_NONE)
      .export_values();

  m_util.def("setLogLevel", &om::setLogLevel, py::arg("level"));
  m_util.def("getLogLevel", &om::getLogLevel);
}
