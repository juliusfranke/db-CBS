#include "db_cbs.hpp"
#include "dynobench/motions.hpp"
#include "dynobench/multirobot_trajectory.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <yaml-cpp/yaml.h>
namespace nb = nanobind;

YAML::Node pythonToYaml(const nb::handle &obj) {
  YAML::Node node;

  if (nb::isinstance<nb::dict>(obj)) {
    // Handle Python dictionary
    nb::dict dict = nb::cast<nb::dict>(obj);
    for (auto [key, value] : dict) {
      std::string key_str = nb::cast<std::string>(key);
      node[key_str] = pythonToYaml(value);
    }
  } else if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
    // Handle Python list/tuple
    node = YAML::Node(YAML::NodeType::Sequence);
    nb::list list = nb::cast<nb::list>(obj);
    for (auto item : list) {
      node.push_back(pythonToYaml(item));
    }
  } else if (nb::isinstance<nb::bool_>(obj)) {
    node = nb::cast<bool>(obj);
  } else if (nb::isinstance<nb::str>(obj)) {
    node = nb::cast<std::string>(obj);
  } else if (nb::isinstance<nb::int_>(obj)) {
    node = nb::cast<int>(obj);
  } else if (nb::isinstance<nb::float_>(obj)) {
    node = nb::cast<double>(obj);
  } else if (obj.is_none()) {
    node = YAML::Node();
  } else {
    throw std::runtime_error(
        "Unsupported Python type for conversion to YAML::Node");
  }
  return node;
}

void processYamlNode(const YAML::Node &node, const std::string &arg1,
                     const std::string &arg2, const std::string &arg3,
                     double arg4) {
  // Example function that takes YAML::Node and other arguments
  std::cout << "Processing YAML node with additional arguments:\n";
  std::cout << "YAML Node:\n" << node << "\n";
  std::cout << "Arguments: " << arg1 << ", " << arg2 << ", " << arg3 << ", "
            << arg4 << "\n";
}

NB_MODULE(dbcbs_py, m) {
  m.def(
      "db_cbs",
      [](nb::dict inputEnv, std::string outputFile,
         std::string optimizationFile, nb::dict inputCfg,
         double timeLimitdbAstar, double timeLimitdbCBS) {
        std::cout.setstate(std::ios::failbit);

        YAML::Node env = pythonToYaml(inputEnv);
        YAML::Node cfg = pythonToYaml(inputCfg);

        Result result = db_cbs(env, outputFile, optimizationFile, cfg,
                               timeLimitdbAstar, timeLimitdbCBS);

        std::cout.clear();
        return result;
      },
      nb::call_guard<nb::gil_scoped_release>(), nb::arg("input_file"),
      nb::arg("output_file"), nb::arg("optimization_file"), nb::arg("cfg"),
      nb::arg("time_limit_db_astar"), nb::arg("time_limit_db_cbs"));
  nb::class_<Result>(m, "Result")
      .def_ro("discrete", &Result::discrete)
      .def_ro("optimized", &Result::optimized)
      .def_ro("runtime", &Result::runtime);
  nb::class_<dynobench::Trajectory>(m, "Trajectory")
      .def_ro("time_stamp", &dynobench::Trajectory::time_stamp)
      .def_ro("cost", &dynobench::Trajectory::cost)
      .def_ro("feasible", &dynobench::Trajectory::feasible)
      .def_ro("fmin", &dynobench::Trajectory::fmin)
      .def_ro("traj_feas", &dynobench::Trajectory::traj_feas)
      .def_ro("goal_feas", &dynobench::Trajectory::goal_feas)
      .def_ro("col_feas", &dynobench::Trajectory::col_feas)
      .def_ro("x_bounds_feas", &dynobench::Trajectory::x_bounds_feas)
      .def_ro("u_bounds_feas", &dynobench::Trajectory::u_bounds_feas)
      .def_ro("max_jump", &dynobench::Trajectory::max_jump)
      .def_ro("max_collision", &dynobench::Trajectory::max_collision)
      .def_ro("goal_distance", &dynobench::Trajectory::goal_distance)
      .def_ro("start_distance", &dynobench::Trajectory::start_distance)
      .def_ro("x_bound_distance", &dynobench::Trajectory::x_bound_distance)
      .def_ro("u_bound_distance", &dynobench::Trajectory::u_bound_distance)
      .def_ro("start", &dynobench::Trajectory::start)
      .def_ro("goal", &dynobench::Trajectory::goal)
      .def_ro("actions", &dynobench::Trajectory::actions)
      .def_ro("states", &dynobench::Trajectory::states)
      .def_ro("primitive_actions", &dynobench::Trajectory::primitive_actions)
      .def_ro("primitive_states", &dynobench::Trajectory::primitive_states);

  nb::class_<MultiRobotTrajectory>(m, "MultiRobotTrajectory")
      .def_ro("trajectories", &MultiRobotTrajectory::trajectories);
  /* std::vector<dynobench::Trajectory> trajectories; */
}
