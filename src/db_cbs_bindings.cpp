#include "db_cbs.hpp"
#include "db_ecbs_lib.hpp"
#include "dynobench/motions.hpp"
#include "dynobench/multirobot_trajectory.hpp"
#include "dynobench/robot_models.hpp"
#include "dynobench/robot_models_base.hpp"
#include <memory>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <vector>
#include <yaml-cpp/yaml.h>
namespace nb = nanobind;
using namespace dynobench;

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

        std::vector<Result> result =
            db_cbs(env, outputFile, optimizationFile, cfg, timeLimitdbAstar,
                   timeLimitdbCBS);

        std::cout.clear();
        // return nb::cast(std::move(result), nb::rv_policy::move);
        return result;
      },
      nb::rv_policy::move, nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("input_file"), nb::arg("output_file"),
      nb::arg("optimization_file"), nb::arg("cfg"),
      nb::arg("time_limit_db_astar"), nb::arg("time_limit_db_cbs"));

  m.def(
      "db_ecbs",
      [](nb::dict inputEnv, std::string outputFile,
         std::string optimizationFile, nb::dict inputCfg,
         double timeLimitdbAstar, double timeLimitdbCBS) {
        std::cout.setstate(std::ios::failbit);

        YAML::Node env = pythonToYaml(inputEnv);
        YAML::Node cfg = pythonToYaml(inputCfg);

        std::vector<Result> result =
            db_ecbs(env, outputFile, optimizationFile, cfg, timeLimitdbAstar,
                   timeLimitdbCBS);
        // std::vector<Result> result;

        std::cout.clear();
        // return nb::cast(std::move(result), nb::rv_policy::move);
        return result;
      },
      nb::rv_policy::move, nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("input_file"), nb::arg("output_file"),
      nb::arg("optimization_file"), nb::arg("cfg"),
      nb::arg("time_limit_db_astar"), nb::arg("time_limit_db_cbs"));

  nb::class_<Result>(m, "Result")
      .def_ro("discrete", &Result::discrete)
      .def_ro("optimized", &Result::optimized)
      .def_ro("runtime", &Result::runtime)
      .def_ro("delta", &Result::delta)
      .def("__del__", [](Result *self) { delete self; });

  nb::class_<dynobench::Trajectory>(m, "Trajectory")
      // .def(nb::init())
      .def("create", []() { return std::make_unique<Trajectory>(); })
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
      .def_rw("start", &dynobench::Trajectory::start)
      .def_rw("goal", &dynobench::Trajectory::goal)
      .def_rw("actions", &dynobench::Trajectory::actions)
      .def_rw("states", &dynobench::Trajectory::states)
      .def_ro("primitive_actions", &dynobench::Trajectory::primitive_actions)
      .def_ro("primitive_states", &dynobench::Trajectory::primitive_states)
      .def("check", &dynobench::Trajectory::check)
      .def("update_feasibility", &dynobench::Trajectory::update_feasibility)
      .def("__del__", [](dynobench::Trajectory *self) { delete self; });

  nb::class_<MultiRobotTrajectory>(m, "MultiRobotTrajectory")
      .def_ro("trajectories", &MultiRobotTrajectory::trajectories)
      .def("__del__", [](MultiRobotTrajectory *self) { delete self; });

  nb::class_<Feasibility_thresholds>(m, "FeasibilityThresholds")
      .def(nb::init());

  nb::class_<Model_robot>(m, "Model_robot")
      .def(nb::init())
      .def("setPositionBounds", &Model_robot::setPositionBounds)
      .def("get_translation_invariance",
           &Model_robot::get_translation_invariance)
      .def("get_x_ub", &Model_robot::get_x_ub)
      .def("set_position_ub", &Model_robot::set_position_ub)
      .def("set_position_lb", &Model_robot::set_position_lb)
      .def("get_x_lb", &Model_robot::get_x_lb)
      .def("get_nx", &Model_robot::get_nx)
      .def("get_nu", &Model_robot::get_nu)

      .def("get_u_ub", &Model_robot::get_u_ub)
      .def("get_u_lb", &Model_robot::get_u_lb)
      .def("get_x_desc", &Model_robot::get_x_desc)
      .def("get_u_desc", &Model_robot::get_u_desc)
      .def("get_u_ref", &Model_robot::get_u_ref)
      .def("stepDiffOut",
           [](Model_robot &robot, Eigen::Ref<Eigen::VectorXd> x,
              Eigen::Ref<Eigen::VectorXd> u, double dt) {
             Eigen::MatrixXd Jx =
                 Eigen::MatrixXd::Zero(robot.get_nx(), robot.get_nx());
             Eigen::MatrixXd Ju =
                 Eigen::MatrixXd::Zero(robot.get_nx(), robot.get_nu());
             robot.stepDiff(Jx, Ju, x, u, dt);
             return std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>(Jx, Ju);
           })

      // .def("stepDiffdt", &Model_robot::stepDiffdt)
      .def("calcDiffVOut",
           [](Model_robot &robot, Eigen::Ref<Eigen::VectorXd> x,
              Eigen::Ref<Eigen::VectorXd> u) {
             Eigen::MatrixXd Jx =
                 Eigen::MatrixXd::Zero(robot.get_nx(), robot.get_nx());
             Eigen::MatrixXd Ju =
                 Eigen::MatrixXd::Zero(robot.get_nx(), robot.get_nu());
             robot.calcDiffV(Jx, Ju, x, u);
             return std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>(Jx, Ju);
           })
      .def("calcV", &Model_robot::calcV)
      .def("step", &Model_robot::step)
      .def("stepOut",
           [](Model_robot &robot, Eigen::Ref<Eigen::VectorXd> x,
              Eigen::Ref<Eigen::VectorXd> u, double dt) {
             Eigen::VectorXd x_next = Eigen::VectorXd::Zero(robot.get_nx());
             robot.step(x_next, x, u, dt);
             return x_next;
           })
      .def("stepR4", &Model_robot::stepR4)
      .def("distance", &Model_robot::distance)
      .def("sample_uniform", &Model_robot::sample_uniform)
      .def("interpolate", &Model_robot::interpolate)
      .def("lower_bound_time", &Model_robot::lower_bound_time)
      .def("collision_distance", &Model_robot::collision_distance)
      .def("collision_distance_diff", &Model_robot::collision_distance_diff)
      .def("get_info", &Model_robot::get_info)
      .def("transformation_collision_geometries",
           &Model_robot::transformation_collision_geometries);

  m.def("robot_factory", &robot_factory);
  m.def("robot_factory_with_env", &robot_factory_with_env);

  m.def("clock_seed", [] { std::srand(std::time(nullptr)); });
  m.def("seed", [](int seed) { std::srand(seed); });
  m.def("rand", [] { return std::rand(); });
  m.def("rand01", [] { return (double)std::rand() / RAND_MAX; });
}
