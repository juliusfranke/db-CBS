#include <algorithm>
#include <bits/stdc++.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <yaml-cpp/yaml.h>
// BOOST
#include <boost/heap/d_ary_heap.hpp>
#include <boost/program_options.hpp>

#include "fcl/broadphase/broadphase_collision_manager.h"
#include "fclStateValidityChecker.hpp"
#include "robotStatePropagator.hpp"
#include "robots.h"
#include <fcl/fcl.h>
// #include "planresult.hpp"
#include "dbcbs_utils.hpp"
#include "dynobench/motions.hpp"
#include "dynoplan/tdbastar/planresult.hpp"
#include "dynoplan/tdbastar/tdbastar.hpp"
#include <dynobench/multirobot_trajectory.hpp>
#include <dynoplan/optimization/multirobot_optimization.hpp>
#include <tuple>
// nn
#include "dynobench/nn.h"
// Conflicts

bool getEarliestConflict(
    const std::vector<LowLevelPlan<dynobench::Trajectory>> &solution,
    const std::vector<std::shared_ptr<dynobench::Model_robot>> &all_robots,
    std::shared_ptr<fcl::BroadPhaseCollisionManagerd> col_mng_robots,
    std::vector<fcl::CollisionObjectd *> &robot_objs,
    Conflict &early_conflict) {
  size_t max_t = 0;
  for (const auto &sol : solution) {
    max_t = std::max(max_t, sol.trajectory.states.size() - 1);
  }
  Eigen::VectorXd node_state;
  std::vector<Eigen::VectorXd> node_states;

  for (size_t t = 0; t <= max_t; ++t) {
    node_states.clear();
    size_t robot_idx = 0;
    size_t obj_idx = 0;
    std::vector<fcl::Transform3d> ts_data;
    for (auto &robot : all_robots) {
      if (t >= solution[robot_idx].trajectory.states.size()) {
        node_state = solution[robot_idx].trajectory.states.back();
      } else {
        node_state = solution[robot_idx].trajectory.states[t];
      }
      node_states.push_back(node_state);
      std::vector<fcl::Transform3d> tmp_ts(1);
      if (robot->name == "car_with_trailers") {
        tmp_ts.resize(2);
      }
      robot->transformation_collision_geometries(node_state, tmp_ts);
      ts_data.insert(ts_data.end(), tmp_ts.begin(), tmp_ts.end());
      // ts_data.insert(ts_data.end(), tmp_ts.back()); // just trailer
      ++robot_idx;
    }
    for (size_t i = 0; i < ts_data.size(); i++) {
      fcl::Transform3d &transform = ts_data[i];
      robot_objs[obj_idx]->setTranslation(transform.translation());
      robot_objs[obj_idx]->setRotation(transform.rotation());
      robot_objs[obj_idx]->computeAABB();
      ++obj_idx;
    }
    col_mng_robots->update(robot_objs);
    fcl::DefaultCollisionData<double> collision_data;
    col_mng_robots->collide(&collision_data,
                            fcl::DefaultCollisionFunction<double>);
    if (collision_data.result.isCollision()) {
      assert(collision_data.result.numContacts() > 0);
      const auto &contact = collision_data.result.getContact(0);

      early_conflict.time = t * all_robots[0]->ref_dt;
      early_conflict.robot_idx_i = (size_t)contact.o1->getUserData();
      early_conflict.robot_idx_j = (size_t)contact.o2->getUserData();
      assert(early_conflict.robot_idx_i != early_conflict.robot_idx_j);
      early_conflict.robot_state_i = node_states[early_conflict.robot_idx_i];
      early_conflict.robot_state_j = node_states[early_conflict.robot_idx_j];
      std::cout << "CONFLICT at time " << t << " " << early_conflict.robot_idx_i
                << " " << early_conflict.robot_idx_j << std::endl;

      // #ifdef DBG_PRINTS
      //             std::cout << "CONFLICT at time " << t << " " <<
      //             early_conflict.robot_idx_i << " " <<
      //             early_conflict.robot_idx_j << std::endl; auto si_i =
      //             all_robots[early_conflict.robot_idx_i]->getSpaceInformation();
      //             si_i->printState(early_conflict.robot_state_i);
      //             auto si_j =
      //             all_robots[early_conflict.robot_idx_j]->getSpaceInformation();
      //             si_j->printState(early_conflict.robot_state_j);
      // #endif
      return true;
    }
  }
  return false;
}
// for heterogeneous case with the residual force
// no prioritization, only create constraints
// doesn't work with car_trailer, assumes robots are in consec.order
bool getEarliestViolations(
    const std::vector<LowLevelPlan<dynobench::Trajectory>> &solution,
    std::vector<std::string> &robot_types,
    std::map<size_t, std::vector<dynoplan::Constraint>> &constraints) {
  double max_f = 0.0981; // in Newton
  float rho;
  std::vector<size_t> involved_robots;
  size_t max_t = 0;
  for (const auto &sol : solution) {
    max_t = std::max(max_t, sol.trajectory.states.size() - 1);
  }
  Eigen::VectorXd state, state1, state2;
  std::vector<Eigen::VectorXd> states;
  for (size_t t = 0; t <= max_t; ++t) {
    states.clear();
    for (size_t robot_idx = 0; robot_idx < solution.size(); ++robot_idx) {
      if (t >= solution[robot_idx].trajectory.states.size()) {
        state = solution[robot_idx].trajectory.states.back();
      } else {
        state = solution[robot_idx].trajectory.states[t];
      }
      states.push_back(state);
    }
    for (size_t i = 0; i < solution.size(); ++i) {
      involved_robots.clear(); // for each robot
      rho = 0;                 // for each robot
      state1 = states.at(i);
      for (size_t j = 0; j < solution.size(); ++j) {
        if (i != j) { // fa for each robot coming from neighbors
          state2 = states.at(j);
          auto dist = state1 - state2;
          if (abs(dist(0)) < 0.2 && abs(dist(1)) < 0.2 && abs(dist(2)) < 1.5) {
            float input[6] = {
                static_cast<float>(dist(0)), static_cast<float>(dist(1)),
                static_cast<float>(dist(2)), static_cast<float>(dist(3)),
                static_cast<float>(dist(4)), static_cast<float>(dist(5))};
            nn_reset();
            const auto nnType = (robot_types[j] == "integrator2_3d_large_v0")
                                    ? NN_ROBOT_LARGE
                                    : NN_ROBOT_SMALL;

            nn_add_neighbor(input, nnType);
            // keep track of contributing robots
            involved_robots.push_back(j);
          }
        }
      }
      // after checking all neighbors
      const auto selfType = (robot_types[i] == "integrator2_3d_large_v0")
                                ? NN_ROBOT_LARGE
                                : NN_ROBOT_SMALL;
      const float *rhoOutput = nn_eval(selfType); // in grams
      rho = rhoOutput[0] / 1000 * 9.81;           // in Newtons
      if (rho < -max_f || rho > max_f) {
        std::cout << "fa violations" << std::endl;
        for (auto &k :
             involved_robots) { // create constraints for each involved neighbor
          constraints[k].push_back({t, states.at(k)});
        }
        // add the self robot
        constraints[i].push_back({t, states.at(i)});
        return true; // as soon as the violation happends
      }
    }
  } // time loop
  return false;
}
void createConstraintsFromConflicts(
    const Conflict &early_conflict,
    std::map<size_t, std::vector<dynoplan::Constraint>> &constraints) {
  constraints[early_conflict.robot_idx_i].push_back(
      {early_conflict.time, early_conflict.robot_state_i});
  constraints[early_conflict.robot_idx_j].push_back(
      {early_conflict.time, early_conflict.robot_state_j});
}
// assumes if residual force, then add 0 to all robots, only homogeneous robots
// for now
void export_solutions(
    const std::vector<LowLevelPlan<dynobench::Trajectory>> &solution,
    std::ofstream *out, bool residual_forse) {
  float cost = 0;
  std::string indent = "  ";
  for (auto &n : solution)
    cost += n.trajectory.cost;
  *out << "cost: " << cost << std::endl;
  *out << "motion_primitves:" << std::endl;
  for (size_t i = 0; i < solution.size(); ++i) {
    for (size_t j = 0; j < solution[i].trajectory.primitive_actions.size();
         ++j) {
      std::vector<Eigen::VectorXd> tmp_prim_state =
          solution[i].trajectory.primitive_states.at(j);
      std::vector<Eigen::VectorXd> tmp_prim_action =
          solution[i].trajectory.primitive_actions.at(j);
      *out << "  - start: " << tmp_prim_state.front().format(dynobench::FMT)
           << std::endl;
      *out << "    goal: " << tmp_prim_state.back().format(dynobench::FMT)
           << std::endl;
      *out << "    states:" << std::endl;
      for (size_t k = 0; k < tmp_prim_state.size(); ++k) {
        *out << "      - ";
        *out << tmp_prim_state.at(k).format(dynobench::FMT) << std::endl;
      }
      *out << "    actions:" << std::endl;
      for (size_t k = 0; k < tmp_prim_action.size(); ++k) {
        *out << "      - ";
        *out << tmp_prim_action.at(k).format(dynobench::FMT) << std::endl;
      }
    }
  }
  *out << "result:" << std::endl;
  for (size_t i = 0; i < solution.size(); ++i) {
    std::vector<Eigen::VectorXd> tmp_states = solution[i].trajectory.states;
    std::vector<Eigen::VectorXd> tmp_actions = solution[i].trajectory.actions;
    *out << "-" << std::endl;
    *out << indent << "states:" << std::endl;
    for (size_t j = 0; j < tmp_states.size(); ++j) {
      if (residual_forse) {
        tmp_states.at(j).conservativeResize(tmp_states.at(j).size() + 1);
        tmp_states.at(j)(tmp_states.at(j).size() - 1) = 0;
      }
      *out << indent << "  - " << tmp_states.at(j).format(dynobench::FMT)
           << std::endl;
    }
    *out << indent << "actions:" << std::endl;
    for (size_t j = 0; j < tmp_actions.size(); ++j) {
      *out << indent << "  - " << tmp_actions.at(j).format(dynobench::FMT)
           << std::endl;
    }
  }
}

void export_intermediate_solutions(
    const std::vector<LowLevelPlan<dynobench::Trajectory>> &solution,
    std::vector<std::vector<dynoplan::Constraint>> constraints,
    const Conflict &early_conflict, std::ofstream *out) {
  float cost = 0;
  for (auto &n : solution)
    cost += n.trajectory.cost;

  size_t all_constraints = 0;
  for (auto &c : constraints) {
    all_constraints += c.size(); // for each robot vector of constraints
  }
  *out << "cost: " << cost << std::endl;
  *out << "result:" << std::endl;
  for (size_t i = 0; i < solution.size(); ++i) {
    std::vector<Eigen::VectorXd> tmp_states = solution[i].trajectory.states;
    std::vector<Eigen::VectorXd> tmp_actions = solution[i].trajectory.actions;
    *out << "  - states:" << std::endl;
    for (size_t j = 0; j < tmp_states.size(); ++j) {
      *out << "      - ";
      *out << tmp_states.at(j).format(dynobench::FMT) << std::endl;
    }
    *out << "    actions:" << std::endl;
    for (size_t j = 0; j < tmp_actions.size(); ++j) {
      *out << "      - ";
      *out << tmp_actions.at(j).format(dynobench::FMT) << std::endl;
    }
  }
  // constraints of the node
  *out << "constraints: " << all_constraints << std::endl;
  // write conflicts
  *out << "conflict:" << std::endl;
  *out << "    time:" << std::endl;
  *out << "      - ";
  *out << early_conflict.time << std::endl;
  // only one state can be used for visualization
  *out << "    states:" << std::endl;
  *out << "      - ";
  *out << early_conflict.robot_state_i.format(dynobench::FMT) << std::endl;
  *out << "      - ";
  *out << early_conflict.robot_state_j.format(dynobench::FMT) << std::endl;
}

void export_constraints(
    const std::vector<std::vector<dynoplan::Constraint>> &final_constraints,
    std::ofstream *out) {
  *out << "constraints:" << std::endl;
  // for (const auto& c : final_constraints){
  for (size_t j = 0; j < final_constraints.size(); j++) {
    if (final_constraints[j].size() > 0) {
      const auto &c = final_constraints[j];
      *out << "  - robot_id: " << j << std::endl;
      *out << "    states:" << std::endl;
      for (size_t i = 0; i < c.size(); i++) {
        if (c[i].constrained_state.size() > 0)
          *out << "      - ";
        *out << c[i].constrained_state.format(dynobench::FMT) << std::endl;
      }
      *out << "    time:" << std::endl;
      for (size_t i = 0; i < c.size(); i++) {
        if (c[i].constrained_state.size() > 0)
          *out << "      - ";
        *out << c[i].time << std::endl;
      }
    }
  }
}

// meta-robot related additions

// Convert Obstacle struct to YAML node
YAML::Node obstacle_to_yaml(const Obstacle &obs) {
  YAML::Node node;
  node["center"] = obs.center;
  node["size"] = obs.size;
  node["type"] = obs.type;
  node["octomap_file"] = obs.octomap_file;
  return node;
}

void get_moving_obstacle_env(YAML::Node &env,
                             // const std::string &initial_guess_file,
                             MultiRobotTrajectory init_guess_multi_robot,
                             const std::string &out_file,
                             std::unordered_set<size_t> &cluster,
                             bool moving_obstacles, bool residual_force) {

  double radius = 0.1;                                   // radius
  Eigen::Vector3d radii = Eigen::Vector3d(.12, .12, .3); // from tro paper
  std::string type = "sphere";
  // YAML::Node env = YAML::LoadFile(env_file);
  const auto &env_min = env["environment"]["min"];
  const auto &env_max = env["environment"]["max"];
  // data
  YAML::Node data;
  data["environment"]["max"] = env_max;
  data["environment"]["min"] = env_min;
  // read the result
  // YAML::Node initial_guess = YAML::LoadFile(initial_guess_file);
  // size_t num_robots = initial_guess["result"].size();
  size_t num_robots = init_guess_multi_robot.trajectories.size();
  for (size_t i = 0; i < num_robots; i++) {
    if (cluster.find(i) != cluster.end()) { // robots that are within cluster
      YAML::Node robot_node;
      if (residual_force) {
        // augment the start/goal
        env["robots"][i]["start"].push_back(0);
        env["robots"][i]["goal"].push_back(0);
        robot_node["type"] =
            "integrator2_3d_res_v0"; // env["robots"][i]["type"];
      } else
        robot_node["type"] = env["robots"][i]["type"];

      robot_node["start"] = env["robots"][i]["start"];
      robot_node["goal"] = env["robots"][i]["goal"];
      data["robots"].push_back(robot_node);
    }
  }
  // static obstacles
  if (env["environment"]["obstacles"]) {
    for (const auto &obs : env["environment"]["obstacles"]) {
      YAML::Node obs_node;
      std::string octomap_filename;
      if (obs["type"].as<std::string>() == "octomap") {
        obs_node["center"] = YAML::Node(YAML::NodeType::Sequence); // Empty list
        obs_node["size"] = YAML::Node(YAML::NodeType::Sequence);   // Empty list
        obs_node["octomap_file"] = obs["octomap_file"];
        obs_node["type"] = "octomap";
      } else {
        obs_node["center"] = obs["center"];
        obs_node["size"] = obs["size"];
        obs_node["type"] = obs["type"];
      }
      data["environment"]["obstacles"].push_back(
          obs_node); // if no moving obs, but clusters
    }
  }

  if (moving_obstacles) {
    size_t max_t = 0;
    size_t index = 0;
    for (const auto &traj : init_guess_multi_robot.trajectories) {
      max_t = std::max(
          max_t, traj.states.size() -
                     1); // among all paths, optimization needs the longest traj
      ++index;
    }
    YAML::Node moving_obstacles_node; // for all robots
    Eigen::VectorXd state;
    std::vector<Obstacle> moving_obs_per_time;
    std::vector<std::vector<Obstacle>> moving_obs;
    std::cout << "MAXT: " << max_t << std::endl;
    for (size_t t = 0; t <= max_t; ++t) {
      moving_obs_per_time.clear();
      for (size_t i = 0; i < num_robots; i++) {
        if (cluster.find(i) == cluster.end()) {
          if (t >= init_guess_multi_robot.trajectories.at(i).states.size()) {
            state = init_guess_multi_robot.trajectories.at(i).states.back();
          } else {
            state = init_guess_multi_robot.trajectories.at(i).states[t];
          }
          // into vector
          Obstacle obs;
          obs.center = {state(0), state(1), state(2)};
          obs.size = {radius}; // {radii(0), radii(1), radii(2)};
          obs.type = "sphere"; // "ellipsoid";
          moving_obs_per_time.push_back({obs});
        }
      }
      // static obstacles per timestamp
      for (const auto &obs : env["environment"]["obstacles"]) {
        Obstacle octomap_obs;
        std::string octomap_filename;
        if (obs["type"].as<std::string>() == "octomap") {
          octomap_obs.center = {};
          octomap_obs.size = {};
          octomap_obs.octomap_file = obs["octomap_file"].as<std::string>();
          octomap_obs.type = "octomap";
          moving_obs_per_time.push_back({octomap_obs});
        }
      }
      moving_obs.push_back(moving_obs_per_time);
    }
    for (const auto &obs_list : moving_obs) {
      YAML::Node yaml_obs_list;
      for (const auto &obs : obs_list) {
        yaml_obs_list.push_back(obstacle_to_yaml(obs));
      }
      data["environment"]["moving_obstacles"].push_back(yaml_obs_list);
    }
  }

  // Write YAML node to file
  std::ofstream fout(out_file);
  fout << data;
  fout.close();
}

// for moving obstacles META-robot. Joint robots become "integrator2_3d_res_v0",
// and start/goal augments moving obstacles have Ellipsoid shape
void get_moving_obstacle(const std::string &env_file,
                         // const std::string &initial_guess_file,
                         MultiRobotTrajectory init_guess_multi_robot,
                         const std::string &out_file,
                         std::unordered_set<size_t> &cluster,
                         bool moving_obstacles, bool residual_force) {
  YAML::Node env = YAML::LoadFile(env_file);
  return get_moving_obstacle_env(env, init_guess_multi_robot, out_file,
                                 cluster);
}
// for meta-robot clustering, it counts how many times each robot collide with
// other members
bool getConflicts(
    // const std::vector<LowLevelPlan<dynobench::Trajectory>>& solution,
    const std::vector<dynobench::Trajectory> &multi_robot_trajectories,
    const std::vector<std::shared_ptr<dynobench::Model_robot>> &all_robots,
    std::shared_ptr<fcl::BroadPhaseCollisionManagerd> col_mng_robots,
    std::vector<fcl::CollisionObjectd *> &robot_objs,
    std::vector<std::vector<int>> &conflict_matrix) {
  bool collision = false;
  size_t max_t = 0;
  for (const auto &traj : multi_robot_trajectories) {
    max_t = std::max(max_t, traj.states.size() - 1);
  }
  Eigen::VectorXd node_state;
  std::vector<Eigen::VectorXd> node_states;
  for (size_t t = 0; t <= max_t; ++t) {
    node_states.clear();
    size_t robot_idx = 0;
    size_t obj_idx = 0;
    std::vector<fcl::Transform3d> ts_data;
    for (auto &robot : all_robots) {
      if (t >= multi_robot_trajectories.at(robot_idx).states.size()) {
        node_state = multi_robot_trajectories.at(robot_idx).states.back();
      } else {
        node_state = multi_robot_trajectories.at(robot_idx).states[t];
      }
      node_states.push_back(node_state);
      std::vector<fcl::Transform3d> tmp_ts(1);
      if (robot->name == "car_with_trailers") {
        tmp_ts.resize(2);
      }
      robot->transformation_collision_geometries(node_state, tmp_ts);
      ts_data.insert(ts_data.end(), tmp_ts.begin(), tmp_ts.end());
      ++robot_idx;
    }
    for (size_t i = 0; i < ts_data.size(); i++) {
      fcl::Transform3d &transform = ts_data[i];
      robot_objs[obj_idx]->setTranslation(transform.translation());
      robot_objs[obj_idx]->setRotation(transform.rotation());
      robot_objs[obj_idx]->computeAABB();
      ++obj_idx;
    }
    col_mng_robots->update(robot_objs);
    fcl::DefaultCollisionData<double> collision_data;
    col_mng_robots->collide(&collision_data,
                            fcl::DefaultCollisionFunction<double>);
    if (collision_data.result.isCollision()) {
      assert(collision_data.result.numContacts() > 0);
      collision = true;
      for (size_t k = 0; k < collision_data.result.numContacts();
           k++) { // not 2 only ?
        const auto &contact = collision_data.result.getContact(k);
        auto idx_i = (size_t)contact.o1->getUserData();
        auto idx_j = (size_t)contact.o2->getUserData();
        assert(idx_i != idx_j);
        // get lower-triangle matrix
        if (idx_i >= idx_j)
          conflict_matrix[idx_i][idx_j] += 1;
        else
          conflict_matrix[idx_j][idx_i] += 1;
        // std::cout << "(Opt) CONFLICT at time " << t << " " << idx_i << " " <<
        // idx_j << std::endl;
      }
    }
  }
  // get the max element
  // int max_collision = 0;
  // for (const auto& cft : conflict_matrix) {
  //     int localMax = *std::max_element(cft.begin(), cft.end());
  //     if (localMax > max_collision) {
  //         max_collision = localMax;
  //     }
  // }
  // return max_collision;
  return collision;
}

// hard-coded, check j, k
void export_solutions_joint(
    const std::vector<LowLevelPlan<dynobench::Trajectory>> &solution,
    std::ofstream *out) {
  float cost = 0;
  size_t max_t = 0;
  size_t max_a = 0;
  int k = 6 + 1; // for the state, residual force + 1
  int j = 3;     // for the actions
  Eigen::VectorXd tmp_state(k * 2);
  Eigen::VectorXd tmp_action(j * 2);
  std::vector<Eigen::VectorXd> joint_states;
  std::vector<Eigen::VectorXd> joint_actions;

  for (auto &n : solution) {
    cost += n.trajectory.cost;
    max_t = std::max(max_t, n.trajectory.states.size() - 1);
    max_a = std::max(max_a, n.trajectory.actions.size() - 1);
  }

  *out << "cost: " << cost << std::endl;
  *out << "result:" << std::endl;
  *out << "  - states:" << std::endl;
  for (size_t t = 0; t <= max_t; ++t) {
    for (size_t i = 0; i < solution.size(); ++i) { // for each robot
      if (t >= solution[i].trajectory.states.size()) {
        tmp_state.segment(i * k, k - 1) =
            solution[i].trajectory.states.back(); // res
        // tmp_state.segment(i*k,k) = solution[i].trajectory.states.back();
      } else {
        tmp_state.segment(i * k, k - 1) =
            solution[i].trajectory.states[t]; // res
        // tmp_state.segment(i*k,k) = solution[i].trajectory.states[t];
      }
      tmp_state((i + 1) * k - 1) = 0; // for the residual force
    }
    joint_states.push_back(tmp_state);
  }
  // write to file
  for (size_t j = 0; j < joint_states.size(); ++j) {
    *out << "      - ";
    *out << joint_states.at(j).format(dynobench::FMT) << std::endl;
  }
  *out << "    actions:" << std::endl;
  for (size_t t = 0; t <= max_a; ++t) {
    for (size_t i = 0; i < solution.size(); ++i) {
      if (t >= solution[i].trajectory.actions.size()) {
        tmp_action.segment(i * j, j) = solution[i].trajectory.actions.back();
      } else {
        tmp_action.segment(i * j, j) = solution[i].trajectory.actions[t];
      }
    }
    joint_actions.push_back(tmp_action);
  }
  // write to file
  for (size_t j = 0; j < joint_actions.size(); ++j) {
    *out << "      - ";
    *out << joint_actions.at(j).format(dynobench::FMT) << std::endl;
  }
}

void extract_motion_primitives(
    dynobench::Problem &problem, MultiRobotTrajectory &multi_robot_opt_out,
    std::map<std::string, std::vector<dynoplan::Motion>> &robot_motions,
    const std::vector<std::shared_ptr<dynobench::Model_robot>> &all_robots,
    int len) {

  size_t robot_idx = 0;
  for (auto robot_trajectory : multi_robot_opt_out.trajectories) {
    int idx = 0;
    int total_actions = robot_trajectory.actions.size();
    while (idx < total_actions) {
      int num_actions = rand() % 11 + len; // without len it's < 10
      num_actions = std::min(num_actions, (total_actions - idx));
      // make sure the left motion primitive has >= 5 length
      if ((total_actions - (idx + num_actions)) < 5 &&
          idx + num_actions < total_actions) {
        num_actions = total_actions - idx;
      }
      dynobench::Trajectory new_trajectory;
      std::vector action_vector(robot_trajectory.actions.begin() + idx,
                                robot_trajectory.actions.begin() + idx +
                                    num_actions);
      new_trajectory.actions = action_vector;

      std::vector state_vector(robot_trajectory.states.begin() + idx,
                               robot_trajectory.states.begin() + idx +
                                   num_actions + 1);
      // rebase
      Eigen::VectorXd first_state = state_vector[0];
      for (auto &state : state_vector) {
        state[0] -= first_state[0];
        state[1] -= first_state[1];
        if (!all_robots[robot_idx]->is_2d)
          state[2] -= first_state[2];
      }
      new_trajectory.states = state_vector;
      dynoplan::Motion new_motion;
      traj_to_motion(new_trajectory, *all_robots[robot_idx], new_motion,
                     /*check collision*/ true);
      new_motion.traj = new_trajectory;
      new_motion.idx = robot_motions[problem.robotTypes[robot_idx]].size();
      robot_motions[problem.robotTypes[robot_idx]].push_back(
          std::move(new_motion));
      idx += num_actions;
    }
    robot_motions[problem.robotTypes[robot_idx]].shrink_to_fit();
    robot_idx++;
  }
}
// #include <boost/heap/d_ary_heap.hpp>

// // Define your element type (e.g., HighLevelNode)
// struct HighLevelNode {
//     // Define your element properties and methods
// };

// // Define your binary heap type
// typedef boost::heap::d_ary_heap<HighLevelNode, boost::heap::arity<2>,
// boost::heap::mutable_<true>> BinaryHeap;

// // Access the handle_type typedef
// typedef BinaryHeap::handle_type HandleType;

// Now you can use HandleType to declare variables that represent handles to
// elements in the binary heap
