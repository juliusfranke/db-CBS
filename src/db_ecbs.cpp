#include <iostream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iterator>
#include <yaml-cpp/yaml.h>
#include <filesystem>
// BOOST
#include <boost/program_options.hpp>
#include <boost/program_options.hpp>
#include <boost/heap/d_ary_heap.hpp>
// DYNOPLAN
#include <dynoplan/optimization/ocp.hpp>
#include "dynoplan/optimization/multirobot_optimization.hpp"
#include "dynoplan/tdbastar/tdbastar.hpp"
#include "dynoplan/tdbastar/tdbastar_epsilon.hpp"
#include "dynoplan/tdbastar/planresult.hpp"


// DYNOBENCH
#include "dynobench/general_utils.hpp"
#include "dynobench/robot_models_base.hpp"

#include "robots.h"
#include "robotStatePropagator.hpp"
#include "fclStateValidityChecker.hpp"
#include "fcl/broadphase/broadphase_collision_manager.h"
#include <fcl/fcl.h>
#include "dbcbs_utils.hpp"

using namespace dynoplan;
namespace fs = std::filesystem;

#define DYNOBENCH_BASE "../dynoplan/dynobench/"
#define REBUILT_FOCAL_LIST
#define CHECK_FOCAL_LIST

int main(int argc, char* argv[]) {
    
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description desc("Allowed options");
    std::string inputFile;
    std::string outputFile;
    std::string optimizationFile;
    std::string cfgFile;
    double timeLimit;

    desc.add_options()
      ("help", "produce help message")
      ("input,i", po::value<std::string>(&inputFile)->required(), "input file (yaml)")
      ("output,o", po::value<std::string>(&outputFile)->required(), "output file (yaml)")
      ("optimization,opt", po::value<std::string>(&optimizationFile)->required(), "optimization file (yaml)")
      ("cfg,c", po::value<std::string>(&cfgFile)->required(), "configuration file (yaml)")
      ("time_limit,t", po::value<double>(&timeLimit)->required(), "time limit for search");

    try {
      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help") != 0u) {
        std::cout << desc << "\n";
        return 0;
      }
    } catch (po::error& e) {
      std::cerr << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return 1;
    }
    YAML::Node cfg = YAML::LoadFile(cfgFile);
    // cfg = cfg["db-ecbs"]["default"];
    float alpha = cfg["alpha"].as<float>();
    bool filter_duplicates = cfg["filter_duplicates"].as<bool>();
    fs::path output_path(outputFile);
    std::string output_folder = output_path.parent_path().string();
    bool save_search_video = false;
    bool save_expanded_trajs = cfg["save_expanded_trajs"].as<bool>();
    std::string conflicts_folder = output_folder + "/conflicts";
    // tdbstar options
    Options_tdbastar options_tdbastar;
    options_tdbastar.outFile = outputFile;
    options_tdbastar.search_timelimit = timeLimit;
    options_tdbastar.cost_delta_factor = 0;
    options_tdbastar.fix_seed = 1;
    options_tdbastar.max_motions = cfg["num_primitives_0"].as<size_t>();
    options_tdbastar.w = cfg["suboptimality_factor"].as<float>(); 
    options_tdbastar.rewire = cfg["rewire"].as<bool>();
    options_tdbastar.always_add_node = cfg["always_add_node"].as<bool>();
    // options_tdbastar.max_expands = 200;
    // tdbastar problem
    dynobench::Problem problem(inputFile);
    dynobench::Problem problem_original(inputFile);
    std::string models_base_path = DYNOBENCH_BASE + std::string("models/");
    problem.models_base_path = models_base_path;
    Out_info_tdb out_tdb;
    std::cout << "*** options_tdbastar ***" << std::endl;
    options_tdbastar.print(std::cout);
    std::cout << "***" << std::endl;

    // load problem description
    YAML::Node env = YAML::LoadFile(inputFile);
    std::vector<std::shared_ptr<fcl::CollisionGeometryd>> collision_geometries;
    const auto &env_min = env["environment"]["min"];
    const auto &env_max = env["environment"]["max"];
    ob::RealVectorBounds position_bounds(env_min.size());
    for (size_t i = 0; i < env_min.size(); ++i) {
        position_bounds.setLow(i, env_min[i].as<double>());
        position_bounds.setHigh(i, env_max[i].as<double>());
    }

    fcl::AABBf workspace_aabb(
        fcl::Vector3f(env_min[0].as<double>(),
        env_min[1].as<double>(),-1),
        fcl::Vector3f(env_max[0].as<double>(), env_max[1].as<double>(), 1));

    std::vector<std::shared_ptr<dynobench::Model_robot>> robots;
    // std::vector<dynobench::Trajectory> ll_trajs;
    std::string motionsFile;
    std::vector<std::string> all_motionsFile;
    for (const auto &robotType : problem.robotTypes){
        std::shared_ptr<dynobench::Model_robot> robot = dynobench::robot_factory(
                (problem.models_base_path + robotType + ".yaml").c_str(), problem.p_lb, problem.p_ub);
        robots.push_back(robot);
        if (robotType == "unicycle1_v0" || robotType == "unicycle1_sphere_v0"){
            motionsFile = "../new_format_motions/unicycle1_v0/unicycle1_v0.msgpack";
        } else if (robotType == "unicycle2_v0"){
            motionsFile = "../new_format_motions/unicycle2_v0/unicycle2_v0.msgpack";
        } else if (robotType == "car1_v0"){
            motionsFile = "../new_format_motions/car1_v0/car1_v0.msgpack";
        } else if (robotType == "integrator2_2d_v0"){
            motionsFile = "../new_format_motions/integrator2_2d_v0/integrator2_2d_v0.msgpack";
        } else if (robotType == "integrator2_3d_v0"){
            motionsFile = "../new_format_motions/integrator2_3d_v0/long_50/integrator2_3d_v0.bin.im.bin.sp.bin";
        } else{
            throw std::runtime_error("Unknown motion filename for this robottype!");
        }
        all_motionsFile.push_back(motionsFile);
    }
    std::map<std::string, std::vector<Motion>> robot_motions;
    // allocate data for conflict checking, check for conflicts
    std::vector<fcl::CollisionObjectd*> robot_objs;
    std::shared_ptr<fcl::BroadPhaseCollisionManagerd> col_mng_robots;
    col_mng_robots = std::make_shared<fcl::DynamicAABBTreeCollisionManagerd>();
    size_t col_geom_id = 0;
    col_mng_robots->setup();
    size_t i = 0;
    for (const auto &robot : robots){
        collision_geometries.insert(collision_geometries.end(), 
                              robot->collision_geometries.begin(), robot->collision_geometries.end());
        auto robot_obj = new fcl::CollisionObject(collision_geometries[col_geom_id]);
        collision_geometries[col_geom_id]->setUserData((void*)i);
        robot_objs.push_back(robot_obj);
        if (robot_motions.find(problem.robotTypes[i]) == robot_motions.end()){
            options_tdbastar.motionsFile = all_motionsFile[i];
            load_motion_primitives_new(options_tdbastar.motionsFile, *robot, robot_motions[problem.robotTypes[i]], 
                                       options_tdbastar.max_motions,
                                       options_tdbastar.cut_actions, false, options_tdbastar.check_cols);
        }
        if (robot->name == "car_with_trailers") {
          col_geom_id++;
          auto robot_obj = new fcl::CollisionObject(collision_geometries[col_geom_id]);
          collision_geometries[col_geom_id]->setUserData((void*)i); // for the trailer
          robot_objs.push_back(robot_obj);
        }
        
        col_geom_id++;
        i++;
    }
    col_mng_robots->registerObjects(robot_objs);
    // Heuristic computation
    size_t robot_id = 0;
    std::vector<ompl::NearestNeighbors<std::shared_ptr<AStarNode>>*> heuristics(robots.size(), nullptr);
    std::vector<dynobench::Trajectory> expanded_trajs_tmp;
    std::vector<LowLevelPlan<dynobench::Trajectory>> tmp_solutions(robots.size());
    if (cfg["heuristic1"].as<std::string>() == "reverse-search"){
      std::cout << "Running the reverse search" << std::endl;
      options_tdbastar.delta = cfg["heuristic1_delta"].as<float>();
      for (const auto &robot : robots){
        // start to inf for the reverse search
        LowLevelPlan<dynobench::Trajectory> tmp_solution;
        problem.starts[robot_id].head(robot->translation_invariance).setConstant(std::sqrt(std::numeric_limits<double>::max()));
        Eigen::VectorXd tmp_state = problem.starts[robot_id];
        problem.starts[robot_id] = problem.goals[robot_id];
        problem.goals[robot_id] = tmp_state;
        expanded_trajs_tmp.clear();
        options_tdbastar.motions_ptr = &robot_motions[problem.robotTypes[robot_id]]; 
        tdbastar_epsilon(problem, options_tdbastar, 
                tmp_solution.trajectory,/*constraints*/{},
                out_tdb, robot_id,/*reverse_search*/true, 
                expanded_trajs_tmp, tmp_solutions, robot_motions,
                robots, col_mng_robots, robot_objs,
                nullptr, &heuristics[robot_id], options_tdbastar.w);
        std::cout << "computed heuristic with " << heuristics[robot_id]->size() << " entries." << std::endl;
        // if (save_expanded_trajs){
        //   std::ofstream out3(output_folder + "/expanded_trajs_reverse_" + std::to_string(robot_id) + ".yaml");
        //   std::cout << "expanded trajs reverse size: " << expanded_trajs_tmp.size() << std::endl;
        //   out3 << "trajs:" << std::endl;
        //   for (auto i = 0; i < expanded_trajs_tmp.size(); i += 100){
        //     auto traj = expanded_trajs_tmp.at(i);
        //     out3 << "  - " << std::endl;
        //     traj.to_yaml_format(out3, "    ");
        //   }
        // }
        robot_id++;
      }
    }
    // return 0;
    if (save_search_video){
      std::cout << "***Going to save all intermediate solutions with conflicts!***" << std::endl;
      if (!fs::exists(conflicts_folder)) {
        fs::create_directory(conflicts_folder);
      }
    }
    bool solved_db = false;
    std::cout << "Running the main loop" << std::endl;
    // main loop
    problem.starts = problem_original.starts;
    problem.goals = problem_original.goals;
    options_tdbastar.delta = cfg["delta_0"].as<float>();
    for (size_t iteration = 0; ; ++iteration) {
      std::cout << "iteration: " << iteration << std::endl;
      if (iteration > 0) {
        if (solved_db) {
            options_tdbastar.delta *= cfg["delta_0"].as<float>();
        } else {
            options_tdbastar.delta *= 0.99;
        }
        options_tdbastar.max_motions *= cfg["num_primitives_rate"].as<float>();
        options_tdbastar.max_motions = std::min<size_t>(options_tdbastar.max_motions, 1e6);
      }
      // disable/enable motions 
      for (auto& iter : robot_motions) {
          for (size_t i = 0; i < problem.robotTypes.size(); ++i) {
              if (iter.first == problem.robotTypes[i]) {
                  disable_motions(robots[i], problem.robotTypes[i], options_tdbastar.delta, filter_duplicates, alpha, 
                                  options_tdbastar.max_motions, iter.second);
                  break;
              }
          }
      }
      solved_db = false;
      HighLevelNodeFocal start;
      start.solution.resize(env["robots"].size());
      start.constraints.resize(env["robots"].size());
      start.result.resize(env["robots"].size());
      start.cost = 0;
      start.id = 0;
      start.LB = 0;
      bool start_node_valid = true;
      robot_id = 0;
      int id = 1;
      std::cout << "Node ID is " << id << ", root" << std::endl;
      for (const auto &robot : robots){
        expanded_trajs_tmp.clear();
        options_tdbastar.motions_ptr = &robot_motions[problem.robotTypes[robot_id]]; 
        tdbastar_epsilon(problem, options_tdbastar, 
                start.solution[robot_id].trajectory, start.constraints[robot_id],
                out_tdb, robot_id,/*reverse_search*/false, 
                expanded_trajs_tmp, start.solution, robot_motions,
                robots, col_mng_robots, robot_objs,
                heuristics[robot_id], nullptr, options_tdbastar.w);
        if(!out_tdb.solved){
          std::cout << "Couldn't find initial solution for robot " << robot_id << "." << std::endl;
          // if (save_expanded_trajs){
          //   std::ofstream out2(output_folder + "/expanded_trajs_fail.yaml");
          //   out2 << "trajs:" << std::endl;
          //   for (auto i = 0; i < 200; i++){
          //     auto traj = expanded_trajs_tmp.at(i);
          //     out2 << "  - " << std::endl;
          //     traj.to_yaml_format(out2, "    ");
          //   }
          // }
          start_node_valid = false;
          break;
        }
        // if (save_expanded_trajs){
        //   std::ofstream out2(output_folder + "/root_" + std::to_string(robot_id) + ".yaml");
        //   out2 << "trajs:" << std::endl;
        //   for (auto i = 0; i < expanded_trajs_tmp.size(); i++){
        //     auto traj = expanded_trajs_tmp.at(i);
        //     out2 << "  - " << std::endl;
        //     traj.to_yaml_format(out2, "    ");
        //   }
        // }
        start.cost += start.solution[robot_id].trajectory.cost;
        start.LB += start.solution[robot_id].trajectory.fmin;
        robot_id++;
      }
      start.focalHeuristic = highLevelfocalHeuristicState(start.solution, robots, col_mng_robots, robot_objs); 
      // std::ofstream out(outputFile);
      // export_solutions(start.solution, &out);
      // return 0;
      if (!start_node_valid) {
            continue;
      }
      
      openset_t open;
      focalset_t focal;

      auto handle = open.push(start);
      (*handle).handle = handle;
      focal.push(handle);
      
      
      size_t expands = 0;
      double best_cost = (*handle).cost;
      
      while (!open.empty()){
#ifdef REBUILT_FOCAL_LIST 
          focal.clear();
          double LB = open.top().LB;
          auto iter = open.ordered_begin();
          auto iterEnd = open.ordered_end();
          for (; iter != iterEnd; ++iter) {
            auto cost = (*iter).cost;
            if (cost <= LB * options_tdbastar.w) {
              const HighLevelNodeFocal& n = *iter;
              focal.push(n.handle);
            }
            else {
              break;
            }
          }
#else 
        {
          double oldbest_best_cost = best_cost;
          best_cost = open.top().cost;
          if (best_cost > oldbest_best_cost) {
            auto iter = open.ordered_begin();
            auto iterEnd = open.ordered_end();
            for (; iter != iterEnd; ++iter) {
              auto cost = (*iter).cost;
              if (cost > oldbest_best_cost * options_tdbastar.w && cost <= best_cost * options_tdbastar.w) { // check, LB ?
                const HighLevelNodeFocal& n = *iter;
                focal.push(n.handle);
              }
              if (cost > best_cost * options_tdbastar.w) {
                break;
              }
            }
          }
        }
#endif
#ifdef CHECK_FOCAL_LIST 
          bool mismatch = false;
          auto LB_test = open.top().LB; // ? cost in Wolfgang's code
          auto iter_test = open.ordered_begin();
          auto iterEnd_test = open.ordered_end();
          for (; iter_test != iterEnd_test; ++iter_test) {
            const auto& node_test = *iter_test;
            auto cost_test = node_test.cost;
            if (cost_test <= LB_test * options_tdbastar.w) {
              if (std::find(focal.begin(), focal.end(), node_test.handle) ==
                  focal.end()) {
                std::cout << "focal misses some nodes " << std::endl;
                mismatch = true;
              }

            } else {
              if (std::find(focal.begin(), focal.end(), node_test.handle) !=
                  focal.end()) {
                std::cout << "focalSet shouldn't have some nodes "  << std::endl;
                mismatch = true;
              }
            }
          }
          assert(!mismatch);
#endif
        std::cout << "focal set size: " << focal.size() << std::endl;
        std::cout << "cost bound: " << LB * options_tdbastar.w << std::endl;
        auto current_handle = focal.top();
        HighLevelNodeFocal P = *current_handle;
        std::cout << "high-level node focalHeuristic: " << P.focalHeuristic << std::endl;
        focal.pop();
        open.erase(current_handle);

        Conflict inter_robot_conflict;
        if (!getEarliestConflict(P.solution, robots, col_mng_robots, robot_objs, inter_robot_conflict)){
            solved_db = true;
            std::cout << "Final solution!" << std::endl; 
            create_dir_if_necessary(outputFile);
            std::ofstream out(outputFile);
            export_solutions(P.solution, &out);
            // get motion_primitives_plot
            // if (save_expanded_trajs){
            //   std::ofstream out2(output_folder + "/expanded_trajs.yaml");
            //   out2 << "trajs:" << std::endl;
            //   for (auto i = 0; i < expanded_trajs_tmp.size(); i+=50){
            //     auto traj = expanded_trajs_tmp.at(i);
            //     out2 << "  - " << std::endl;
            //     traj.to_yaml_format(out2, "    ");
            //   }
            // }
            return 0;
            bool sum_robot_cost = true;
            bool feasible = execute_optimizationMultiRobot(inputFile,
                                          outputFile, 
                                          optimizationFile,
                                          DYNOBENCH_BASE,
                                          sum_robot_cost);

            if (feasible) {
              std::ofstream fout(optimizationFile, std::ios::app); 
              fout << "  nodes: " << id << std::endl;
              return 0;
            }
            break;
        }
        ++expands;
        if (expands % 100 == 0) {
         std::cout << "HL expanded: " << expands << " open: " << open.size() << " cost " << P.cost << " conflict at " << inter_robot_conflict.time << std::endl;
        }

        std::map<size_t, std::vector<Constraint>> constraints;
        createConstraintsFromConflicts(inter_robot_conflict, constraints);
        if(save_search_video){
          // get the plot of high-level node solution with conflicts
          auto filename = conflicts_folder + "/" + std::to_string(P.id) + ".yaml";
          std::cout << filename << std::endl;
          std::ofstream int_out(filename);
          export_intermediate_solutions(P.solution, P.constraints, inter_robot_conflict, &int_out);
        }
        
        for (const auto& c : constraints){
          id++;
          HighLevelNodeFocal newNode = P;
          size_t tmp_robot_id = c.first;
          newNode.id = id;
          std::cout << "Node ID is " << id << std::endl;
          newNode.constraints[tmp_robot_id].insert(newNode.constraints[tmp_robot_id].end(), c.second.begin(), c.second.end());
          newNode.cost -= newNode.solution[tmp_robot_id].trajectory.cost;
          newNode.LB -= newNode.solution[tmp_robot_id].trajectory.fmin;
          Out_info_tdb tmp_out_tdb; 
          expanded_trajs_tmp.clear();
          options_tdbastar.motions_ptr = &robot_motions[problem.robotTypes[tmp_robot_id]]; 
          tdbastar_epsilon(problem, options_tdbastar, 
                newNode.solution[tmp_robot_id].trajectory, newNode.constraints[tmp_robot_id],
                tmp_out_tdb, tmp_robot_id, /*reverse_search*/false, 
                expanded_trajs_tmp, newNode.solution, robot_motions,
                robots, col_mng_robots, robot_objs,
                heuristics[tmp_robot_id], nullptr, options_tdbastar.w, /*run_focal_heuristic*/true);
          if (tmp_out_tdb.solved){
              if (save_expanded_trajs){
                std::ofstream out2(output_folder + "/inter_" + std::to_string(tmp_robot_id) + ".yaml");
                out2 << "trajs:" << std::endl;
                for (auto i = 0; i < 200; i++){
                  auto traj = expanded_trajs_tmp.at(i);
                  out2 << "  - " << std::endl;
                  traj.to_yaml_format(out2, "    ");
                }
              }
              newNode.cost += newNode.solution[tmp_robot_id].trajectory.cost;
              newNode.LB += newNode.solution[tmp_robot_id].trajectory.fmin;
              newNode.focalHeuristic = highLevelfocalHeuristicState(newNode.solution, robots, col_mng_robots, robot_objs); 
              std::cout << "New node solution cost:  " << newNode.solution[tmp_robot_id].trajectory.cost << std::endl;
              std::cout << "New node cost: " << newNode.cost << " New node LB: " << newNode.LB << std::endl;
              std::cout << "New node focal heuristic: " << newNode.focalHeuristic << std::endl;
              auto handle = open.push(newNode);
              (*handle).handle = handle;
              // if (newNode.cost <= best_cost * options_tdbastar.w)
              if (newNode.cost <= open.top().LB * options_tdbastar.w){ 
                focal.push(handle);
              }
          }
          // id++;

        } 

      } 

    } 
   
  return 0;
}
