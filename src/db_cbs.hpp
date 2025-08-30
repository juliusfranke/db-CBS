#pragma once
#include "dynobench/multirobot_trajectory.hpp"
#include <Eigen/Dense>
#include <array>
#include <string>
#include <tuple>
#include <vector>
#include <yaml-cpp/yaml.h>

struct Result {
  MultiRobotTrajectory discrete;
  MultiRobotTrajectory optimized;
  double runtime;
  double delta;
};

std::vector<Result> db_cbs(YAML::Node &env, std::string outputFile,
                           std::string optimizationFile, YAML::Node &cfg,
                           double timeLimit, double timeLimitdbCBS);
