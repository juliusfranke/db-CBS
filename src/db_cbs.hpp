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

std::vector<Result> db_cbs(YAML::Node &inputFile, std::string outputFile,
                           std::string optimizationFile, YAML::Node &cfgFile,
                           double timeLimit, double timeLimitdbCBS);
