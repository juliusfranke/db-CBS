#include "dynobench/multirobot_trajectory.hpp"
#include <Eigen/Dense>
#include <string>
#include <yaml-cpp/yaml.h>


MultiRobotTrajectory db_cbs(YAML::Node &inputFile, std::string outputFile,
                   std::string optimizationFile, YAML::Node &cfgFile,
                   double timeLimit);
