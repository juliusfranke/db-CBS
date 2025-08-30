#pragma once
#include "db_cbs.hpp"
#include <Eigen/Dense>
#include <string>
#include <yaml-cpp/yaml.h>

std::vector<Result> db_ecbs(YAML::Node &env, std::string outputFile,
                            std::string optimizationFile, YAML::Node &cfg,
                            double timeLimit, double timeLimitdbeCBS);
