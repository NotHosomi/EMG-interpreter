#pragma once

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

//#define PRINT_FF
//#define PRINT_DLOSS
//#define PRINT_BP
//#define PRINT_UPDATES

namespace Common
{
	double sigmoid(double x);
	double dsigmoid(double x);
	double tangent(double x);
	double dtangent(double x);

	VectorXd loss(VectorXd outputs, VectorXd targets);
	VectorXd dloss(VectorXd outputs, VectorXd targets);
	double accuracy(VectorXd outputs, VectorXd targets);
};

