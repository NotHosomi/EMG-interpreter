#pragma once

#include <vector>
#include <Eigen/Dense>
#include "Common.h"

using namespace Eigen;
using namespace Common;

class DenseLayer
{
public:
	DenseLayer(int input_size, int output_size, double alpha);

	VectorXd feedForward(VectorXd x_t);
	VectorXd backProp(VectorXd gradient, unsigned int t);
	void applyUpdates();

	void resize(int new_depth);

private:
	int INPUT_SIZE;
	int OUTPUT_SIZE;
	int depth;
	double alpha;

	std::vector<VectorXd> x_history;
	std::vector<VectorXd> y_history;
	std::vector<VectorXd> z_history; // saves having to recompute Z for backprop
	MatrixXd w;
	MatrixXd grad;
	MatrixXd delta_grad;

	void clearCaches();
};

