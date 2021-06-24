#pragma once

#include <vector>
#include <Eigen/Dense>
#include "GenericLayer.h"

using namespace Eigen;

class ElmanLayer :
    public GenericLayer
{
public:
	ElmanLayer(int input_size, int output_size, double alpha);

	VectorXd feedForward(VectorXd x_t) override;
	VectorXd backProp(VectorXd gradient, unsigned int t) override;
	void applyUpdates() override;

	void resize(size_t new_depth) override;

private:

	std::vector<VectorXd> x_history;
	std::vector<VectorXd> y_history;
	std::vector<VectorXd> z_history; // saves having to recompute Z for backprop
	MatrixXd w;
	MatrixXd grad;
	MatrixXd momentum;
	MatrixXd rms_prop;

	// delta from context neurons
	VectorXd dc;

	void clearCaches() override;
};

