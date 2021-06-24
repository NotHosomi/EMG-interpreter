#include "DenseLayer.h"
#include <assert.h>
#include <iostream>

DenseLayer::DenseLayer(int input_size, int output_size, double alpha) :
	GenericLayer(input_size, output_size, alpha)
{
	// init W matrices for gates
	w = MatrixXd::Random(OUTPUT_SIZE, INPUT_SIZE + 1); // + 1 (bias)

	// init Adam matrices
	momentum.setZero(OUTPUT_SIZE, INPUT_SIZE + 1);
	rms_prop.setZero(OUTPUT_SIZE, INPUT_SIZE + 1);

	// add a 0th tick (everything set to 0)
	clearCaches();
}

VectorXd DenseLayer::feedForward(VectorXd x_t)
{
	VectorXd in(INPUT_SIZE + 1);
	in << x_t, 1;
	x_history.push_back(in);

	VectorXd z = w * in;
	z_history.push_back(z);

	VectorXd output = z.unaryExpr(&sigmoid);
	y_history.push_back(output);
	return output;
}

VectorXd DenseLayer::backProp(VectorXd gradient, unsigned int t)
{
	//VectorXd da_dz = z_history[t].unaryExpr(&dsigmoid);
	VectorXd de_dz = z_history[t].unaryExpr(&dsigmoid).cwiseProduct(gradient);
	MatrixXd de_dw = de_dz * x_history[t].transpose();

	grad += de_dw;
	return w.block(0, 0, OUTPUT_SIZE, INPUT_SIZE).transpose() * de_dz;
}

// Hyperparameters:
// Alpha: requires tuning
// Beta1: 0.9
// Beta2: 0.999
// Epsilon: 10^-8
void DenseLayer::applyUpdates()
{
	++adam_t;
	// Momentum
	// Vdw = Beta1 * Vdw + (1 - Beta1) * dw
	momentum = 0.9 * momentum + 0.1 * grad;
	// RMS Prop
	// Sdw = Beta2 * Sdw + (1-Beta2) * dw^2
	rms_prop = 0.999 * rms_prop + 0.001 * grad.cwiseProduct(grad);

	// Bias correction
	MatrixXd Vc = momentum / (1- pow(0.9, adam_t));
	MatrixXd Sc = rms_prop / (1- pow(0.999, adam_t));

	// apply update sums
	// w = w - alpha * Vdwc / (root(Sdwc) + eps)
	w -= (alpha * Vc.array() / (sqrt(Sc.array()) + 1e-8)).matrix();

	clearCaches();
}

void DenseLayer::resize(size_t new_depth)
{
	clearCaches();
	new_depth++;
	x_history.reserve(new_depth);
	y_history.reserve(new_depth);
	z_history.reserve(new_depth);
}

// reset memory, excluding weights
void DenseLayer::clearCaches()
{
	x_history.clear();
	y_history.clear();
	z_history.clear();

	VectorXd empt(INPUT_SIZE);
	empt.setZero();
	x_history.push_back(empt);

	empt = VectorXd(OUTPUT_SIZE);
	empt.setZero();
	y_history.push_back(empt);
	z_history.push_back(empt);

	grad.setZero(OUTPUT_SIZE, INPUT_SIZE + 1);
}