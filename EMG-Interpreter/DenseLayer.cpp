#include "DenseLayer.h"
#include <assert.h>
#include <iostream>

DenseLayer::DenseLayer(int input_size, int output_size, double alpha) :
	INPUT_SIZE(input_size), OUTPUT_SIZE(output_size), alpha(alpha)
{
	// init W matrices for gates
	w = MatrixXd::Random(OUTPUT_SIZE, INPUT_SIZE + 1); // + 1 (bias)

	// add a 0th tick (everything set to 0)
	VectorXd empt(INPUT_SIZE + 1);
	empt.setZero();
	x_history.push_back(empt);

	empt = VectorXd(OUTPUT_SIZE);
	empt.setZero();
	y_history.push_back(empt);
	z_history.push_back(empt);

	grad.setZero(OUTPUT_SIZE, INPUT_SIZE + 1);
	delta_grad.setZero(OUTPUT_SIZE, INPUT_SIZE + 1);
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

void DenseLayer::applyUpdates()
{
	// TODO, implement Adam
	// Vdw = Beta * Vdw + (1 - Beta) * dw
	delta_grad = 0.9 * delta_grad + 0.1 * grad.cwiseProduct(grad);

	// apply update sums
	// TODO use adam instead of just RMS prop
	w -= (alpha / sqrt(delta_grad.array() + 1e-8) * grad.array()).matrix();
	clearCaches();
}

void DenseLayer::resize(int new_depth)
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

	// clear update buffers
	grad.setZero(OUTPUT_SIZE, INPUT_SIZE + 1);
	delta_grad.setZero(OUTPUT_SIZE, INPUT_SIZE + 1);
}