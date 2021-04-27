#include "DenseLayer.h"
#include <assert.h>

DenseLayer::DenseLayer(int input_size, int output_size, double alpha) :
	INPUT_SIZE(input_size), OUTPUT_SIZE(output_size), alpha(alpha)
{
	// init W matrices for gates
	w = MatrixXd::Random(OUTPUT_SIZE, INPUT_SIZE + 1); // + 1 (bias)

	// add a 0th tick (everything set to 0)
	VectorXd empt(INPUT_SIZE);
	empt.setZero();
	x_history.emplace_back(empt);

	empt = VectorXd(OUTPUT_SIZE);
	empt.setZero();
	y_history.emplace_back(empt);

	grad.setZero(OUTPUT_SIZE, INPUT_SIZE + 1);
	delta_grad.setZero(OUTPUT_SIZE, INPUT_SIZE + 1);
}

#pragma region UTILS
double DenseLayer::sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double DenseLayer::dsigmoid(double x)
{
	return (1 / (1 + exp(-x))) * (1 - (1 / (1 + exp(-x))));
}

double DenseLayer::tangent(double x)
{
	return tanh(x);
}

double DenseLayer::dtangent(double x)
{
	//return (1 / cosh(x)) * (1 / cosh(x)); // is cosh unsafe?

	// simplified tanh derivative
	double th = tanh(x);
	return 1 - th * th;
}

// clamps between -6 and 6 to prevent vanishing
// redundant?
double DenseLayer::m_clamp(double x)
{
	if (x > 6)
		return 6;
	if (x < -6)
		return -6;
	return x;
}

double DenseLayer::reciprocal(double x)
{
	return 1 / x;
}

// reset memory, excluding weights
void DenseLayer::clearCaches()
{
	x_history.clear();
	y_history.clear();

	VectorXd empt(INPUT_SIZE);
	empt.setZero();
	x_history.emplace_back(empt);

	empt = VectorXd(OUTPUT_SIZE);
	empt.setZero();
	y_history.emplace_back(empt);

	// clear update buffers
	grad.setZero(OUTPUT_SIZE, INPUT_SIZE + 1);
	delta_grad.setZero(OUTPUT_SIZE, INPUT_SIZE + 1);
}
#pragma endregion

VectorXd DenseLayer::feedForward(VectorXd x_t)
{
	VectorXd in(INPUT_SIZE + 1);
	in << x_t, 1;
	x_history.emplace_back(in);

	VectorXd output = (w * in).unaryExpr(&sigmoid);
	y_history.emplace_back(output);
	return output;
}

VectorXd DenseLayer::backProp(VectorXd gradient, unsigned int t)
{
	VectorXd de_db = (w * x_history[t]).unaryExpr(&dsigmoid).cwiseProduct(gradient);
	MatrixXd de_dw = de_db * x_history[t].transpose();

	grad += de_dw;
	return w.block(0, 0, OUTPUT_SIZE, INPUT_SIZE).transpose() * de_db;
}

void DenseLayer::applyUpdates()
{
	// TODO, implement Adam
	// Vdw = Beta * Vdw + (1 - Beta) * dw
	delta_grad = 0.9 * delta_grad + 0.1 * grad.cwiseProduct(grad);

	// apply update sums
	// TODO use adam instead of just RMS prop
	w += (alpha / sqrt(delta_grad.array() + 1e-8) * grad.array()).matrix();
}

void DenseLayer::resize(int new_depth)
{
	clearCaches();
	new_depth++;
	x_history.reserve(new_depth);
	y_history.reserve(new_depth);
}
