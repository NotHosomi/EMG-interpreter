#include "DenseLayer.h"
#include <assert.h>
#include <iostream>

DenseLayer::DenseLayer(int input_size, int output_size, double alpha) :
	INPUT_SIZE(input_size), OUTPUT_SIZE(output_size), alpha(alpha)
{
	// init W matrices for gates
	w = MatrixXd::Random(OUTPUT_SIZE, INPUT_SIZE + 1); // + 1 (bias)

	// add a 0th tick (everything set to 0)
	VectorXd empt(INPUT_SIZE);
	empt.setZero();
	x_history.push_back(empt);

	empt = VectorXd(OUTPUT_SIZE);
	empt.setZero();
	y_history.push_back(empt);
	z_history.push_back(empt);

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
	double s = 1 / (1 + exp(-x));
	return s * (1 - s);
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

// doesn't this just emulate r*c?
MatrixXd DenseLayer::mtable(VectorXd rows, RowVectorXd cols)
{
	return rows.replicate(1, cols.size()).cwiseProduct(
		cols.replicate(rows.size(), 1)
	);
}
#pragma endregion

VectorXd DenseLayer::feedForward(VectorXd x_t)
{
	VectorXd in(INPUT_SIZE + 1);
	in << x_t, 1;
	x_history.push_back(in);
	//std::cout << "\n\n\nIN:\n" << in << std::endl;
	//std::cout << "W:\n" << w << std::endl;

	VectorXd z = w * in;
	z_history.push_back(z);
	//std::cout << "Z:\n" << z << std::endl;

	VectorXd output = (z).unaryExpr(&sigmoid);
	y_history.push_back(output);
	//std::cout << "A:\n" << output << std::endl;
	return output;
}

VectorXd DenseLayer::backProp(VectorXd gradient, unsigned int t)
{
	VectorXd da_dz = z_history[t].unaryExpr(&dsigmoid);
	//std::cout << "da_dz:\n" << da_dz << std::endl;
	VectorXd de_dz = da_dz.cwiseProduct(gradient);
	//std::cout << "de_dz:\n" << de_dz << std::endl;

	//MatrixXd de_dw = de_dz * x_history[t].transpose();
	MatrixXd de_dw = mtable(de_dz, x_history[t].transpose());
	//std::cout << "de_dw:\n" << de_dw << std::endl;

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
	w += (alpha / sqrt(delta_grad.array() + 1e-8) * grad.array()).matrix();
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
