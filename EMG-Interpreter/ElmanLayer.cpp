#include "ElmanLayer.h"
#include <assert.h>
#include <iostream>
#include "Common.h"

using namespace Common;

ElmanLayer::ElmanLayer(int input_size, int output_size, double alpha) :
	GenericLayer(input_size, output_size, alpha)
{
	// init W matrices for gates
	w = MatrixXd::Random(OUTPUT_SIZE, OUTPUT_SIZE+INPUT_SIZE + 1); // + 1 (bias)

	// init Adam matrices
	momentum.setZero(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1);
	rms_prop.setZero(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1);

	// add a 0th tick (everything set to 0)
	clearCaches();
}

VectorXd ElmanLayer::feedForward(VectorXd x_t)
{
	VectorXd in(OUTPUT_SIZE + INPUT_SIZE + 1);
	in << y_history.back(), x_t, 1;
	x_history.push_back(x_t);

	VectorXd z = w * in;
	z_history.push_back(z);

	VectorXd output = z.unaryExpr(&sigmoid);
#ifdef PRINT_FF
	std::cout << "\nHS:\n"
		<< y_history.back()
		<< "\nX:\n" << x_t
		<< "\nZ:" << z << std::endl;
#endif
	y_history.push_back(output);

	return output;
}

VectorXd ElmanLayer::backProp(VectorXd gradient, unsigned int t)
{
#ifdef PRINT_BP
	VectorXd old_dc = dc;
	VectorXd old_grad = gradient;
#endif
	// include the gradient from the future context
	gradient += dc;
	VectorXd in(OUTPUT_SIZE + INPUT_SIZE + 1);
	in << y_history[t-1], x_history[t], 1;

	//VectorXd da_dz = z_history[t].unaryExpr(&dsigmoid);
	VectorXd de_dz = z_history[t].unaryExpr(&dsigmoid).cwiseProduct(gradient);
	MatrixXd de_dw = de_dz * in.transpose();

	grad += de_dw;
	dc = w.block(0, 0, OUTPUT_SIZE, OUTPUT_SIZE).transpose() * de_dz;


#ifdef PRINT_BP
	std::cout
		<< "\nHS:\n" << y_history[t-1]
		<< "\nX:\n" << x_history[t]
		<< "\nZ:" << z_history[t]
		<< "\nY:" << y_history[t]
		<< "\nw\n" << w
		<< "\nGrad\n" << old_grad
		<< "\ndc\n" << old_dc
		<< "\nCombined Grad:\n" << gradient
		<< "\nde_dz\n" << de_dz
		<< "\nde_dw\n" << de_dw
		<< "\nnext dc\n" << dc
		<< std::endl;
#endif

	return w.block(0, OUTPUT_SIZE, OUTPUT_SIZE, INPUT_SIZE).transpose() * de_dz;
}

void ElmanLayer::applyUpdates()
{
	++adam_t;
	// Momentum
	// Vdw = Beta1 * Vdw + (1 - Beta1) * dw
	momentum = 0.9 * momentum + 0.1 * grad;
	// RMS Prop
	// Sdw = Beta2 * Sdw + (1-Beta2) * dw^2
	rms_prop = 0.999 * rms_prop + 0.001 * grad.cwiseProduct(grad);

	// Bias correction
	MatrixXd Vc = momentum / (1 - pow(0.9, adam_t));
	MatrixXd Sc = rms_prop / (1 - pow(0.999, adam_t));

	// apply update sums
	// w = w - alpha * Vdwc / (root(Sdwc) + eps)
	w -= (alpha * Vc.array() / (sqrt(Sc.array()) + 1e-8)).matrix();

	clearCaches();
}

void ElmanLayer::resize(size_t new_depth)
{
	clearCaches();
	new_depth++;
	x_history.reserve(new_depth);
	y_history.reserve(new_depth);
	z_history.reserve(new_depth);
}

void ElmanLayer::print()
{
	IOFormat Fmt(4, 0, ", ", ";\n", "", "", "[", "]");
	std::cout << "w:\n" << w.format(Fmt) << std::endl;
}

// reset memory, excluding weights
void ElmanLayer::clearCaches()
{
	x_history.clear();
	y_history.clear();
	z_history.clear();

	VectorXd empt(OUTPUT_SIZE + INPUT_SIZE + 1);
	empt.setZero();
	x_history.push_back(empt);

	empt = VectorXd(OUTPUT_SIZE);
	empt.setZero();
	y_history.push_back(empt);
	z_history.push_back(empt);

	dc.setZero(OUTPUT_SIZE);

	grad.setZero(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1);
}