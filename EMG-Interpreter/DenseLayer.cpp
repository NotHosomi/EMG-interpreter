#include "DenseLayer.h"
#include <assert.h>
#include <iostream>

DenseLayer::DenseLayer(int input_size, int output_size, double alpha, double beta1) :
	GenericLayer(input_size, output_size, alpha, beta1)
{
	// init W matrices for gates
	w = MatrixXd::Random(OUTPUT_SIZE, INPUT_SIZE + 1); // + 1 (bias)

	// init Adam matrices
	momentum.setZero(OUTPUT_SIZE, INPUT_SIZE + 1);
	rms_prop.setZero(OUTPUT_SIZE, INPUT_SIZE + 1);

	// add a 0th tick (everything set to 0)
	clearCaches();
}

// Feedforward through the layer
VectorXd DenseLayer::feedForward(VectorXd x_t)
{
	// append a bias 
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
	// dZ = sig'(Z) * dA;
	VectorXd de_dz = z_history[t].unaryExpr(&dsigmoid).cwiseProduct(gradient);
	// dW = dZ * x^T;
	MatrixXd de_dw = de_dz * x_history[t].transpose();

	// add the local weights derivative to the sum
	grad += de_dw;

	// extract gradients of the input
	return w.block(0, 0, OUTPUT_SIZE, INPUT_SIZE).transpose() * de_dz;
}

// Hyperparameters:
// Alpha: requires tuning
// Beta1: 0.9
// Beta2: 0.999
// Epsilon: 10^-8
void DenseLayer::applyUpdates()
{
	// average gradients over sequence
	// This isn't NORMAL, but it seems to mitigate the gradient bug a tiny bit
#if AVG_GRAD
	grad /= x_history.size();
#endif

	// Momentum
	// Vdw = Beta1 * Vdw + (1 - Beta1) * dw
	momentum = beta1 * momentum + (1-beta1) * grad;
	// RMS Prop
	// Sdw = Beta2 * Sdw + (1-Beta2) * dw^2
	rms_prop = 0.999 * rms_prop + 0.001 * grad.cwiseProduct(grad);

	// increment the bias correction factor
	++adam_t;
	// Bias correction
	MatrixXd Vc = momentum / (1- pow(beta1, adam_t));
	MatrixXd Sc = rms_prop / (1- pow(0.999, adam_t));

	// apply update sums
	// w = w - alpha * Vdwc / (root(Sdwc) + eps)
	w -= (alpha * Vc.array() / (sqrt(Sc.array()) + 1e-8)).matrix();

	clearCaches();
}

// reserve cache memory to prevent relocation as they fill up
void DenseLayer::resize(size_t new_depth)
{
	clearCaches();
	new_depth++;
	x_history.reserve(new_depth);
	y_history.reserve(new_depth);
	z_history.reserve(new_depth);
}

void DenseLayer::loadWeights(std::ifstream& file)
{
	w = MatrixXd(OUTPUT_SIZE, INPUT_SIZE + 1);

	std::for_each(w.data(), w.data() + w.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double));
		});
}

void DenseLayer::save(std::ofstream& file)
{
	char type = 'D';
	file.write(reinterpret_cast<char*>(&type), sizeof(char));
	file.write(reinterpret_cast<char*>(&OUTPUT_SIZE), sizeof(int));
	std::for_each(w.data(), w.data() + w.size(), [&file](double val)
		{ file.write(reinterpret_cast<char*>(&val), sizeof(double)); });
}

void DenseLayer::print()
{
	IOFormat Fmt(4, 0, ", ", ";\n", "", "", "[", "]");
	std::cout << "w:\n" << w.format(Fmt) << std::endl;
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

void DenseLayer::readWeightBuffer(const std::vector<double>& theta, int& pos)
{
	std::for_each(w.data(), w.data() + w.size(), [theta, &pos](double& val)
		{
			val = theta[pos];
			++pos;
		});
}

void DenseLayer::writeWeightBuffer(std::vector<double>& theta, int& pos)
{
	std::for_each(w.data(), w.data() + w.size(), [&theta, &pos](double val)
		{
			theta.push_back(val);
			++pos;
		});
}

void DenseLayer::writeUpdateBuffer(VectorXd& theta, int& pos)
{
	std::for_each(grad.data(), grad.data() + grad.size(), [&theta, &pos](double val)
		{
			theta[pos] = val;
			++pos;
		});
}