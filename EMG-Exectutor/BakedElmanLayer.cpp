#include "BakedElmanLayer.h"

BakedElmanLayer::BakedElmanLayer(int input_size, int output_size) :
	BakedGenericLayer(input_size, output_size)
{
	w = MatrixXd(OUTPUT_SIZE, OUTPUT_SIZE+INPUT_SIZE + 1);
	context_activations = VectorXd::Zero(OUTPUT_SIZE);
}

VectorXd BakedElmanLayer::feedForward(VectorXd x_t)
{
	VectorXd in(OUTPUT_SIZE + INPUT_SIZE + 1);
	// activations from previous timestep, input from current timestep, and bias
	in << context_activations, x_t, 1;
	context_activations = (w * in).unaryExpr(&BakedGenericLayer::sigmoid);
	return context_activations;
}
