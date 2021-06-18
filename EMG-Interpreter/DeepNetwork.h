#pragma once
#include "DenseLayer.h"

class DeepNetwork
{
public:
	double train(std::vector<VectorXd> inputs, std::vector<VectorXd> labels);
	double eval(std::vector<VectorXd> inputs, std::vector<VectorXd> labels);

	VectorXd run(VectorXd inputs);

private:
	void feedForward(VectorXd input);
	void backProp(VectorXd label);

	int INPUT_SIZE = 2;
	int OUTPUT_SIZE = 1;
	DenseLayer L1 = DenseLayer(2, 4, 0.15);
	DenseLayer L2 = DenseLayer(4, 4, 0.15);
	DenseLayer L3 = DenseLayer(4, 1, 0.15);

	// last input and output
	VectorXd x;
	VectorXd y;
};

