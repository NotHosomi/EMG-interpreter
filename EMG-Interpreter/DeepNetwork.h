#pragma once
#include "DenseLayer.h"

class DeepNetwork
{
public:
	double train(std::vector<VectorXd> inputs, std::vector<VectorXd> labels);

private:
	void feedForward(VectorXd input);
	void backProp(VectorXd label);

	VectorXd loss(VectorXd outputs, VectorXd targets);
	VectorXd dloss(VectorXd outputs, VectorXd targets);

	int INPUT_SIZE = 2;
	int OUTPUT_SIZE = 1;
	DenseLayer L1 = DenseLayer(2, 4, 0.01);
	DenseLayer L2 = DenseLayer(4, 1, 0.01);

	// last input and output
	VectorXd x;
	VectorXd y;
};

