#pragma once

#include <fstream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class BakedGenericLayer
{
public:
	BakedGenericLayer(int input_size, int output_size);
	virtual void loadWeights(std::ifstream& file);

	virtual VectorXd feedForward(VectorXd x_t) = 0;

	virtual std::stringstream print();

	// activation functions
	static double sigmoid(double x);
	static double tangent(double x);
protected:
	int INPUT_SIZE;
	int OUTPUT_SIZE;

	MatrixXd w; // weights
};

