#pragma once

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class GenericLayer
{
public:
	GenericLayer(int input_size, int output_size, double alpha);



	virtual VectorXd feedForward(VectorXd x_t) = 0;
	virtual VectorXd backProp(VectorXd gradient, unsigned int t) = 0;
	virtual void applyUpdates() = 0;

	virtual void resize(size_t new_depth) = 0;

	int getInputSize() { return INPUT_SIZE; };
	int getOutputSize() { return OUTPUT_SIZE; };

	virtual void print() = 0;
protected:
	int INPUT_SIZE; // not const, but these two should never change after construction
	int OUTPUT_SIZE; // could const these and have loadCell() be a static that returns a new Lstm
	int depth;
	// Optimizer components
	double alpha;
	int adam_t = 0;

	virtual void clearCaches() = 0;
};

