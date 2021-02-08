#pragma once

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

// temp, for example
#include <vector>

class DumbLstm
{
public:
	explicit DumbLstm(std::ifstream& file);

private:
	// internal utils
	static double sigmoid(double x);
	static double tangent(double x);

public:
	// primary functionality
	VectorXd feedForward(VectorXd x_t);

private:
	int INPUT_SIZE; // not const, but these two should never change after construction
	int OUTPUT_SIZE; // could const these and have loadCell() be a static that returns a new Lstm

	// Cell IO
	VectorXd cell_state;
	VectorXd hidden_state;
	// Gate weights
	MatrixXd f; // forget
	MatrixXd i; // ignore
	MatrixXd c; // candidate
	MatrixXd o; // output
};