#pragma once

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class DenseLayer
{
public:
	DenseLayer(int input_size, int output_size, double eta);

private:
	// internal utils
	static double sigmoid(double x);
	static double dsigmoid(double x);
	static double tangent(double x);
	static double dtangent(double x);
	static double m_clamp(double x);
	static double reciprocal(double x);
	void clearCaches();

public:
	VectorXd feedForward(VectorXd x_t);
	VectorXd backProp(VectorXd gradient, unsigned int t);
	void applyUpdates();

private:
	int INPUT_SIZE; // not const, but these two should never change after construction
	int OUTPUT_SIZE; // could const these and have loadCell() be a static that returns a new Lstm
	int depth;
	double alpha;


	MatrixXd f; // forget
	std::vector<VectorXd> x_history;
	std::vector<VectorXd> y_history;
	MatrixXd w;
	MatrixXd grad;
	MatrixXd delta_grad;
};

