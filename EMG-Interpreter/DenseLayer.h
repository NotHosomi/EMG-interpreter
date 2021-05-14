#pragma once

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class DenseLayer
{
public:
	DenseLayer(int input_size, int output_size, double alpha);

private:
	// internal utils
	static double sigmoid(double x);
	static double dsigmoid(double x);
	static double tangent(double x);
	static double dtangent(double x);
	static double m_clamp(double x);
	static double reciprocal(double x);
	void clearCaches();
	static MatrixXd mtable(VectorXd rows, RowVectorXd cols);

public:
	VectorXd feedForward(VectorXd x_t);
	VectorXd backProp(VectorXd gradient, unsigned int t);
	void applyUpdates();

	void resize(int new_depth);

private:
	int INPUT_SIZE;
	int OUTPUT_SIZE;
	int depth;
	double alpha;

	std::vector<VectorXd> x_history;
	std::vector<VectorXd> y_history;
	MatrixXd w;
	MatrixXd grad;
	MatrixXd delta_grad;
};

