#pragma once

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

// temp, for example
#include <vector>

class Lstm
{
public:
	Lstm(int input_size, int output_size, double alpha);
	Lstm(std::ifstream& file, double alpha);

	// external utils
	void resize(int new_depth);
	void printGates();
	bool saveCell();
	void loadGates(MatrixXd forget, MatrixXd ignore, MatrixXd candidate, MatrixXd output);

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
	// primary functionality
	VectorXd feedForward(VectorXd x_t);
	VectorXd backProp(VectorXd gradient, unsigned int t);
	void applyUpdates();

private:
	int INPUT_SIZE; // not const, but these two should never change after construction
	int OUTPUT_SIZE; // could const these and have loadCell() be a static that returns a new Lstm
	int depth;
	double alpha; // learning rate		// TODO make dynamic

	// Cell IO
	std::vector<VectorXd> x_history;
	std::vector<VectorXd> cs_history;
	std::vector<VectorXd> hs_history;
	// Gate weights
	MatrixXd f; // forget
	MatrixXd i; // ignore
	MatrixXd c; // candidate
	MatrixXd o; // output
	// Gate output history
	std::vector<VectorXd> f_history;
	std::vector<VectorXd> i_history;
	std::vector<VectorXd> c_history;
	std::vector<VectorXd> o_history;
	// Delta gradients
	MatrixXd Gf;
	MatrixXd Gi;
	MatrixXd Gc;
	MatrixXd Go;
	// TODO: gate bias (VectorXd, for each weight from the bias), or just +1 to the X vector each time maybe?

	// intercell gradients
	VectorXd dcs;
	VectorXd dhs;
	// Backprop update sums
	MatrixXd tfu;
	MatrixXd tiu;
	MatrixXd tcu;
	MatrixXd tou;
};