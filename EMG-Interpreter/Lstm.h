#pragma once

#include <vector>
#include <Eigen/Dense>
#include "GenericLayer.h"
using namespace Eigen;


class Lstm : public GenericLayer
{
public:
	Lstm(int input_size, int output_size, double alpha);
	Lstm(std::ifstream& file, double alpha);

	// external utils
	void resize(size_t new_depth);
	void print() override;
	bool saveCell();
	void loadGates(MatrixXd forget, MatrixXd ignore, MatrixXd candidate, MatrixXd output);

	// primary functionality
	VectorXd feedForward(VectorXd x_t) override;
	VectorXd backProp(VectorXd gradient, unsigned int t) override;
	void applyUpdates() override;

private:
	void clearCaches() override;

	// Cell IO
	std::vector<VectorXd> x_history;
	std::vector<VectorXd> cs_history;
	std::vector<VectorXd> hs_history;
	// Gate weights
	MatrixXd f; // forget
	MatrixXd i; // ignore
	MatrixXd c; // candidate
	MatrixXd o; // output
	// Gate cache history
	std::vector<VectorXd> fz_history;
	std::vector<VectorXd> iz_history;
	std::vector<VectorXd> cz_history;
	std::vector<VectorXd> oz_history;
	// Gate output history
	std::vector<VectorXd> f_history;
	std::vector<VectorXd> i_history;
	std::vector<VectorXd> c_history;
	std::vector<VectorXd> o_history;
	// optimizer components
	MatrixXd momentum_f;
	MatrixXd momentum_i;
	MatrixXd momentum_c;
	MatrixXd momentum_o;
	MatrixXd rms_prop_f;
	MatrixXd rms_prop_i;
	MatrixXd rms_prop_c;
	MatrixXd rms_prop_o;
	// TODO: gate bias (VectorXd, for each weight from the bias), or just +1 to the X vector each time maybe?

	// intercell gradients
	VectorXd dcs;
	VectorXd dhs;
	// Backprop update sums
	MatrixXd tfu;
	MatrixXd tiu;
	MatrixXd tcu;
	MatrixXd tou;

	void evalUpdates(MatrixXd gate, MatrixXd updates, char name);
};