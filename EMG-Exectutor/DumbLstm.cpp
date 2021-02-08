#include "DumbLstm.h"
#include <iostream>
#include <fstream>

DumbLstm::DumbLstm(std::ifstream& file)
{
	file.read(reinterpret_cast<char*>(&INPUT_SIZE), sizeof(unsigned int));
	file.read(reinterpret_cast<char*>(&OUTPUT_SIZE), sizeof(unsigned int));

	f = MatrixXd(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	i = MatrixXd(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	c = MatrixXd(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	o = MatrixXd(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);

	std::for_each(f.data(), f.data() + f.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double)); });
	std::for_each(i.data(), i.data() + i.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double)); });
	std::for_each(c.data(), c.data() + c.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double)); });
	std::for_each(o.data(), o.data() + o.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double)); });

	cell_state = VectorXd(OUTPUT_SIZE);
	cell_state.setZero();
	hidden_state = VectorXd(OUTPUT_SIZE);
	hidden_state.setZero();
}

#pragma region UTILS

double DumbLstm::sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double DumbLstm::tangent(double x)
{
	return tanh(x);
}
#pragma endregion

VectorXd DumbLstm::feedForward(VectorXd x_t)
{
	VectorXd in(INPUT_SIZE + OUTPUT_SIZE + 1); // + 1; (bias)

	// concatenate HS<t-1> and X<t>
	in << hidden_state, x_t, 1; // , 1; (bias)

	VectorXd forget = (f * in).unaryExpr(&sigmoid);
	cell_state = forget.cwiseProduct(cell_state);

	// Create and apply the new Cell State candidate
	VectorXd ignore = (i * in).unaryExpr(&sigmoid);
	VectorXd candidate = (c * in).unaryExpr(&tangent);
	cell_state += ignore.cwiseProduct(candidate);

	// Create the new output
	VectorXd output = (o * in).unaryExpr(&sigmoid);
	hidden_state = output.cwiseProduct(cell_state.unaryExpr(&tangent));

	return hidden_state;
}
