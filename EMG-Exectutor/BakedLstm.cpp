#include "BakedLstm.h"

BakedLstm::BakedLstm(int input_size, int output_size) :
	BakedGenericLayer(input_size, output_size)
{
	f = MatrixXd(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1);
	i = MatrixXd(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1);
	w = MatrixXd(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1); // w acting as c
	o = MatrixXd(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1);

	cell_state = VectorXd::Zero(OUTPUT_SIZE);
	hidden_state = VectorXd::Zero(OUTPUT_SIZE);
}

void BakedLstm::loadWeights(std::ifstream& file)
{
	std::for_each(f.data(), f.data() + f.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double)); });
	std::for_each(i.data(), i.data() + i.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double)); });
	std::for_each(w.data(), w.data() + w.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double)); });
	std::for_each(o.data(), o.data() + o.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double)); });
}

VectorXd BakedLstm::feedForward(VectorXd x_t)
{
	VectorXd in(INPUT_SIZE + OUTPUT_SIZE + 1);
	// concatenate HS<t-1> and X<t>
	in << hidden_state, x_t, 1;

	VectorXd forget = (f * in).unaryExpr(&BakedGenericLayer::sigmoid);
	cell_state = forget.cwiseProduct(cell_state);

	// Create and apply the new Cell State candidate
	VectorXd ignore = (i * in).unaryExpr(&BakedGenericLayer::sigmoid);
	VectorXd candidate = (w * in).unaryExpr(&BakedGenericLayer::tangent); // w acting as c
	cell_state += ignore.cwiseProduct(candidate);

	// Create the new hidden state
	VectorXd output_gate = (o * in).unaryExpr(&BakedGenericLayer::sigmoid); // intermediate vector 'output_gate' named as such to avoid conflict with BakedGenericLayer::output
	hidden_state = output_gate.cwiseProduct(cell_state.unaryExpr(&BakedGenericLayer::tangent));

	return hidden_state;
}

std::stringstream BakedLstm::print()
{
	IOFormat Fmt(4, 0, ", ", ";\n", "", "", "[", "]");
	std::stringstream ss;
	ss << "f:\n" << f.format(Fmt) << "\n"
		<< "i:\n" << i.format(Fmt) << "\n"
		<< "c:\n" << w.format(Fmt) << "\n"
		<< "o:\n" << o.format(Fmt) << "\n";
	return ss;
}