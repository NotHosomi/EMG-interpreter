#pragma once
#include "BakedGenericLayer.h"
class BakedLstm : public BakedGenericLayer
{
public:
	BakedLstm(int input_size, int output_size);
	void loadWeights(std::ifstream& file) override;

	VectorXd feedForward(VectorXd x_t) override;

	std::stringstream print() override;
private:
	// persistent states
	VectorXd cell_state;
	VectorXd hidden_state;

	// Weight matrices
	MatrixXd f;
	MatrixXd i;
	// inherited matrix 'w' used as candidate gate
	MatrixXd o;
};

