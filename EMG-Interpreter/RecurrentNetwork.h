#pragma once
#include "Lstm.h"
#include "DenseLayer.h"

class RecurrentNetwork
{
	/* 
	* HARDCODED TOPOLOGY
	*  LSTM 16
	*  LSTM 16
	*  Dense 5
	*/
public:
	VectorXd feedForward(VectorXd input);
	double backProp(std::vector<VectorXd> labels);

	void resize(int new_depth);
	// TODO: implement net loading

private:
	int INPUT_SIZE = 3;
	int OUTPUT_SIZE = 5;
	int depth;

	// TODO: refine Alphas
	// TODO: make topology more dynamic
	Lstm L1 = Lstm(3, 16, 0.15);
	Lstm L2 = Lstm(16, 16, 0.15);
	DenseLayer L3 = DenseLayer(16, 5, 0.15);
	
	std::vector<VectorXd> y_history;
};

