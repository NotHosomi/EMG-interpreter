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
	//RecurrentNetwork(std::vector<std::tuple<char, int>> topology, int input_size);
	double trainSingle(VectorXd input, VectorXd label);
	void train(std::vector<VectorXd> inputs, std::vector<VectorXd> labels);
	double trainEx(std::vector<VectorXd> inputs, std::vector<VectorXd> labels);
	double trainSeqBatch(std::vector<std::vector<VectorXd>> inputs, std::vector<std::vector<VectorXd>> labels);

	double evalSample(VectorXd input, VectorXd label);
	double evalSeq(std::vector<VectorXd> inputs, std::vector<VectorXd> labels);
	double evalSeqBatch(std::vector<std::vector<VectorXd>> input_sequences, std::vector<std::vector<VectorXd>> label_sequences);


	// TODO: implement net loading
	void save(std::string net_name);
	void load(std::string net_name);
private:
	VectorXd feedForward(VectorXd input);
	double backProp(std::vector<VectorXd> labels);

	void resize(int new_depth);
	VectorXd loss(VectorXd outputs, VectorXd targets);
	VectorXd dloss(VectorXd outputs, VectorXd targets);
	//int INPUT_SIZE = 3;
	//int OUTPUT_SIZE = 5;
	int depth;

	// TODO: refine Alphas
	// TODO: make topology more dynamic
	//Lstm L1 = Lstm(3, 16, 0.15);
	//Lstm L2 = Lstm(16, 16, 0.15);
	//DenseLayer L3 = DenseLayer(16, 5, 0.15);
	
	std::vector<VectorXd> y_history;

	int INPUT_SIZE = 1;
	int OUTPUT_SIZE = 3;
	Lstm L1 = Lstm(1, 3, 0.15);

	//std::vector<Layer*> Layers;
};

