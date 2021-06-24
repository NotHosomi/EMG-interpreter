#pragma once
#include "Lstm.h"
#include "DenseLayer.h"
#include "ElmanLayer.h"

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
	RecurrentNetwork();


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

	void resize(size_t new_depth);
	int depth;

	// TODO: refine Alphas
	// TODO: make topology more dynamic
	int INPUT_SIZE = 3;
	int OUTPUT_SIZE = 5;
	Lstm L1 = Lstm(3, 16, 0.15);
	Lstm L2 = Lstm(16, 16, 0.15);
	DenseLayer L3 = DenseLayer(16, 5, 0.15);
	
	std::vector<VectorXd> y_history;

	//int INPUT_SIZE = 1;
	//int OUTPUT_SIZE = 3;
	//Lstm L1 = Lstm(1, 5, 0.15);
	//DenseLayer L2 = DenseLayer(5, 3, 0.15);
	 
	//int INPUT_SIZE = 1;
	//int OUTPUT_SIZE = 3;
	////Lstm L1 = Lstm(2, 1, 0.15);
	//Lstm L1 = Lstm(1, 6, 0.15);
	//DenseLayer L2 = DenseLayer(6, 3, 0.15);

	//std::vector<Layer*> Layers;
};

