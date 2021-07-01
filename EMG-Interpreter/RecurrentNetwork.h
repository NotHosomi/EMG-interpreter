#pragma once
#include <fstream>
#include "Lstm.h"
#include "DenseLayer.h"
#include "ElmanLayer.h"
#include "Dataset.h"

#define LOG_ACC 1
#define LOG_SEQ 1

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
	RecurrentNetwork() = delete;
	RecurrentNetwork(GenericLayer* first_layer, double alpha);
	RecurrentNetwork(int input_size, double _alpha);
	~RecurrentNetwork();

	void addLayer(char layer_type, int output_size);
	// TODO: implement net save/load
	void save(std::string net_name);
	void load(std::string net_name);


	double trainSeq(std::vector<VectorXd> inputs, std::vector<VectorXd> labels);
	double trainSet(Dataset<std::vector<VectorXd>> data);

	double evalSeq(std::vector<VectorXd> inputs, std::vector<VectorXd> labels);
	double evalSet(Dataset<std::vector<VectorXd>> data);

	void print();
private:
	VectorXd feedForward(VectorXd input);
	double backProp(std::vector<VectorXd> labels);

	void resize(size_t new_depth);

	// TODO: refine Alphas
	// TODO: make topology more dynamic
	//int INPUT_SIZE = 3;
	//int OUTPUT_SIZE = 5;
	//Lstm L1 = Lstm(3, 16, 0.15);
	//Lstm L2 = Lstm(16, 16, 0.15);
	//DenseLayer L3 = DenseLayer(16, 5, 0.15);
	
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

	int INPUT_SIZE;
	int OUTPUT_SIZE;
	double alpha;
	std::vector<GenericLayer*> layers;

#if LOG_ACC
	std::ofstream acc_log;
#endif
};

