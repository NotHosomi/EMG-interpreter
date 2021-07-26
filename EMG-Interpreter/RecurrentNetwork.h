#pragma once
#include <fstream>
#include "Lstm.h"
#include "DenseLayer.h"
#include "ElmanLayer.h"
#include "Dataset.h"

#define LOG_ACC 0
#define LOG_SEQ 1
#define LOG_SEQ_TRAIN 0

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
	RecurrentNetwork(int input_size, double _alpha, double _beta1, std::string _name);
	RecurrentNetwork(std::ifstream& file, std::string _name, double _alpha = 0, double _beta1 = 0.9);
	~RecurrentNetwork();

	void addLayer(char layer_type, int output_size);
	// TODO: implement net save/load
	void save(bool checkpoint = false);



	double trainSeq(std::vector<VectorXd> inputs, std::vector<VectorXd> labels);
	double trainSet(Dataset<std::vector<VectorXd>> data);

	double evalSeq(std::vector<VectorXd> inputs, std::vector<VectorXd> labels);
	double evalSet(Dataset<std::vector<VectorXd>> data);
	double evalSetNoMetrics(Dataset<std::vector<VectorXd>> data);

	void print();
	void gradCheck(std::vector<VectorXd> inputs, std::vector<VectorXd> labels);
	void gradCheckAtT(std::vector<VectorXd> inputs, std::vector<VectorXd> labels, int timestep);

	void useCheckpoints(bool useCkp);

	int getInputSize() { return INPUT_SIZE; };
	int getOutputSize() { return OUTPUT_SIZE; };
private:
	VectorXd feedForward(VectorXd input);
	double backProp(std::vector<VectorXd> labels);
	void applyUpdates();
	void clearCaches();

	void resize(size_t new_depth);
	

	std::string name;
	int INPUT_SIZE;
	int OUTPUT_SIZE;
	double alpha;
	double beta1;
	std::vector<GenericLayer*> layers;
	std::vector<VectorXd> y_history;
	double best_loss = 1024;
	bool checkpoints_enabled = true;

#if LOG_ACC
	std::ofstream acc_log;
#endif
};

