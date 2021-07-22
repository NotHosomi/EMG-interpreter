#pragma once

#include <fstream>
#include <mutex>
#include "BakedLstm.h"
#include "BakedDenseLayer.h"
#include "BakedElmanLayer.h"
#include "Timer.h"

class BakedNetwork
{
public:
	BakedNetwork() = delete;
	BakedNetwork(std::ifstream& file);
	void addLayer(char layer_type, int output_size);

	void run();
	void addInput(VectorXd inputs);
	VectorXd getOutput();
	bool isUpdated();
	void stop();
private:
	int INPUT_SIZE;
	int OUTPUT_SIZE;
	std::vector<BakedGenericLayer*> layers;

	void feedForward();
	VectorXd input; // latest input 
	VectorXd output;
	bool queued = false;
	bool updated = false;

	std::mutex input_lock;
	std::mutex output_lock;

	Timer update_tmr;
	float tick = 100;

	std::atomic<bool> stop_signal = false;
};

