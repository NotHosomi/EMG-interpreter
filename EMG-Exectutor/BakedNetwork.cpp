#include "BakedNetwork.h"
#include "TimeConstants.h"
#include <windows.h>

#include <QDebug>

BakedNetwork::BakedNetwork(std::ifstream& file)
{
    int num_layers = 0;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
    file.read(reinterpret_cast<char*>(&INPUT_SIZE), sizeof(int));
    OUTPUT_SIZE = INPUT_SIZE;
    
    double file_alpha;  // Used for training, can just discard
    file.read(reinterpret_cast<char*>(&file_alpha), sizeof(double));
    
    for (int l = 0; l < num_layers; ++l)
    {
        char layer_type = '\0';
        int layer_size = 0;
        file.read(reinterpret_cast<char*>(&layer_type), sizeof(char));
        file.read(reinterpret_cast<char*>(&layer_size), sizeof(int));
    
        addLayer(layer_type, layer_size);
        layers.back()->loadWeights(file);
    }
    file.close();

    // pre-init vectors
    input = VectorXd(INPUT_SIZE);
    input.setZero();
    output = VectorXd(OUTPUT_SIZE);
    output.setZero();
}

void BakedNetwork::addLayer(char layer_type, int output_size)
{
    switch (layer_type)
    {
    case 'D': layers.emplace_back(new BakedDenseLayer(OUTPUT_SIZE, output_size));
        break;
    case 'L': layers.emplace_back(new BakedLstm(OUTPUT_SIZE, output_size));
        break;
    case 'E': layers.emplace_back(new BakedElmanLayer(OUTPUT_SIZE, output_size));
        break;
    default: throw std::invalid_argument("received invalid layer type. Bad net file.");
        return;
    }
    OUTPUT_SIZE = output_size;
}

void BakedNetwork::run()
{
    update_tmr.mark();
    do
    {
        if (update_tmr.peek() < SAMPLE_TIME)
        {
            Sleep(POLL_DELAY);
            continue;
        }

        input_lock.lock();
        if (!queued)
        {
            input_lock.unlock();
            Sleep(POLL_DELAY);
            continue;
        }

        input_lock.unlock();
        feedForward();
    } while (!stop_signal);
}

void BakedNetwork::addInput(VectorXd inputs)
{
    input_lock.lock();
    input = inputs;
    queued = true;
    input_lock.unlock();
}

VectorXd BakedNetwork::getOutput()
{
    std::lock_guard<std::mutex> lg(output_lock);
    return output;
}

bool BakedNetwork::isUpdated()
{
    std::lock_guard<std::mutex> lg(output_lock);
    return updated;
}

void BakedNetwork::stop()
{
    stop_signal = true;
}

void BakedNetwork::feedForward()
{
    input_lock.lock();
    VectorXd activations = input;
    input_lock.unlock();

	for (auto L : layers)
	{
		activations = L->feedForward(activations);
	}
    if (activations == VectorXd::Zero(activations.size()))
    {
        qDebug() << "Outputs all zero";
        std::stringstream w;
        for (auto L : layers)
        {
            qDebug() << L->print().str().c_str();
        }
    }
    output_lock.lock();
	output = activations;
    updated = true;
    output_lock.unlock();
}
