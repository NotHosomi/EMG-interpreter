#include "RecurrentNetwork.h"
#include <iostream>
#include "Common.h"
#include <assert.h>
#include <fstream>


RecurrentNetwork::RecurrentNetwork(GenericLayer* first_layer, double _alpha)
{
    alpha = _alpha;
    INPUT_SIZE = first_layer->getInputSize();
    OUTPUT_SIZE = first_layer->getOutputSize();
    layers.push_back(first_layer);

#if LOG_ACC
    acc_log.open("logs/accuracy.txt", std::ios::trunc);
    if (!acc_log)
    {
        std::cout << "Failed to open file logs/accuracy.txt" << std::endl;
    }
#endif
}
RecurrentNetwork::RecurrentNetwork(int input_size, double _alpha)
{
    alpha = _alpha;
    INPUT_SIZE = input_size;
    OUTPUT_SIZE = input_size;

#if LOG_ACC
    acc_log.open("logs/accuracy.txt", std::ios::trunc);
    if (!acc_log)
    {
        std::cout << "Failed to open file logs/accuracy.txt" << std::endl;
    }
#endif
}

RecurrentNetwork::~RecurrentNetwork()
{
#if LOG_ACC
    acc_log.close();
#endif
}

void RecurrentNetwork::addLayer(char layer_type, int output_size)
{
    switch (layer_type)
    {
    case 'D': layers.emplace_back(new DenseLayer(OUTPUT_SIZE, output_size, alpha));
        break;
    case 'L': layers.emplace_back(new Lstm(OUTPUT_SIZE, output_size, alpha));
        break;
    case 'E': layers.emplace_back(new ElmanLayer(OUTPUT_SIZE, output_size, alpha));
        break;
    default: std::cout << "Invalid layer type " << layer_type << "\t(D, L, E)" << std::endl;
        return;
    }
    OUTPUT_SIZE = layers.back()->getOutputSize();
}


// Takes one input & one label sequence. Returns average loss.
double RecurrentNetwork::trainSeq(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
{
    resize(inputs.size());
    double net_loss = 0;
    for (int i = 0; i < inputs.size(); ++i)
    {
        VectorXd outputs = feedForward(inputs[i]);
        net_loss += Common::loss(outputs, labels[i]).sum() / OUTPUT_SIZE;
    }
    backProp(labels);
    net_loss /= inputs.size();
    return net_loss;
}

// Takes input & label sequences. Returns average loss.
double RecurrentNetwork::trainSet(Dataset<std::vector<VectorXd>> data)
{
    double net_loss = 0;
#if LOG_ACC
    double local_acc = 0;
    double net_acc = 0;
#endif
    for (int seq = 0; seq < data.inputs.size(); ++seq)
    {
#if LOG_SEQ
        std::ofstream log_seq("logs/train_seqs/seq_" + std::to_string(seq) + ".txt", std::ios::trunc);
        if (!log_seq)
        {
            std::cout << "Failed to open file logs/train_seq_" + std::to_string(seq) + ".txt" << std::endl;
            return -1;
        }
        log_seq << data.labels[seq].back().size() << " ";
#endif
        double local_loss = 0;
        resize(data.inputs[seq].size());
        for (int i = 0; i < data.inputs[seq].size(); ++i)
        {
            VectorXd outputs = feedForward(data.inputs[seq][i]);
            local_loss += Common::loss(outputs, data.labels[seq][i]).sum() / OUTPUT_SIZE;

#if LOG_SEQ
            for (int j = 0; j < OUTPUT_SIZE; ++j)
            {
                log_seq << outputs[j] << " "
                    << data.labels[seq][i][j] << " ";
            }
#endif
#if LOG_ACC
            local_acc += Common::accuracy(outputs, data.labels[seq][i]);
#endif
        }

#if LOG_SEQ
        log_seq.close();
#endif
#if LOG_ACC
        net_acc += local_acc;
        local_acc = 0;
#endif

        backProp(data.labels[seq]);
        local_loss /= data.inputs[seq].size();
        net_loss += local_loss;
    }
#if LOG_ACC
    acc_log << net_acc << " ";
#endif
    net_loss /= data.inputs.size();
    return net_loss;
}

double RecurrentNetwork::evalSeq(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
{
    resize(inputs.size());
    double net_loss = 0;
    for (int i = 0; i < inputs.size(); ++i)
    {
        VectorXd outputs = feedForward(inputs[i]);
        net_loss += Common::loss(outputs, labels[i]).sum() / OUTPUT_SIZE;
    }
    net_loss /= inputs.size();
    return net_loss;
}

double RecurrentNetwork::evalSet(Dataset<std::vector<VectorXd>> data)
{
    double net_loss = 0;
#if LOG_ACC
    double local_acc = 0;
    double net_acc = 0;
#endif
    for (int seq = 0; seq < data.inputs.size(); ++seq)
    {
#if LOG_SEQ
        std::ofstream log_seq("logs/test_seqs/seq_" + std::to_string(seq) + ".txt", std::ios::trunc);
        if (!log_seq)
        {
            std::cout << "Failed to open file logs/test_seq_" + std::to_string(seq) + ".txt" << std::endl;
            return -1;
        }
        log_seq << data.labels[seq].back().size() << " ";
#endif
        double local_loss = 0;
        resize(data.inputs[seq].size());
        for (int i = 0; i < data.inputs[seq].size(); ++i)
        {
            VectorXd outputs = feedForward(data.inputs[seq][i]);
            local_loss += Common::loss(outputs, data.labels[seq][i]).sum() / OUTPUT_SIZE;

#if LOG_SEQ
            for (int j = 0; j < OUTPUT_SIZE; ++j)
            {
                log_seq << outputs[j] << " "
                    << data.labels[seq][i][j] << " ";
            }
#endif
#if LOG_ACC
            local_acc += Common::accuracy(outputs, data.labels[seq][i]);
#endif
        }
#if LOG_SEQ
        log_seq.close();
#endif
#if LOG_ACC
        net_acc += local_acc;
        local_acc = 0;
#endif

        local_loss /= data.inputs[seq].size();
        net_loss += local_loss;
    }
#if LOG_ACC
    acc_log << net_acc << " ";
#endif
    net_loss /= data.inputs.size();
    return net_loss;
}


void RecurrentNetwork::print()
{
    for (auto L : layers)
    {
        std::cout << "\n---------------------------------------------------------" << std::endl;
        L->print();
    }
    std::cout << "\n---------------------------------------------------------" << std::endl;
}

// Internals
VectorXd RecurrentNetwork::feedForward(VectorXd input)
{
    assert(input.size() == INPUT_SIZE && "!!! Input size mismatch !!!");
    VectorXd activations = input;
    for (auto L : layers)
    {
        activations = L->feedForward(activations);
    }
    //activations = L1.feedForward(input);
    //activations = L2.feedForward(activations);
    //activations = L3.feedForward(activations);
    y_history.emplace_back(activations);
    assert(activations.size() == OUTPUT_SIZE && "!!! Output size mismatch !!!");
    return activations;
}

double RecurrentNetwork::backProp(std::vector<VectorXd> labels)
{
    assert(labels.back().size() == OUTPUT_SIZE && "!!! Label size mismatch !!!");
    double net_cost = 0.0;

    //std::cout << "\nlabel count:\t" << labels.size()
    //    << "\n output count:\t" << y_history.size();

    int r_limit = labels.size() < y_history.size() - 1 ? y_history.size() - labels.size() : 1;
    if (r_limit == 0)
        int d = 0;
    for (int t = labels.size(); t >= r_limit; --t)
    {
        //Debug
#ifdef PRINT_BP
        std::cout << "\nRNN Yt:\n" << y_history[t-1];
        std::cout << "\nRNN Label:\n" << labels[t-1];
#endif
        VectorXd grad = Common::dloss(y_history[t-1], labels[t-1]);
        net_cost += 2 * grad.sum() / grad.size();
        //grad = L3.backProp(grad, t);
        //grad = L2.backProp(grad, t);
        //L1.backProp(grad, t);
        for (auto L = layers.rbegin(); L != layers.rend(); ++L)
        {
            grad = (*L)->backProp(grad, t);
        }
    }

    for (auto L : layers)
    {
        L->applyUpdates();
    }
    //L1.applyUpdates();
    //L2.applyUpdates();
    //L3.applyUpdates();

    y_history.clear();
    return std::abs(net_cost) / (labels.size());// -r_limit);
}

void RecurrentNetwork::resize(size_t new_depth)
{
    for (auto L : layers)
    {
        L->resize(new_depth);
    }
    //L1.resize(new_depth);
    //L2.resize(new_depth);
    //L3.resize(new_depth);
}