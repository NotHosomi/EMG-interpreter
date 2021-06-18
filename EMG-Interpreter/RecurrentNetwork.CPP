#include "RecurrentNetwork.h"
#include <iostream>
#include "Common.h"

//#define CROSS_ENTROPY

//RecurrentNetwork::RecurrentNetwork(std::vector<std::tuple<char, int>> topology, int input_size)
//{
//    INPUT_SIZE = input_size;
//    OUTPUT_SIZE = std::get<1>(topology.back());
//
//    for (auto& L : topology)
//    {
//
//    }
//}

// Takes one input & one label sequence.
void RecurrentNetwork::train(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
{
    resize(inputs.size());
    for (auto& input : inputs)
    {
        feedForward(input);
    }
    backProp(labels);
}

// Takes one input & one label sequence. Returns average loss.
double RecurrentNetwork::trainEx(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
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
double RecurrentNetwork::trainSeqBatch(std::vector<std::vector<VectorXd>> input_sequences, std::vector<std::vector<VectorXd>> label_sequences)
{
    double net_loss = 0;
    for (int seq = 0; seq < input_sequences.size(); ++seq)
    {
        double local_loss = 0;
        resize(input_sequences[seq].size());
        for (int i = 0; i < input_sequences[seq].size(); ++i)
        {
            VectorXd outputs = feedForward(input_sequences[seq][i]);
            local_loss += Common::loss(outputs, label_sequences[seq][i]).sum() / OUTPUT_SIZE;
        }
        backProp(label_sequences[seq]);
        local_loss /= input_sequences[seq].size();
        net_loss += local_loss;
    }
    net_loss /= input_sequences.size();
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

double RecurrentNetwork::evalSeqBatch(std::vector<std::vector<VectorXd>> input_sequences, std::vector<std::vector<VectorXd>> label_sequences)
{
    double net_loss = 0;
    for (int seq = 0; seq < input_sequences.size(); ++seq)
    {
        double local_loss = 0;
        resize(input_sequences[seq].size());
        for (int i = 0; i < input_sequences[seq].size(); ++i)
        {
            VectorXd outputs = feedForward(input_sequences[seq][i]);
            local_loss += Common::loss(outputs, label_sequences[seq][i]).sum() / OUTPUT_SIZE;
        }
        local_loss /= input_sequences[seq].size();
        net_loss += local_loss;
    }
    net_loss /= input_sequences.size();
    return net_loss;
}


// Internals
VectorXd RecurrentNetwork::feedForward(VectorXd input)
{
    VectorXd activations = L1.feedForward(input);
    //activations = L2.feedForward(activations);
    //activations = L3.feedForward(activations);
    y_history.emplace_back(activations);
    return activations;
}

double RecurrentNetwork::backProp(std::vector<VectorXd> labels)
{
    double net_cost = 0.0;

    int r_limit = labels.size() < y_history.size() - 1 ? y_history.size() - labels.size() : 1;
    for (int t = labels.size()-1; t >= r_limit; --t)
    {
        VectorXd grad = Common::dloss(y_history[t], labels[t]);
        net_cost += 2 * grad.sum() / grad.size();
        //grad = L3.backProp(grad, t);
        //grad = L2.backProp(grad, t);
        L1.backProp(grad, t);
    }

    L1.applyUpdates();
    //L2.applyUpdates();
    //L3.applyUpdates();

    y_history.clear();
    VectorXd empt = VectorXd(OUTPUT_SIZE);
    empt.setZero();
    y_history.emplace_back(empt);
    return std::abs(net_cost) / (labels.size() - r_limit);
}

void RecurrentNetwork::resize(int new_depth)
{
    L1.resize(new_depth);
    //L2.resize(new_depth);
    //L3.resize(new_depth);
}