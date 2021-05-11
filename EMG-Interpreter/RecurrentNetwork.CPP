#include "RecurrentNetwork.h"
#include <iostream>

void RecurrentNetwork::train(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
{
    resize(inputs.size());
    for (auto& input : inputs)
    {
        feedForward(input);
    }
    backProp(labels);
}

double RecurrentNetwork::trainEx(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
{
    resize(inputs.size());
    double net_loss = 0;
    for (int i = 0; i < inputs.size(); ++i)
    {
        VectorXd outputs = feedForward(inputs[i]);
        net_loss += loss(outputs, labels[i]).sum() / OUTPUT_SIZE;
    }
    backProp(labels);
    net_loss /= inputs.size();
    return net_loss;
}

double RecurrentNetwork::eval(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
{
    resize(inputs.size());
    double net_loss = 0;
    for (int i = 0; i < inputs.size(); ++i)
    {
        VectorXd outputs = feedForward(inputs[i]);
        net_loss += loss(outputs, labels[i]).sum() / OUTPUT_SIZE;
    }
    net_loss /= inputs.size();
    return net_loss;
}

VectorXd RecurrentNetwork::feedForward(VectorXd input)
{
    VectorXd activations = L1.feedForward(input);
    activations = L2.feedForward(activations);
    activations = L3.feedForward(activations);
    y_history.emplace_back(activations);
    return activations;
}

double RecurrentNetwork::backProp(std::vector<VectorXd> labels)
{
    double net_cost = 0.0;

    int r_limit = labels.size() < y_history.size() - 1 ? y_history.size() - labels.size() : 1;
    for (int t = labels.size()-1; t >= r_limit; --t)
    {
        VectorXd grad = dloss(labels[t], y_history[t]);
        net_cost += 2 * grad.sum() / grad.size();
        grad = L3.backProp(grad, t);
        grad = L2.backProp(grad, t);
        L1.backProp(grad, t);
    }

    L1.applyUpdates();
    L2.applyUpdates();
    L3.applyUpdates();

    y_history.clear();
    VectorXd empt = VectorXd(OUTPUT_SIZE);
    empt.setZero();
    y_history.emplace_back(empt);
    return std::abs(net_cost) / (labels.size() - r_limit);
}

void RecurrentNetwork::resize(int new_depth)
{
    L1.resize(new_depth);
    L2.resize(new_depth);
    L3.resize(new_depth);
}

VectorXd RecurrentNetwork::loss(VectorXd outputs, VectorXd targets)
{
    return (outputs - targets).cwiseAbs();
    //VectorXd dloss(OUTPUT_SIZE);
    //for (int i = 0; i < OUTPUT_SIZE; ++i)
    //{
    //    dloss[i] = targets[i] * (outputs[i] - 1) + (1 - targets[i]) * outputs[i];
    //}
    //return dloss;
}

VectorXd RecurrentNetwork::dloss(VectorXd outputs, VectorXd targets)
{
    return 2 * (outputs - targets);
    //VectorXd loss(OUTPUT_SIZE);
    //for (int i = 0; i < OUTPUT_SIZE; ++i)
    //{
    //    loss[i] = -targets[i] * log(outputs[i]) - (1 - targets[i]) * log(1 - outputs[i]);
    //}
    //return loss;
}

// Binary Cross-Entropy Loss
// CE = -SUM(target[i] * log(output[i])
// 
// CE[i] = -target[i] * log(output[i]) - (1 - target[i]) * log(1 - output[i])
// dCE[i] = target[i] * (output[i] - 1) + (1-target[i]) * output[i]