#include "RecurrentNetwork.h"
#include <iostream>

double RecurrentNetwork::train(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
{
    resize(inputs.size());
    for (auto& input : inputs)
    {
        feedForward(input);
    }
    return backProp(labels);
}

double RecurrentNetwork::train_x(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
{
    resize(inputs.size());
    for (int i = 0; i < inputs.size(); ++i)
    {
        VectorXd outputs = feedForward(inputs[i]);
        for (int j = 0; j < labels.back().size(); ++j)
        {
            std::cout << std::to_string(outputs[i]) << "\t" << labels[i][j] << "\n";
        }
        std::cout << std::endl;
    }
    return backProp(labels);
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
        VectorXd grad = 2 * (labels[t] - y_history[t]); // TODO: switch to sigmoid cross-entropy
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
