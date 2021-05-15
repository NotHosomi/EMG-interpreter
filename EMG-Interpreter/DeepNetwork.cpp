#include "DeepNetwork.h"
#include <iostream>

double DeepNetwork::train(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
{
    double net_loss = 0;
    for (int i = 0; i < inputs.size(); ++i)
    {
        feedForward(inputs[i]);
        net_loss += loss(y, labels[i]).sum() / OUTPUT_SIZE;
        backProp(labels[i]);
    }
    net_loss /= inputs.size();
    return net_loss;
}

double DeepNetwork::eval(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
{
    double net_loss = 0;
    for (int i = 0; i < inputs.size(); ++i)
    {
        feedForward(inputs[i]);
        net_loss += loss(y, labels[i]).sum() / OUTPUT_SIZE;
    }
    net_loss /= inputs.size();
    return net_loss;
}

// Internals
void DeepNetwork::feedForward(VectorXd input)
{
    VectorXd a = L1.feedForward(input);
    a = L2.feedForward(a);
    y = a; 
}

void DeepNetwork::backProp(VectorXd label)
{
    VectorXd grad = dloss(y, label);
    grad = L2.backProp(grad, 1);
    grad = L1.backProp(grad, 1);

    L1.applyUpdates();
    L2.applyUpdates();
}

VectorXd DeepNetwork::loss(VectorXd outputs, VectorXd targets)
{
    return (outputs - targets).cwiseAbs();
}

VectorXd DeepNetwork::dloss(VectorXd outputs, VectorXd targets)
{
    return (outputs - targets);
    //ArrayXd out = outputs.array();
    //return ((out - targets.array()) * out * (1 - out)).matrix();
}
