#include "DeepNetwork.h"
#include "Common.h"
#include <iostream>

double DeepNetwork::train(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
{
    double net_loss = 0;
    for (int i = 0; i < inputs.size(); ++i)
    {
        feedForward(inputs[i]);
        net_loss += Common::loss(y, labels[i]).sum() / OUTPUT_SIZE;
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
        net_loss += Common::loss(y, labels[i]).sum() / OUTPUT_SIZE;
    }
    net_loss /= inputs.size();
    return net_loss;
}

VectorXd DeepNetwork::run(VectorXd inputs)
{
    feedForward(inputs);
    return y;
}

// Internals
void DeepNetwork::feedForward(VectorXd input)
{
    VectorXd a = L1.feedForward(input);
    //a = L2.feedForward(a);
    //a = L3.feedForward(a);
    y = a; 
}

void DeepNetwork::backProp(VectorXd label)
{
    VectorXd grad = Common::dloss(y, label);
    //grad = L3.backProp(grad, 1);
    //grad = L2.backProp(grad, 1);
    grad = L1.backProp(grad, 1);

    L1.applyUpdates();
    //L2.applyUpdates();
    //L3.applyUpdates();
}
