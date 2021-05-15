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
    //return (outputs - targets).cwiseAbs();
    VectorXd loss(OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        loss[i] = targets[i] * (outputs[i] - 1) + (1 - targets[i]) * outputs[i];
    }
    return loss;
}

VectorXd DeepNetwork::dloss(VectorXd outputs, VectorXd targets)
{
    double eps = 1e-8; // epsilon, used to prevent log(0) (not a real number)
    VectorXd dloss(OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        if (outputs[i] == 0)
            outputs[i] += eps;
        else if (outputs[i] == 1)
            outputs[i] -= eps;
        dloss[i] = -targets[i] * log(outputs[i]) - (1 - targets[i]) * log(1 - outputs[i]);

        // correct for the derivative being in the wrong direction when the delta SHOULD be negative!
        // This is totally not a hack
        if (outputs[i] > targets[i])
            dloss[i] *= -1;
    }
    return dloss;
}
