#include "Common.h"
#include <iostream>

#pragma region NONLINEARITIES

double Common::sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double Common::dsigmoid(double x)
{
    return (1 / (1 + exp(-x))) * (1 - (1 / (1 + exp(-x))));
}

double Common::tangent(double x)
{
    return tanh(x);
}

double Common::dtangent(double x)
{
    //return (1 / cosh(x)) * (1 / cosh(x)); // is cosh unsafe?

    // simplified tanh derivative
    double th = tanh(x);
    return 1 - th * th;
}
#pragma endregion

#pragma region METRICS

// Multiclass Binary Cross-Entropy
// CE = T log(y) + (1 - t) log(1-y)
// dCE = -(T / y) + (1 - T) / (1 - y)
VectorXd Common::loss(VectorXd outputs, VectorXd targets)
{
    double eps = 1e-8; // epsilon, used to prevent log(0) (NaN)
    VectorXd loss(outputs.size());
    for (int i = 0; i < outputs.size(); ++i)
    {
        if (outputs[i] == 0)
            outputs[i] += eps;
        else if (outputs[i] == 1)
            outputs[i] -= eps;

        loss[i] = -(targets[i] * log(outputs[i]) + (1 - targets[i]) * log(1 - outputs[i]));
    }
    //std::cout << "Loss: " << loss << "\nY " << outputs << "\nT " << targets << std::endl;
    return loss;
}

VectorXd Common::dloss(VectorXd outputs, VectorXd targets)
{
#ifdef PRINT_BP
    std::cout << "\n---------------------------------------------------------------------------------------------------"
        << "\nY\n" << outputs << "\nT\n" << targets << std::endl;
#endif
    double eps = 1e-8; // epsilon, used to prevent log(0) (NaN)
    VectorXd dloss(outputs.size());
    for (int i = 0; i < outputs.size(); ++i)
    {
        if (outputs[i] == 0)
            outputs[i] += eps;
        else if (outputs[i] == 1)
            outputs[i] -= eps;

#ifdef UNIFIER
        // traditional + unifier
        double unifier = 2 * targets[i] - 1; // will either be 1 or -1
        dloss[i] = -(targets[i] / outputs[i]) + (1 - targets[i]) / (1 - outputs[i]) + unifier;
#else
        dloss[i] = -(targets[i] / outputs[i]) + (1 - targets[i]) / (1 - outputs[i]);
#endif

        //double unifier = 2 * targets[i] - 1; // will either be 1 or -1
        // traditional + unifier
        dloss[i] = -(targets[i] / outputs[i]) + (1 - targets[i]) / (1 - outputs[i]);// +unifier;
    }
#ifdef PRINT_DLOSS
    std::cout << "dloss:\t" << dloss << "\nY\t" << outputs << "\nT\t" << targets << std::endl;
#endif
    return dloss;
}

double Common::accuracy(VectorXd outputs, VectorXd targets)
{
    double diff = 0;
    for (int i = 0; i < outputs.size(); ++i)
    {
        diff += abs(outputs[i] - targets[i]);
    }
    return diff / outputs.size();
}
#pragma endregion