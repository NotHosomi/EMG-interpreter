#include "RecurrentNetwork.h"
#include <iostream>
#include "Common.h"
#include <assert.h>
#include <fstream>

RecurrentNetwork::RecurrentNetwork(int input_size, double _alpha, std::string _name)
{
    name = _name;
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
// Load from file
RecurrentNetwork::RecurrentNetwork(std::ifstream& file, std::string _name, double _alpha)
{
    name = _name;
    int num_layers = 0;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
    file.read(reinterpret_cast<char*>(&INPUT_SIZE), sizeof(int));
    OUTPUT_SIZE = INPUT_SIZE;

    double file_alpha;
    file.read(reinterpret_cast<char*>(&file_alpha), sizeof(double));
    alpha = _alpha ? _alpha : file_alpha;

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

// to disable checkpointing, set best_loss to 0
void RecurrentNetwork::save(bool checkpoint)
{
    std::string filename = name;
    if (checkpoint)
    {
        filename += "_CKP";
    }
    std::ofstream file;
    file.open("nets/" + filename + ".dat", std::ios::binary | std::ios::out | std::ios::trunc);
    if (!file)
    {
        std::cout << "Failed to open \"nets/" << filename << ".dat\"" << std::endl;
        return;
    }

    int n = layers.size();
    file.write(reinterpret_cast<char*>(&n), sizeof(int));
    file.write(reinterpret_cast<char*>(&INPUT_SIZE), sizeof(int));
    file.write(reinterpret_cast<char*>(&alpha), sizeof(double));
    for (auto L : layers)
    {
        L->save(file);
    }
    std::cout << "Saved to \"nets/" + filename + ".dat\"" << std::endl;
    file.close();
}


// Takes one input & one label sequence. Returns average loss.
double RecurrentNetwork::trainSeq(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
{
    resize(inputs.size());
    double net_loss = 0;
    for (int t = 0; t < inputs.size(); ++t)
    {
        VectorXd outputs = feedForward(inputs[t]);
        net_loss += Common::loss(outputs, labels[t]).sum() / OUTPUT_SIZE;
    }
    backProp(labels);
    applyUpdates();
    clearCaches();
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
#if LOG_SEQ_TRAIN
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
        for (int t = 0; t < data.inputs[seq].size(); ++t)
        {
            VectorXd outputs = feedForward(data.inputs[seq][t]);
            local_loss += Common::loss(outputs, data.labels[seq][t]).sum() / OUTPUT_SIZE;

#if LOG_SEQ_TRAIN
            for (int i = 0; i < OUTPUT_SIZE; ++i)
            {
                log_seq << outputs[i] << " "
                    << data.labels[seq][t][i] << " ";
            }
#endif
#if LOG_ACC
            local_acc += Common::accuracy(outputs, data.labels[seq][i]);
#endif
        }

#if LOG_SEQ_TRAIN
        log_seq.close();
#endif
#if LOG_ACC
        net_acc += local_acc / data.inputs[seq].size();
        local_acc = 0;
#endif

        backProp(data.labels[seq]);
        applyUpdates();
        clearCaches();
        local_loss /= data.inputs[seq].size();
        net_loss += local_loss;
    }
#if LOG_ACC
    net_acc /= data.inputs.size();
    acc_log << net_acc << " ";
#endif
    net_loss /= data.inputs.size();
    return net_loss;
}

double RecurrentNetwork::evalSeq(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
{
    resize(inputs.size());
    double local_loss = 0;
    for (int i = 0; i < inputs.size(); ++i)
    {
        VectorXd outputs = feedForward(inputs[i]);
        local_loss += Common::loss(outputs, labels[i]).sum() / OUTPUT_SIZE;
    }
    clearCaches();
    local_loss /= inputs.size();
    return local_loss;
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
        net_acc += local_acc / data.inputs[seq].size();;
        local_acc = 0;
#endif

        local_loss /= data.inputs[seq].size();
        net_loss += local_loss;
    }
#if LOG_ACC
    acc_log << net_acc << " ";
#endif
    y_history.clear();
    for (auto L : layers)
    {
        L->clearCaches();
    }
    net_loss /= data.inputs.size();
    if (net_loss < best_loss && checkpoints_enabled)
    {
        best_loss = net_loss;
        save(true);
    }
    return net_loss;
}

double RecurrentNetwork::evalSetNoMetrics(Dataset<std::vector<VectorXd>> data)
{
    double net_loss = 0;
    for (int seq = 0; seq < data.inputs.size(); ++seq)
    {
        double local_loss = 0;
        resize(data.inputs[seq].size());
        for (int i = 0; i < data.inputs[seq].size(); ++i)
        {
            VectorXd outputs = feedForward(data.inputs[seq][i]);
            local_loss += Common::loss(outputs, data.labels[seq][i]).sum() / OUTPUT_SIZE;
        }
        local_loss /= data.inputs[seq].size();
        net_loss += local_loss;
    }
    clearCaches();
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

void RecurrentNetwork::gradCheck(std::vector<VectorXd> inputs, std::vector<VectorXd> labels)
{
    const double EPSILON = 10e-7;
    std::vector<double> theta;
    int pos = 0;
    for (auto L : layers)
    {
        // unwrap the weights
        L->writeWeightBuffer(theta, pos);
    }

#pragma region Approx Weight Grads

    std::vector<double> thetaPlus;
    std::vector<double> thetaMinus;
    VectorXd gradApprox(theta.size());
    double theta_plus_loss = 0;
    double theta_minus_loss = 0;

    // numerically approximate the gradient with respect to each weight
    for (int i = 0; i < theta.size(); ++i)
    {
        // calculate thetaPlus loss
        thetaPlus = theta;
        thetaPlus[i] += EPSILON;
        pos = 0;
        for (auto L : layers)
        {
            L->readWeightBuffer(thetaPlus, pos);
        }
        theta_plus_loss = evalSeq(inputs, labels);
        // calculate thetaMinus loss
        thetaMinus = theta;
        thetaMinus[i] -= EPSILON;
        pos = 0;
        for (auto L : layers)
        {
            L->readWeightBuffer(thetaMinus, pos);
        }
        theta_minus_loss = evalSeq(inputs, labels);
        gradApprox[i] = (theta_plus_loss - theta_minus_loss) / (2 * EPSILON);
    }
    gradApprox *= inputs.size(); // inverse of the actual grads being divided by 
#pragma endregion

    // compute Backprop's gradients
    pos = 0;
    for (auto L : layers)
    {
        // put the original weights back into the network
        L->readWeightBuffer(theta, pos);
    }
    resize(inputs.size());
    double net_loss = 0;
    for (int t = 0; t < inputs.size(); ++t)
    {
        VectorXd outputs = feedForward(inputs[t]);
        net_loss += Common::loss(outputs, labels[t]).sum() / OUTPUT_SIZE;
    }

#pragma region toplevel loss comparison
    // Top level loss confirmed accurate
    //for (int t = 0; t < inputs.size(); ++t)
    //{
    //    std::cout << "\n TIMESTEP: " << t << std::endl;
    //    std::cout << "\nRNN Yt:\n" << y_history[t];
    //    std::cout << "\nRNN Label:\n" << labels[t];
    //    VectorXd grad_dloss = Common::dloss(y_history[t], labels[t]);
    //    std::cout << "\ndY:\n" << grad_dloss << std::endl;
    //    VectorXd y_gradApprox(y_history[t].size());
    //    VectorXd y_plus;
    //    VectorXd y_minus;
    //    double y_plus_loss = 0;
    //    double y_minus_loss = 0;
    //    for (int i = 0; i < y_history[t].size(); ++i)
    //    {
    //        // calculate thetaPlus loss
    //        y_plus = y_history[t];
    //        y_plus[i] += EPSILON;
    //        y_plus_loss = Common::loss(y_plus, labels[t]).sum();
    //        // calculate thetaMinus loss
    //        y_minus = y_history[t];
    //        y_minus[i] -= EPSILON;
    //        y_minus_loss = Common::loss(y_minus, labels[t]).sum();
    //        y_gradApprox[i] = (y_plus_loss - y_minus_loss) / (2 * EPSILON);
    //    }
    //    std::cout << "\ndY approx:\n" << y_gradApprox << std::endl;
    //}
#pragma endregion
    backProp(labels);
    // Pull functional gradient vector from the network
    VectorXd dVec(theta.size());
    pos = 0;
    for (auto L : layers)
    {
        L->writeUpdateBuffer(dVec, pos);
    }
    clearCaches();

    // compare grad approx to the updates
    double ratio = (gradApprox - dVec).norm() / (gradApprox.norm() - dVec.norm());
    std::cout << "gradApprox difference: " << ratio << std::endl;

#pragma region OUTPUTTING

    std::cout << "check differences grads" << std::endl;
    VectorXd vDiff = (gradApprox - dVec).cwiseAbs();
    std::vector<double> diff_list;
    for (int i = 0; i < theta.size(); ++i)
    {
        diff_list.push_back(vDiff[i]);
    }
    pos = 0;
    for (auto L : layers)
    {
        // print out the gradient inaccuracies in respective matrices.
        L->readWeightBuffer(diff_list, pos);
        L->print();
    }


    std::cout << "backprop grads" << std::endl;
    std::vector<double> grad_list;
    for (int i = 0; i < theta.size(); ++i)
    {
        grad_list.push_back(dVec[i]);
    }
    pos = 0;
    for (auto L : layers)
    {
        // print out the gradient inaccuracies in respective matrices.
        L->readWeightBuffer(grad_list, pos);
        L->print();
    }


    std::cout << "approx grads" << std::endl;
    std::vector<double> approx_list;
    for (int i = 0; i < theta.size(); ++i)
    {
        approx_list.push_back(gradApprox[i]);
    }
    pos = 0;
    for (auto L : layers)
    {
        // print out the gradient inaccuracies in respective matrices.
        L->readWeightBuffer(approx_list, pos);
        L->print();
    }

#pragma endregion
    // return the weights to their original values
    pos = 0;
    for (auto L : layers)
    {
        L->readWeightBuffer(theta, pos);
    }
}

// TODO: Investigate why the corruption only occurs with multiple timesteps
void RecurrentNetwork::gradCheckAtT(std::vector<VectorXd> inputs, std::vector<VectorXd> labels, int timestep)
{
    const double EPSILON = 10e-7;
    std::vector<double> theta;
    int pos = 0;
    for (auto L : layers)
    {
        // unwrap the weights
        L->writeWeightBuffer(theta, pos);
    }

#pragma region Approx Weight Grads

    std::vector<double> thetaPlus;
    std::vector<double> thetaMinus;
    VectorXd gradApprox(theta.size());
    double theta_plus_loss = 0;
    double theta_minus_loss = 0;

    // numerically approximate the gradient with respect to each weight
    for (int i = 0; i < theta.size(); ++i)
    {
        // calculate thetaPlus loss
        thetaPlus = theta;
        thetaPlus[i] += EPSILON;
        pos = 0;
        for (auto L : layers)
        {
            L->readWeightBuffer(thetaPlus, pos);
        }
        theta_plus_loss = Common::loss(feedForward(inputs[timestep]), labels[timestep]).sum() / OUTPUT_SIZE;
        // calculate thetaMinus loss
        thetaMinus = theta;
        thetaMinus[i] -= EPSILON;
        pos = 0;
        for (auto L : layers)
        {
            L->readWeightBuffer(thetaMinus, pos);
        }
        theta_minus_loss = Common::loss(feedForward(inputs[timestep]), labels[timestep]).sum() / OUTPUT_SIZE;
        gradApprox[i] = (theta_plus_loss - theta_minus_loss) / (2 * EPSILON);
    }
#pragma endregion

#pragma region Calc Actual Grads

    // compute Backprop's gradients
    pos = 0;
    for (auto L : layers)
    {
        // put the original weights back into the network
        L->readWeightBuffer(theta, pos);
    }
    // FEEDFORWARD
    resize(inputs.size());
    for (int i = 0; i < inputs.size(); ++i)
    {
        VectorXd outputs = feedForward(inputs[i]);
        y_history.push_back(outputs);
    }
    // BACKPROP
    VectorXd grad = Common::dloss(y_history[timestep], labels[timestep]);
    for (auto L = layers.rbegin(); L != layers.rend(); ++L)
    {
        grad = (*L)->backProp(grad, timestep+1);
    }
    // pull gradients from net
    VectorXd dVec(theta.size());
    pos = 0;
    for (auto L : layers)
    {
        L->writeUpdateBuffer(dVec, pos);
    }
#pragma endregion


#pragma region OUTPUTTING
    std::cout << "\nTIMESTEP: " << timestep << std::endl;
    // compare grad approx to the updates
    double diff = (gradApprox - dVec).norm() / (gradApprox.norm() - dVec.norm());
    std::cout << "gradApprox difference: " << diff << std::endl;

    // difference per parameter
    std::cout << "check differences grads" << std::endl;
    VectorXd vDiff = (gradApprox - dVec).cwiseAbs();
    std::vector<double> diff_list;
    for (int i = 0; i < theta.size(); ++i)
    {
        diff_list.push_back(vDiff[i]);
    }
    pos = 0;
    for (auto L : layers)
    {
        // print out the gradient inaccuracies in respective matrices.
        L->readWeightBuffer(diff_list, pos);
        L->print();
    }


    std::cout << "backprop grads" << std::endl;
    std::vector<double> grad_list;
    for (int i = 0; i < theta.size(); ++i)
    {
        grad_list.push_back(dVec[i]);
    }
    pos = 0;
    for (auto L : layers)
    {
        // print out the gradient inaccuracies in respective matrices.
        L->readWeightBuffer(grad_list, pos);
        L->print();
    }


    std::cout << "approx grads" << std::endl;
    std::vector<double> approx_list;
    for (int i = 0; i < theta.size(); ++i)
    {
        approx_list.push_back(gradApprox[i]);
    }
    pos = 0;
    for (auto L : layers)
    {
        // print out the gradient inaccuracies in respective matrices.
        L->readWeightBuffer(approx_list, pos);
        L->print();
    }


    // return the weights to their original values
    std::cout << "Actual weights:" << std::endl;
    pos = 0;
    for (auto L : layers)
    {
        L->readWeightBuffer(theta, pos);
        L->print();
    }

    std::cout << "In:\n" << inputs[timestep].transpose() << std::endl;
    std::cout << "Y:\n" << y_history[timestep].transpose() << std::endl;

    clearCaches();
#pragma endregion
}
/*
    Gradient checking todo:
    check the grads of each timestep, find out WHICH timestep causes it
    maybe only happens if full Feedforward rather than single timestep, for example
    what are the possible sources of the issue
    write out what is different between single and multiple sample sequence
*/

void RecurrentNetwork::useCheckpoints(bool useCkp)
{
    checkpoints_enabled = useCkp;
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

    for (int t = labels.size(); t >= 1; --t)
    {
        //Debug
#ifdef PRINT_BP
        std::cout << "\nRNN Yt:\n" << y_history[t-1];
        std::cout << "\nRNN Label:\n" << labels[t-1];
#endif
        VectorXd grad = Common::dloss(y_history[t-1], labels[t-1]);
        net_cost += 2 * grad.sum() / grad.size();
        for (auto L = layers.rbegin(); L != layers.rend(); ++L)
        {
            grad = (*L)->backProp(grad, t);
        }
    }

    // Div each layer's updates by their cache size - 1

    y_history.clear();
    return std::abs(net_cost) / (labels.size());// -r_limit);
}

void RecurrentNetwork::applyUpdates()
{
    for (auto L : layers)
    {
        L->applyUpdates();
    }
}

void RecurrentNetwork::clearCaches()
{
    for (auto L : layers)
    {
        L->clearCaches();
    }
    y_history.clear();
}

void RecurrentNetwork::resize(size_t new_depth)
{
    for (auto L : layers)
    {
        L->resize(new_depth);
    }
}