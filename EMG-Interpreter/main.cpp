#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "Lstm.h"

int INPUT_SIZE = 3;
int LABEL_SIZE = 5;
int CELL_SIZE = 10;

#define _NO_CELL_LOADING

int main()
{
#pragma region BUILD NET

#ifndef _NO_CELL_LOADING
    Lstm* lstm;
    std::string fileaddress;
    std::ifstream net_file;
    std::cout << "Cell File: ";
    std::cin >> fileaddress;
    net_file.open("nets/" + fileaddress + ".dat", std::ios::binary || std::ios::in);
    if (net_file)
    {
        std::cout << "Loading 'nets/" << fileaddress << ".dat'" << std::endl;
        lstm = new Lstm(net_file, 0.15);
    }
    else
    {
        std::cout << "Creating new network \"fileaddress.dat\"" << std::endl;
        lstm = new Lstm(INPUT_SIZE, CELL_SIZE, 0.15);
    }
#else
    Lstm* lstm = new Lstm(INPUT_SIZE, CELL_SIZE, 0.15);
    std::string fileaddress;
#endif

#pragma endregion

#pragma region LOAD SAMPLES

    std::ifstream data_file;
    while (1)
    {
        std::cout << "Signal File: ";
        std::cin >> fileaddress;
        data_file.open("data/" + fileaddress + ".emg");
        if (data_file)
        {
            break;
        }
        std::cout << "Failed to open file 'data/" << fileaddress << ".emg'" << std::endl;
    }
    std::cout << "Mounting signal data..." << std::endl;

    // 8 = input
    std::vector<int> values;
    std::vector<std::vector<VectorXd>> input_sequences;
    std::vector<std::vector<VectorXd>> label_sequences;
    std::string line;
    while (std::getline(data_file, line))
    {
        input_sequences.emplace_back();
        label_sequences.emplace_back();
        std::string sample;
        std::stringstream linestream(line);
        while (getline(linestream, sample, '!'))
        {
            std::string value;
            std::stringstream samplestream(sample);
            while (getline(samplestream, value, '-'))
            {
                try
                {
                    values.emplace_back(std::stoi(value));
                }
                catch (...)
                {
                    std::cout << "BAD SAMPLE: \"" << sample << "\"" << std::endl;
                    values.clear();
                    // TODO: Do I need to pop back of input/label sequences
                    // Yes yes, I know GOTO is bad, but its the cleanest way to escape nested loops
                    goto panic;
                }
            }

            VectorXd input(INPUT_SIZE);
            for (int i = 0; i < INPUT_SIZE; ++i)
            {
                input[i] = values[i];
            }
            input /= 1024.0; // 1024 is the range of the Myoware signal output
            input_sequences.back().emplace_back(input);

            VectorXd label(CELL_SIZE);
            label.setZero();
            for (int i = 0; i < values.size() - INPUT_SIZE; ++i)
            {
                label[i] = values[i + INPUT_SIZE];
            }
            label_sequences.back().emplace_back(label);

            values.clear();
        }
        // Bad sample jump point
        panic:;
    }
    data_file.close();
    std::cout << "Signal data mounted" << std::endl;

#pragma endregion

#pragma region TRAIN NET
    for (int epoch = 0; epoch < 10; ++epoch)
    {
        std::cout << "Epoch:\t" << epoch << std::endl;
        double net_error = 0;
        double smoothed_error = 0;
        const double smoothing = 0.3;
        for (int seq = 0; seq < input_sequences.size(); ++seq)
        {
            //std::cout << "Training on sequence " << std::to_string(seq) << std::endl;

            lstm->resize(input_sequences[seq].size());
            for (auto& input : input_sequences[seq])
            {
                lstm->feedForward(input);
            }
            double avg_err = lstm->backProp(label_sequences[seq]);
            avg_err /= input_sequences[seq].size();

            std::cout << "Last Error:\t" << std::to_string(avg_err) << std::endl;
            //smoothed_error = (smoothed_error * smoothing + avg_err) / (smoothing + 1);
            //std::cout << "Smoothed Error:\t" << std::to_string(smoothed_error) << std::endl;
            net_error += avg_err;
        }
        net_error /= input_sequences.size();
        std::cout << "Error:\t" << std::to_string(net_error) << std::endl;
    }

    //lstm->saveCell();

#pragma endregion

    delete lstm;
    return 0;
}
