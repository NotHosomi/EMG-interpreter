#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "RecurrentNetwork.h"

#define INPUT_SIZE 3
#define LABEL_SIZE 5

int main()
{
#pragma region BUILD NET

#ifdef RNN_LOADING
    RecurrentNetwork* rnn;
    std::string fileaddress;
    std::ifstream net_file;
    std::cout << "Cell File: ";
    std::cin >> fileaddress;
    net_file.open("nets/" + fileaddress + ".dat", std::ios::binary || std::ios::in);
    if (net_file)
    {
        std::cout << "Loading 'nets/" << fileaddress << ".dat'" << std::endl;
        rnn = new Lstm(net_file, 0.15);
    }
    else
    {
        std::cout << "Creating new network \"fileaddress.dat\"" << std::endl;
        rnn = new RecurrentNetwork(INPUT_SIZE, LABEL_SIZE, 0.15);
    }
#else
    RecurrentNetwork* rnn = new RecurrentNetwork();
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
                values.emplace_back(std::stoi(value));
            }

            VectorXd input(INPUT_SIZE);
            for (int i = 0; i < INPUT_SIZE; ++i)
            {
                input[i] = values[i];
            }
            input /= 1024.0; // 1024 is the range of the Myoware signal output
            input_sequences.back().emplace_back(input);

            VectorXd label(LABEL_SIZE);
            label.setZero();
            for (int i = 0; i < values.size() - INPUT_SIZE; ++i)
            {
                label[i] = values[i + INPUT_SIZE];
            }
            label_sequences.back().emplace_back(label);

            values.clear();
        }
    }
    data_file.close();
    std::cout << "Signal data mounted" << std::endl;

#pragma endregion

#pragma region TRAIN NET

    for (int seq = 0; seq < input_sequences.size(); ++seq)
    {
        std::cout << "Training on sequence " << std::to_string(seq) << std::endl;
        rnn->resize(input_sequences[seq].size());
        for (auto& input : input_sequences[seq])
        {
            rnn->feedForward(input);
        }
        double avg_err = rnn->backProp(label_sequences[seq]);
        avg_err /= input_sequences[seq].size();
        std::cout << "Avg Error: " << std::to_string(avg_err) << std::endl;
    }
    // TODO: implement network saving
    //rnn->save();

#pragma endregion

    delete rnn;
    return 0;
}
