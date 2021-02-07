#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "Lstm.h"

int INPUT_SIZE = 3;
int LABEL_SIZE = 5;
int CELL_SIZE = 10;

int main()
{
    bool training;
    // input "training? " y/n


#pragma region BUILD NET

    Lstm* lstm;
    std::string fileaddress;
    std::ifstream net_file;
    std::cout << "Cell File: ";
    std::cin >> fileaddress;
    net_file.open(fileaddress + ".dat", std::ios::binary || std::ios::in);
    if (net_file)
    {
        std::cout << "Loading " << fileaddress << ".dat" << std::endl;
        lstm = new Lstm(net_file, 0.15);
    }
    else
    {
        std::cout << "Creating new network \"fileaddress\"" << std::endl;
        lstm = new Lstm(INPUT_SIZE, CELL_SIZE, 0.15);
    }

#pragma endregion

#pragma region LOAD SAMPLES

    std::ifstream data_file;
    while (1)
    {
        std::cout << "Signal File: ";
        std::cin >> fileaddress;
        data_file.open(fileaddress + ".emg");
        if (data_file)
        {
            break;
        }
        std::cout << "Failed to open file " << fileaddress << ".emg" << std::endl;
    }

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

            VectorXd label(CELL_SIZE);
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

#pragma endregion

#pragma region TRAIN NET

    for (int seq = 0; seq < input_sequences.size(); ++seq)
    {
        lstm->resize(input_sequences[seq].size());
        for (auto& input : input_sequences[seq])
        {
            lstm->feedForward(input);
        }
        lstm->backProp(label_sequences[seq]);
    }

#pragma endregion
    return 0;
}
