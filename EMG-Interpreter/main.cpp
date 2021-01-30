#include <iostream>
#include <fstream>
#include <string>

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

const int INPUT_SIZE = 3;
const int LABEL_SIZE = 5;

int main()
{
    bool training;
    // input "training? " y/n

    std::string fileaddress;
    std::ifstream file;
    while (1)
    {
        std::cout << "Sample File: ";
        std::cin >> fileaddress;
        file.open(fileaddress + ".emg");
        if (file)
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
    while (std::getline(file, line))
    {
        input_sequences.emplace_back();
        label_sequences.emplace_back();
        std::string sample;
        std::stringstream linestream(line);
        while (getline(linestream, sample, '!'))
        {
            std::string value;
            std::stringstream samplestream(sample);
            while (getline(samplestream, value, '!'))
            {
                values.emplace_back(std::stoi(value));
            }

            VectorXd input(INPUT_SIZE);
            input << values[0], values[1], values[2];
            input /= 1024.0; // 1024 is the range of the Myoware signal output
            input_sequences.back().emplace_back(input);

            VectorXd label(LABEL_SIZE);
            label << values[3], values[4], values[5], values[6], values[7];
            label_sequences.back().emplace_back(label);

            values.clear();
        }
    }
    file.close();

    for (int seq = 0; seq < input_sequences.size(); ++seq)
    {
        for (auto& input : input_sequences[seq])
        {
            //feedforward(input)
        }
        //backprop(label_sequences[seq])
    }
}
