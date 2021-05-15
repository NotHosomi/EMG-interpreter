#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include "UnitTests.h"

#include "RecurrentNetwork.h"
#include "DeepNetwork.h"

#define INPUT_SIZE 3
#define LABEL_SIZE 5

#define DNN 1
#define RNN 0


MatrixXd mtable(VectorXd rows, RowVectorXd cols)
{
    return rows.replicate(1, cols.size()).cwiseProduct(
        cols.replicate(rows.size(), 1)
    );
}

int main()
{
#if RNN
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

#if 0
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

            VectorXd label(LABEL_SIZE);
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
#else
    std::vector<Dataset<std::vector<VectorXd>>> batches;
    //for (int i = 0; i < 10; ++i)
    //    batches.push_back(UnitTests::MealSequence(30, 30));
    Dataset<std::vector<VectorXd>> train = UnitTests::MealSequence(500, 50);
    Dataset<std::vector<VectorXd>> test = UnitTests::MealSequence(50, 50);
#endif

#pragma endregion

#pragma region BEGIN LOG
#if 1
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    time_t tt = std::chrono::system_clock::to_time_t(now);
    tm time;
    localtime_s(&time, &tt);
    std::string filename = "2" + std::to_string(time.tm_year) + "-"
        + std::to_string(time.tm_mon) + "-"
        + std::to_string(time.tm_mday) + "_"
        + std::to_string(time.tm_hour) + "-"
        + std::to_string(time.tm_min) + "-"
        + std::to_string(time.tm_sec);
    std::ofstream log("logs/" + filename + ".txt", std::ios::app);
    if (!log)
    {
        std::cout << "Failed to open file " << filename << ".txt" << std::endl;
        return 0xBAADDA7E;
    }
#endif
#pragma endregion

#pragma region TRAIN NET

    for (int epoch = 0; epoch < 100; ++epoch)
    {
        std::cout << "Epoch:\t" << epoch << std::endl;
        double err = 0;
        // Train
        //err = rnn->trainSeqBatch(batches[epoch].inputs, batches[epoch].labels);
        err = rnn->trainSeqBatch(train.inputs, train.labels);
        std::cout << std::to_string(err) << std::endl;
        log << std::to_string(err) << " ";
        // Test
        err = rnn->evalSeqBatch(test.inputs, test.labels);
        std::cout << std::to_string(err) << std::endl;
        log << std::to_string(err) << " ";
    }

#pragma endregion



    //log.close();
    delete rnn;
#elif DNN
    DeepNetwork dnn;
    Dataset<VectorXd> train = UnitTests::gate(30, UnitTests::AND);
    //Dataset<VectorXd> test = UnitTests::gate(20, UnitTests::OR); 
    for (int epoch = 0; epoch < 30; ++epoch)
    {
        std::cout << "Epoch:\t" << epoch << std::endl;
        std::cout << dnn.train(train.inputs, train.labels) << std::endl;
    }
#endif

    return 0;
}
