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

#define RNN 0
#define DNN !RNN
#define RNN_TEST 1
#define RNN_LOADING 0
#define LOGGING 0


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

#if RNN_LOADING
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

#if !RNN_TEST
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
    //std::vector<Dataset<std::vector<VectorXd>>> batches;
    //for (int i = 0; i < 10; ++i)
    //    batches.push_back(UnitTests::MealSequence(30, 30));
    //Dataset<std::vector<VectorXd>> train = UnitTests::MealSequence(500, 20);
    //Dataset<std::vector<VectorXd>> test = UnitTests::MealSequence(100, 20);

    //Dataset<std::vector<VectorXd>> train = UnitTests::LstmDebugA(80, 10);
    //Dataset<std::vector<VectorXd>> test = UnitTests::LstmDebugA(20, 10);
    
    //Dataset<std::vector<VectorXd>> train = UnitTests::LstmMaximizer(80, 10, 4, 1, false);
    //Dataset<std::vector<VectorXd>> test = UnitTests::LstmMaximizer(20, 10, 4, 1, false);
     
    Dataset<std::vector<VectorXd>> train = UnitTests::LstmSingle(800, UnitTests::GateType::AND);
    Dataset<std::vector<VectorXd>> test = UnitTests::LstmSingle(200, UnitTests::GateType::AND);
    
    //Dataset<std::vector<VectorXd>> train = UnitTests::Noise(80, 10, 4, 1); 
    //Dataset<std::vector<VectorXd>> test = UnitTests::Noise(20, 10, 4, 1);
#endif

#pragma endregion

#pragma region BEGIN LOG
#if LOGGING
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

    for (int epoch = 0; epoch < 10; ++epoch)
    {
        std::cout << "Epoch:\t" << epoch << std::endl;
        double train_err = 0;
        double test_err = 0;
        // Train
        //err = rnn->trainSeqBatch(batches[epoch].inputs, batches[epoch].labels);
        train_err = rnn->trainSeqBatch(train.inputs, train.labels);
        std::cout << std::to_string(train_err) << std::endl;
        // Test
        test_err = rnn->evalSeqBatch(test.inputs, test.labels);
        std::cout << std::to_string(test_err) << std::endl;
#if LOGGING
        log << std::to_string(train_err) << " ";
        log << std::to_string(test_err) << " ";
#endif
    }

#pragma endregion


#if LOGGING
    log.close();
#endif
    delete rnn;
#elif DNN
    DeepNetwork dnn;
    Dataset<VectorXd> train = UnitTests::gate(30, UnitTests::AND);
    Dataset<VectorXd> test = UnitTests::gate(20, UnitTests::AND);
    for (int epoch = 0; epoch < 30; ++epoch)
    {
        std::cout << "Epoch:\t" << epoch << std::endl;
        std::cout << dnn.train(train.inputs, train.labels) << std::endl;
        std::cout << dnn.eval(test.inputs, test.labels) << std::endl;
    }

    VectorXd in = VectorXd(2);
    in.setZero();
    std::cout << "\nAND Test:\n" << dnn.run(in) << std::endl;
    in[1] = 1;
    std::cout << dnn.run(in) << std::endl;
    in[0] = 1;
    in[1] = 0;
    std::cout << dnn.run(in) << std::endl;
    in[1] = 1;
    std::cout << dnn.run(in) << std::endl;
#endif

    return 0;
}
