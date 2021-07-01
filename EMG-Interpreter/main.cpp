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

#define RNN 1
#define DNN !RNN
#define RNN_TEST 1
#define RNN_LOADING 0
#define LOGGING 1
#define LOG_NEW_FILE 0

int main()
{
    // Eigen's matrix random init uses rand()
    srand(static_cast<int>(time(0)));
#if RNN
#pragma region BUILD NET

    std::cout << "Building model" << std::endl;
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
#if !RNN_TEST
    RecurrentNetwork* rnn = new RecurrentNetwork(new Lstm(1, 16, 0.15), 0.15);
    rnn->addLayer('L', 16);
    rnn->addLayer('D', 3);
#else
    RecurrentNetwork* rnn = new RecurrentNetwork(1, 0.15);
    rnn->addLayer('L', 6);
    rnn->addLayer('L', 6);
    rnn->addLayer('D', 3);
#endif
#endif
#pragma endregion

#pragma region LOAD SAMPLES

#if !RNN_TEST
    std::ifstream data_file;
    //std::string fileaddress;
    //while (1)
    //{
    //    std::cout << "Signal File: ";
    //    std::cin >> fileaddress;
    //    data_file.open("data/" + fileaddress + ".emg");
    //    if (data_file)
    //    {
    //        break;
    //    }
    //    std::cout << "Failed to open file 'data/" << fileaddress << ".emg'" << std::endl;
    //}
    data_file.open("data/bulk.emg");
    if (!data_file)
    {
        std::cout << "Failed to open file 'data/bulk.emg'" << std::endl;
    }

    std::cout << "Mounting signal data..." << std::endl;

    // 8 = input
    std::vector<int> values;
    Dataset<std::vector<VectorXd>> train_set;
    std::string line;
    int seq_id = -1;
    while (std::getline(data_file, line))
    {
        ++seq_id;
        train_set.inputs.emplace_back();
        train_set.labels.emplace_back();
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
                    std::cout << "BAD SAMPLE: \"" << sample << "\" in sequence " << seq_id << " of file" << std::endl;
                    values.clear();
                    // TODO: Do I need to pop back of input/label sequences??
                    train_set.inputs.pop_back();
                    train_set.labels.pop_back();
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
            train_set.inputs.back().emplace_back(input);

            VectorXd label(LABEL_SIZE);
            label.setZero();
            for (int i = 0; i < values.size() - INPUT_SIZE; ++i)
            {
                label[i] = values[i + INPUT_SIZE];
            }
            train_set.labels.back().emplace_back(label);

            values.clear();
        }
        // Bad sample jump point
        panic:;
    }
    data_file.close();
    std::cout << "Signal data mounted" << std::endl;

    //train_set.shuffle();
    Dataset<std::vector<VectorXd>> test_set = train_set.split(0.9);
    std::cout << "Train split:\t" << train_set.inputs.size()
        << "\tTest split:\t" << test_set.inputs.size() << std::endl;
#else
    //std::vector<Dataset<std::vector<VectorXd>>> batches;
    //for (int i = 0; i < 10; ++i)
    //    batches.push_back(UnitTests::MealSequence(30, 30));
    Dataset<std::vector<VectorXd>> train_set = UnitTests::MealSequence(500, 50);
    Dataset<std::vector<VectorXd>> test_set = UnitTests::MealSequence(100, 20);

    //Dataset<std::vector<VectorXd>> train_set = UnitTests::LstmDebugA(80, 10);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::LstmDebugA(20, 10);
    
    //Dataset<std::vector<VectorXd>> train_set = UnitTests::LstmMaximizer(80, 10, 4, 1, false);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::LstmMaximizer(20, 10, 4, 1, false);
     
    //Dataset<std::vector<VectorXd>> train_set = UnitTests::SeqSingle(800, UnitTests::GateType::AND);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::SeqSingle(200, UnitTests::GateType::AND);

    //Dataset<std::vector<VectorXd>> train_set = UnitTests::SeqGates(120, 10, UnitTests::GateType::AND);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::SeqGates(20, 10, UnitTests::GateType::AND);

    //Dataset<std::vector<VectorXd>> train_set = UnitTests::Toggle(120, 10);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::Toggle(20, 10);

    //Dataset<std::vector<VectorXd>> train_set = UnitTests::SeqGatesLinked(20, 20, UnitTests::GateType::NAND);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::SeqGatesLinked(5, 20, UnitTests::GateType::NAND);

    //Dataset<std::vector<VectorXd>> train_set = UnitTests::ElmanXORSet(500, 20);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::ElmanXORSet(50, 13);
    
    //Dataset<std::vector<VectorXd>> train_set = UnitTests::Noise(80, 10, 4, 1); 
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::Noise(20, 10, 4, 1);
#endif

#pragma endregion

#pragma region BEGIN LOG
    std::ofstream log("logs/loss.txt", std::ios::trunc);
    if (!log)
    {
        std::cout << "Failed to open file 'logs/loss.txt'" << std::endl;
        return 3;
    }
#pragma endregion

#pragma region TRAIN NET
    int stable_stop_counter = 0;
    int stable_stop_threshold = 10;
    float avrg = 0;
    float devi;
    int devi_stop_counter = 0;
    int devi_stop_threshold = 50;
    for (int epoch = 0; epoch < 100; ++epoch)
    {
        std::cout << "Epoch:\t" << epoch << std::endl;
        double train_err = 0;
        double test_err = 0;
        // Train
        train_err = rnn->trainSet(train_set);
        std::cout << std::to_string(train_err) << std::endl;
        // Test
        test_err = rnn->evalSet(test_set);
        std::cout << std::to_string(test_err) << std::endl;
#if LOGGING
        log << std::to_string(train_err) << " ";
        log << std::to_string(test_err) << " ";
#endif
        if (test_err < 0.0002)
        {
            if (stable_stop_counter++ > stable_stop_threshold)
            {
                std::cout << "Sufficiently stable. Stopping." << std::endl;
                break;
            }
        }
        else
        {
            stable_stop_counter = 0;
        }
        avrg = ((float)test_err + avrg * epoch) / (epoch + 1);
        devi = abs((float)test_err - avrg);
        if (devi < 0.1)
        {
            if (devi_stop_counter++ > devi_stop_threshold)
            {
                std::cout << "No deviation in 50 epochs. Stopping." << std::endl;
                break;
            }
            else
            {
                devi_stop_counter = 0;
            }
        }
        //train_set.shuffle();
    }
    rnn->print();
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




