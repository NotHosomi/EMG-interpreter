#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include "UnitTests.h"


#define RNN 1
#if RNN
#include "RecurrentNetwork.h"
#else
#include "DeepNetwork.h"
#endif
#define MODEL_IO 0
#define USE_PROCEDURAL_DATA 1
#define TRAIN 1
#define GRAD_CHECK 1
#define DEFAULT_HYPERPARAMS 1

// EMG constants
#define INPUT_SIZE 3
#define LABEL_SIZE 5
#define EMG_MAX 1023

int main()
{
    int epochs = 1;
    double alpha = 0.15;
    double input_gain = 2;
    bool shuffle_data = true;
#if !DEFAULT_HYPERPARAMS
    // TODO user input
#endif

    // Eigen's matrix random init uses rand()
    srand(static_cast<int>(time(0)));
#if RNN
    RecurrentNetwork* rnn;

    std::cout << "Building model" << std::endl;
#if MODEL_IO
    std::string net_name;
    std::ifstream net_file;
    std::cout << "Net File: ";
    std::cin >> net_name;
    if (net_name == "" || net_name == " ")
    {
        net_name = "Untitled";
    }
    net_file.open("nets/" + net_name + ".dat", std::ios::binary || std::ios::in);
    bool new_file = false;
    if (net_file.is_open() && net_name != "Untitled")
    {
        std::cout << "Loading 'nets/" << net_name << ".dat'" << std::endl;
        rnn = new RecurrentNetwork(net_file, net_name);
    }
    else
    {
        new_file = true;
        std::cout << "Creating new network \"" << net_name << ".dat\"" << std::endl;
        rnn = new RecurrentNetwork(INPUT_SIZE, alpha, net_name);
        rnn->addLayer('L', 16);
        rnn->addLayer('L', 16);
        rnn->addLayer('D', LABEL_SIZE);
    }
    net_file.close();
#else
#if !USE_PROCEDURAL_DATA
    rnn = new RecurrentNetwork(INPUT_SIZE, alpha, "Untitled");
    rnn->addLayer('L', 16);
    rnn->addLayer('L', 16);
    rnn->addLayer('D', LABEL_SIZE);
#else
    rnn = new RecurrentNetwork(2, alpha, "Untitled");
    rnn->addLayer('D', 1);
#endif
#endif // MODEL_IO

    // Load samples
#if !USE_PROCEDURAL_DATA
    std::ifstream data_file;
    data_file.open("data/bulk.emg");
    if (!data_file)
    {
        std::cout << "Failed to open file 'data/bulk.emg'" << std::endl;
        return 0;
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
        getline(linestream, sample, '!'); // Dump the first sample in the sequence due to false label bug
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
            input /= EMG_MAX / input_gain;
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

    if(shuffle_data)
        train_set.shuffle();
    Dataset<std::vector<VectorXd>> test_set = train_set.split(0.9);
    std::cout << "Train split:\t" << train_set.inputs.size()
        << "\tTest split:\t" << test_set.inputs.size() << std::endl;
#else // Procedural data gen
    //std::vector<Dataset<std::vector<VectorXd>>> batches;
    //for (int i = 0; i < 10; ++i)
    //    batches.push_back(UnitTests::MealSequence(30, 30));
    //Dataset<std::vector<VectorXd>> train_set = UnitTests::MealSequence(500, 10);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::MealSequence(100, 20);

    //Dataset<std::vector<VectorXd>> train_set = UnitTests::LstmDebugA(80, 10);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::LstmDebugA(20, 10);
    
    //Dataset<std::vector<VectorXd>> train_set = UnitTests::LstmMaximizer(80, 10, 4, 1, false);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::LstmMaximizer(20, 10, 4, 1, false);
     
    //Dataset<std::vector<VectorXd>> train_set = UnitTests::SeqSingle(800, UnitTests::GateType::AND);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::SeqSingle(200, UnitTests::GateType::AND);

    Dataset<std::vector<VectorXd>> train_set = UnitTests::SeqGatesIsolated(1, 1, UnitTests::GateType::AND);
    Dataset<std::vector<VectorXd>> test_set = UnitTests::SeqGatesIsolated(1, 1, UnitTests::GateType::AND);

    //Dataset<std::vector<VectorXd>> train_set = UnitTests::Toggle(120, 10);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::Toggle(20, 10);

    //Dataset<std::vector<VectorXd>> train_set = UnitTests::SeqGatesLinked(20, 10, UnitTests::GateType::XOR);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::SeqGatesLinked(5, 10, UnitTests::GateType::XOR);

    //Dataset<std::vector<VectorXd>> train_set = UnitTests::ElmanXORSet(500, 25);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::ElmanXORSet(50, 22);
    
    //Dataset<std::vector<VectorXd>> train_set = UnitTests::Noise(80, 10, 4, 1); 
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::Noise(20, 10, 4, 1);
#endif // USE_PROCEDURAL_DATA

    std::ofstream log("logs/loss.txt", std::ios::trunc);
    if (!log)
    {
        std::cout << "Failed to open file 'logs/loss.txt'" << std::endl;
        return 3;
    }

#if TRAIN
#pragma region TRAIN_NET
    int stable_stop_counter = 0;
    int stable_stop_threshold = 10;
    float avrg = 0;
    float devi;
    int devi_stop_counter = 0;
    int devi_stop_threshold = 50;
    for (int epoch = 0; epoch < epochs; ++epoch)
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
        log << std::to_string(train_err) << " ";
        log << std::to_string(test_err) << " ";
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
#if SHUFFLE_DATA
        train_set.shuffle();
#endif
    }
    rnn->save();
#pragma endregion
#endif // TRAIN

#if GRAD_CHECK
    //for (int seq = 0; seq < train_set.inputs.size(); ++seq)
    //{
    //    std::cout << "\nGrad check for seq " << seq << std::endl;
    //    rnn->gradCheck(train_set.inputs[seq], train_set.labels[seq]);
    //}
    rnn->gradCheck(train_set.inputs[0], train_set.labels[0]);
#endif // GRAD_CHECK


    log.close();
    delete rnn;

#if  MODEL_IO
    //std::string net_name = "Untitled";
    //std::ifstream net_file;
    net_name += "_CKP";
    net_file.open("nets/" + net_name + ".dat", std::ios::binary | std::ios::in);
    if (net_file)
    {
        std::cout << "Loading 'nets/" << net_name << ".dat'" << std::endl;
        rnn = new RecurrentNetwork(net_file, net_name);
        rnn->print();

        double test_err = rnn->evalSet(test_set);
        std::cout << "\nBest: " << std::to_string(test_err) << std::endl;
        delete rnn;
    }
#endif //  MODEL_IO

#elif // DNN
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
#endif // RNN
    return 0;
}




