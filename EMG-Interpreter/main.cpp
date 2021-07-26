#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include "UnitTests.h"

// Development & debug settings
#define RNN 1
#if RNN
#include "RecurrentNetwork.h"
#else
#include "DeepNetwork.h"
#endif
#define MODEL_IO 1
#define USE_PROCEDURAL_DATA 0
#define TRAIN 1
#define INTRAEPOCH_LOGGING 0
#define GRAD_CHECK 0
#define DEFAULT_TOPOLOGY 0
#define DEFAULT_HYPERPARAMS 0

// EMG constants
#define INPUT_SIZE 3
#define LABEL_SIZE 5
#define EMG_MAX 1023

// default hyperparams
#define EPOCHS 50
#define ALPHA 0.15;
#define GAIN 2;


std::vector<std::tuple<char, int>> buildTopology();

int main()
{
    // Init hyperparameters
    unsigned int epochs = EPOCHS;
    double alpha = ALPHA;
    double input_gain = GAIN;
    bool shuffle_data = true;
    bool training_mode = true;
    int window_size = 1;
#if !DEFAULT_HYPERPARAMS
    std::cout << "\n-- HYPERPARAMERS --\n(enter 0 to use defaults)" << std::endl;

    std::string txt;
    std::cout << "Mode (Train/Eval): ";
    while (1)
    {
        std::cin >> txt;
        txt[0] = std::tolower(txt[0]);
        if (txt == "0" || txt == "train" || txt == "training")
        {
            training_mode = true;
            break;
        }
        if (txt == "1" || txt == "eval" || txt == "evaluate" || txt == "evaluating")
        {
            training_mode = false;
            break;
        }
        std::cout << "Invalid input - [Train/Eval]" << std::endl;
    }

    if (training_mode)
    {
        epochs = 0;
        std::cout << "Epochs: ";
        while (!(std::cin >> epochs))
        {
            std::cout << "Invalid input, please enter an interger" << std::endl;
            std::cin.clear();
            while (std::cin.get() != '\n'); // flush cin buffer
        }
        if (epochs <= 0)
        {
            epochs = EPOCHS;
            std::cout << "Using default number of epochs (" << epochs << ")" << std::endl;
        }

        alpha = 0;
        std::cout << "Alpha (Learning Rate): ";
        while (!(std::cin >> alpha))
        {
            std::cout << "Invalid input, please enter an decimal" << std::endl;
            std::cin.clear();
            while (std::cin.get() != '\n'); // flush cin buffer
        }
        if (alpha <= 0)
        {
            alpha = ALPHA;
            std::cout << "Using default alpha (" << std::to_string(alpha) << ")" << std::endl;
        }
    }

    input_gain = 0;
    std::cout << "Input gain: ";
    while (!(std::cin >> input_gain))
    {
        std::cout << "Invalid input, please enter an decimal" << std::endl;
        std::cin.clear();
        while (std::cin.get() != '\n'); // flush cin buffer
    }
    if (input_gain <= 0)
    {
        input_gain = GAIN;
        std::cout << "Using default input gain (" << std::to_string(input_gain) << ")" << std::endl;
    }
#endif

    // Eigen's matrix random init uses rand()
    srand(static_cast<int>(time(0)));
#if RNN
    RecurrentNetwork* rnn;

    std::cout << "\n-- BUILD NET --" << std::endl;
#if MODEL_IO
    std::string net_name;
    std::ifstream net_file;
    std::cout << "Net File: ";
    std::cin >> net_name;
    if (net_name == "untitled" || net_name == "new" || net_name == "New" || net_name == "0")
    {
        net_name = "Untitled";
    }
    net_file.open("nets/" + net_name + ".dat", std::ios::binary | std::ios::in);
    bool new_file = false;
    if (net_file.is_open() && net_name != "Untitled")
    {
        // Load an existing model from file
        std::cout << "Loading 'nets/" << net_name << ".dat'" << std::endl;
        rnn = new RecurrentNetwork(net_file, net_name);
    }
    else
    {
        // Create a new model
        new_file = true;
        std::cout << "Creating new network \"" << net_name << ".dat\"" << std::endl;
        rnn = new RecurrentNetwork(INPUT_SIZE, alpha, net_name);
        for (auto L : buildTopology())
        {
            rnn->addLayer(std::get<char>(L), std::get<int>(L));
        }
    }
    net_file.close();
#else
    std::cout << "\nBuilding model..." << std::endl;
#if !USE_PROCEDURAL_DATA
    rnn = new RecurrentNetwork(INPUT_SIZE, alpha, "Untitled");
    rnn->addLayer('L', 16);
    rnn->addLayer('L', 16);
    rnn->addLayer('D', LABEL_SIZE);
#else
    rnn = new RecurrentNetwork(2, alpha, "Untitled");
    rnn->addLayer('E', 2);
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

    std::cout << "\nMounting signal data..." << std::endl;

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
    // A whole lot of generated datasets (See UnitTests.h)

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

    Dataset<std::vector<VectorXd>> train_set = UnitTests::SeqGatesIsolated(1, 3, UnitTests::GateType::AND);
    Dataset<std::vector<VectorXd>> test_set = UnitTests::SeqGatesIsolated(1, 3, UnitTests::GateType::AND);

    //Dataset<std::vector<VectorXd>> train_set = UnitTests::Toggle(120, 10);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::Toggle(20, 10);

    //Dataset<std::vector<VectorXd>> train_set = UnitTests::SeqGatesLinked(20, 10, UnitTests::GateType::XOR);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::SeqGatesLinked(5, 10, UnitTests::GateType::XOR);

    //Dataset<std::vector<VectorXd>> train_set = UnitTests::ElmanXORSet(500, 25);
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::ElmanXORSet(50, 22);
    
    //Dataset<std::vector<VectorXd>> train_set = UnitTests::Noise(80, 10, 4, 1); 
    //Dataset<std::vector<VectorXd>> test_set = UnitTests::Noise(20, 10, 4, 1);
#endif // USE_PROCEDURAL_DATA

    // eval only mode
    if (!training_mode)
    {
        rnn->useCheckpoints(false);
        double test_err = rnn->evalSet(test_set);
        std::cout << "\nLoss: " << std::to_string(test_err) << std::endl;
        delete rnn;
        return 0;
    }

    // open the log file to record the training curve
    std::ofstream log("logs/loss.txt", std::ios::trunc);
    if (!log)
    {
        std::cout << "Failed to open file 'logs/loss.txt'\nDoes the directory exist?" << std::endl;
        return 3;
    }

#if TRAIN
    std::cout << "\n-- TRAIN NET --" << std::endl;
#pragma region TRAIN_NET
    // prep early-exit vars
    int stable_stop_counter = 0;
    int stable_stop_threshold = 10;
    float avrg = 0;
    // The training loop
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
        // log
        log << std::to_string(train_err) << " ";
        log << std::to_string(test_err) << " ";

        // early exit check
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
#if SHUFFLE_DATA
        train_set.shuffle();
#endif
    }
    rnn->save();
#pragma endregion
#elif INTRAEPOCH_LOGGING
    std::cout << "\n-- TRAIN NET --" << std::endl;
    // The training loop
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (int seq = 0; seq < train_set.inputs.size(); ++seq)
        {
            double train_err = 0;
            double test_err = 0;
            // Train
            train_err = rnn->trainSeq(train_set.inputs[seq], train_set.labels[seq]);
            std::cout << std::to_string(train_err) << std::endl;
            // log
            log << std::to_string(train_err) << " 0 ";
        }
    }
#endif // TRAIN

    // For debugging
#if GRAD_CHECK
    std::cout << "Running grad checks" << std::endl;
    //for (int seq = 0; seq < train_set.inputs.size(); ++seq)
    //{
    //    std::cout << "\nGrad check for seq " << seq << std::endl;
    //    rnn->gradCheck(train_set.inputs[seq], train_set.labels[seq]);
    //}
    rnn->gradCheck(train_set.inputs[0], train_set.labels[0]);
    rnn->gradCheckAtT(train_set.inputs[0], train_set.labels[0], 0);
    rnn->gradCheckAtT(train_set.inputs[0], train_set.labels[0], 1);
    rnn->gradCheckAtT(train_set.inputs[0], train_set.labels[0], 2);
    std::cout << "Finished grad checks" << std::endl;
#endif // GRAD_CHECK

    log.close();
    delete rnn;

    // load the checkpoint to create output data of the best model
#if  MODEL_IO
    std::cout << "\n-- BEST ITERATION --" << std::endl;
    net_name += "_CKP";
    net_file.open("nets/" + net_name + ".dat", std::ios::binary | std::ios::in);
    if (net_file)
    {
        std::cout << "Loading 'nets/" << net_name << ".dat'" << std::endl;
        rnn = new RecurrentNetwork(net_file, net_name);
        rnn->useCheckpoints(false);

        double test_err = rnn->evalSet(test_set);
        std::cout << "\nLoss: " << std::to_string(test_err) << std::endl;
        delete rnn;
    }
#endif //  MODEL_IO

    // an implementation of a regular DNN
    // The RNN class is basically just a batch DNN anyway tho
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

std::vector<std::tuple<char, int>> buildTopology()
{
    std::vector<std::tuple<char, int>> topology_desc;
#if DEFAULT_TOPOLOGY
    topology_desc.emplace_back('L', 16);
    topology_desc.emplace_back('L', 16);
    topology_desc.emplace_back('D', 5);
    return topology_desc;
#endif
    std::cout << "-- TOPOLOGY --\n(Output layer must be of size 5. Enter 'done' to proceed. Enter 'pop' to remove last layer)" << std::endl;

    while (1)
    {
        std::cout << "Add Layer: ";
        std::string txt;
        std::cin >> txt;
        txt[0] = std::toupper(txt[0]);
        // exit check
        if (txt == "Done")
        {
            if (std::get<int>(topology_desc.back()) != LABEL_SIZE)
            {
                std::cout << "Final layer must contain " << LABEL_SIZE << " neurons" << std::endl;
                continue;
                //std::cout << "Final layer must contain " << LABEL_SIZE << " neurons. Appending D" << LABEL_SIZE << std::endl;
                //topology_desc.emplace_back('D', LABEL_SIZE);
                }
            else if (std::get<char>(topology_desc.back()) == 'L')
            {
                std::cout << "Final layer must use sigmoid activation (E, D)" << std::endl;
                continue;
                //std::cout << "Final layer must use sigmoid activation (E, D). Appending D" << LABEL_SIZE << std::endl;
                //topology_desc.emplace_back('D', LABEL_SIZE);
            }
            break;
            }
        if (txt == "Pop")
        {
            std::cout << "Removed " << std::get<char>(topology_desc.back()) << std::get<int>(topology_desc.back()) << std::endl;
            topology_desc.pop_back();
            continue;
        }
        // contents validation
        if (txt.size() < 2)
        {
            std::cout << "Invalid input - expected: (char) layer type, (int) layer size\n i.e. L16, E3, D5" << std::endl;
            continue;
        }
        // layer-type validation
        if (txt[0] != 'D' && txt[0] != 'E' && txt[0] != 'L')
        {
            std::cout << "Invalid layer type. Must be (L)stm, (E)lman or (D)ense" << std::endl;
            continue;
        }
        // determine type
        char type = txt[0];
        txt.erase(txt.begin());
        // determine size
        if (std::find_if(txt.begin(), txt.end(), [](char c) { return !std::isdigit(c); }) != txt.end())
        {
            std::cout << "Invalid input - expected: (char) layer type, (int) layer size\n i.e. L16, E3, D5" << std::endl;
            continue;
        }
        int size = std::stoi(txt);

        topology_desc.emplace_back(type, size);
    }
    std::cout << "Proceeding with ";
    for (auto L : topology_desc)
    {
        std::cout << std::get<char>(L) << std::get<int>(L) << " ";
    }
    std::cout << std::endl;

}