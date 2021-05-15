#pragma once
#include <random>
#include <vector>
#include <Eigen/Dense>
using namespace Eigen;

template <typename T>
struct Dataset
{
	std::vector<T> inputs;
	std::vector<T> labels;
};

namespace UnitTests
{
	Dataset<std::vector<VectorXd>> MealSequence(int dataset_size, int seq_length)
	{
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> r(0, 2);
        /*
        outputs: pie, burger, chicken
        inputs: sunny, rainy
        if sunny
          same as yesterday
        if rainy
          if pie
            burger
          if burger
            chicken
          if chicken
            pie
        */
        Dataset<std::vector<VectorXd>> data;
        for (int i = 0; i < seq_length; ++i)
        {
            data.inputs.emplace_back(std::vector<VectorXd>());
            data.labels.emplace_back(std::vector<VectorXd>());
            data.inputs.back().emplace_back(VectorXd(1));
            data.inputs.back().back().setZero();
            data.labels.back().emplace_back(VectorXd(3));
            data.labels.back().back().setZero();
            data.labels.back().back()[rand() % 3];
            for (int j = 0; j < 30; ++j)
            {
                double x = r(mt);
                data.inputs.back().emplace_back(VectorXd(1));
                data.inputs.back().back()[0] = x;

                VectorXd label = VectorXd(3);
                if (x)
                {
                    label = data.labels.back().back();
                }
                else
                {
                    if (data.labels.back().back()[0])
                        label << 0, 1, 0;
                    else if (data.labels.back().back()[1])
                        label << 0, 0, 1;
                    else
                        label << 1, 0, 0;
                }
                data.labels.back().push_back(label);
            }
        }
        return data;
	}

    enum GateType
    {
        AND, OR, XOR
    };
    Dataset<VectorXd> gate(int dataset_size, GateType gt)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> r(0, 1);

        Dataset<VectorXd> data;
        VectorXd input(2);
        VectorXd output(1);
        for (int i = 0; i < dataset_size; ++i)
        {
            input[0] = r(mt);
            input[1] = r(mt);
            switch (gt)
            {
            case GateType::AND:
                output[0] = (input[0] && input[1]);
                break;
            case GateType::OR:
                output[0] = (input[0] || input[1]);
                break;
            case GateType::XOR:
                output[0] = (input[0] != input[1]);
                break;
            }
            data.inputs.push_back(input);
            data.labels.push_back(output);
        }
        return data;
    }
}