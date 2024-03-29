#pragma once
#include <random>
#include <vector>
#include <Eigen/Dense>
#include "Dataset.h"
using namespace Eigen;


namespace UnitTests
{
    enum GateType
    {
        AND, OR, XOR, NAND
    };

    Dataset<std::vector<VectorXd>> MealSequence(int dataset_size, int seq_length)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> r(0, 1);
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
        for (int i = 0; i < dataset_size; ++i)
        {
            data.inputs.emplace_back(std::vector<VectorXd>());
            data.labels.emplace_back(std::vector<VectorXd>());
            data.inputs.back().emplace_back(VectorXd(1));
            data.inputs.back().back().setOnes();
            data.labels.back().emplace_back(VectorXd(3));
            data.labels.back().back().setZero();
            data.labels.back().back()[rand() % 3] = 1;
            for (int j = 0; j < seq_length; ++j)
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

                //std::cout << data.inputs.back().back() << "\t"
                //    << data.labels.back().back().transpose() << std::endl;
            }
           // std::cout << "--------------------------------------------------" << std::endl;
        }
        return data;
    }

    Dataset<std::vector<VectorXd>> LstmDebugA(int dataset_size, int seq_length)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> r(0, 1);
        /*
        outputs: y
        inputs: f, i, c, o
        if f
          cs = 0
        if i & c
          cs = 1
        if o
          y = cs
        else
          y = 0
        */
        Dataset<std::vector<VectorXd>> data;
        for (int i = 0; i < dataset_size; ++i)
        {
            data.inputs.emplace_back(std::vector<VectorXd>());
            data.labels.emplace_back(std::vector<VectorXd>());
            data.inputs.back().emplace_back(VectorXd(4));
            data.inputs.back().back().setZero();
            data.labels.back().emplace_back(VectorXd(1));
            data.labels.back().back().setZero();
            double cs = 0;
            for (int j = 0; j < seq_length; ++j)
            {
                double f = r(mt);
                double i = r(mt);
                double c = r(mt);
                double o = r(mt);
                data.inputs.back().emplace_back(VectorXd(4));
                data.inputs.back().back()[0] = f;
                data.inputs.back().back()[1] = i;
                data.inputs.back().back()[2] = c;
                data.inputs.back().back()[3] = o;

                if (f)
                    cs = 0;
                if (i)
                    cs += c;

                VectorXd y = VectorXd(1);
                if (o)
                    y[0] = 1;
                else
                    y[0] = 0;
                data.labels.back().push_back(y);
            }
        }
        return data;
    }

    Dataset<std::vector<VectorXd>> SeqSingle(int dataset_size, GateType gt)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> r(0, 1);
        Dataset<std::vector<VectorXd>> data;
        for (int i = 0; i < dataset_size; ++i)
        {
            data.inputs.emplace_back(std::vector<VectorXd>());
            data.labels.emplace_back(std::vector<VectorXd>());
            data.inputs.back().emplace_back(VectorXd(2));
            data.inputs.back().back().setZero();
            data.labels.back().emplace_back(VectorXd(1));
            data.labels.back().back().setZero();

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
                case GateType::NAND:
                    output[0] = !(input[0] && input[1]);
                    break;
                }
                data.inputs.push_back(input);
                data.labels.push_back(output);
            }
        }
        return data;
    }

    Dataset<std::vector<VectorXd>> SeqGatesIsolated(int dataset_size, int seq_length, GateType gt)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> r(0, 1);
        Dataset<std::vector<VectorXd>> data;
        VectorXd input(2);
        VectorXd output(1);
        for (int i = 0; i < dataset_size; ++i)
        {
            data.inputs.emplace_back(std::vector<VectorXd>());
            data.labels.emplace_back(std::vector<VectorXd>());
            for (int j = 0; j < seq_length; ++j)
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
                case GateType::NAND:
                    output[0] = !(input[0] && input[1]);
                    break;
                }
                data.inputs.back().push_back(input);
                data.labels.back().push_back(output);
            }
        }
        return data;
    }

    Dataset<std::vector<VectorXd>> SeqGatesLinked(int dataset_size, int seq_length, GateType gt)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> r(0, 1);
        Dataset<std::vector<VectorXd>> data;
        VectorXd input(1);
        VectorXd output(1);
        for (int i = 0; i < dataset_size; ++i)
        {
            data.inputs.emplace_back(std::vector<VectorXd>());
            data.labels.emplace_back(std::vector<VectorXd>());
            data.inputs.back().emplace_back(VectorXd(1));
            data.inputs.back().back().setZero();
            data.labels.back().emplace_back(VectorXd(1));
            data.labels.back().back().setZero();
            for (int j = 0; j < seq_length; ++j)
            {
                input[0] = r(mt);
                switch (gt)
                {
                case GateType::AND:
                    output[0] = (input[0] && data.inputs.back().back()[0]);
                    break;
                case GateType::OR:
                    output[0] = (input[0] || data.inputs.back().back()[0]);
                    break;
                case GateType::XOR:
                    output[0] = (input[0] != data.inputs.back().back()[0]);
                    break;
                case GateType::NAND:
                    output[0] = !(input[0] && data.inputs.back().back()[0]);
                    break;
                }
                data.inputs.back().push_back(input);
                data.labels.back().push_back(output);
            }
            data.inputs.back().erase(data.inputs.back().begin());
            data.labels.back().erase(data.labels.back().begin());
        }
        return data;
    }

    Dataset<std::vector<VectorXd>> Toggle(int dataset_size, int seq_length)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> r(0, 1);
        Dataset<std::vector<VectorXd>> data;
        VectorXd input(1);
        VectorXd output(1);
        for (int i = 0; i < dataset_size; ++i)
        {
            data.inputs.emplace_back(std::vector<VectorXd>());
            data.labels.emplace_back(std::vector<VectorXd>());
            data.inputs.back().emplace_back(VectorXd(1));
            data.inputs.back().back().setZero();
            data.labels.back().emplace_back(VectorXd(1));
            data.labels.back().back().setZero();
            bool state = false;
            for (int j = 0; j < seq_length; ++j)
            {
                input[0] = r(mt);
                if (input[0])
                    state = !state;
                output[0] = (double)state;
                data.inputs.back().push_back(input);
                data.labels.back().push_back(output);
            }
        }
        return data;
    }

    Dataset<std::vector<VectorXd>> LstmMaximizer(int dataset_size, int seq_length, int input_size, int label_size, bool target)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> r(0, 1);
        Dataset<std::vector<VectorXd>> data;
        for (int i = 0; i < dataset_size; ++i)
        {
            data.inputs.emplace_back(std::vector<VectorXd>());
            data.labels.emplace_back(std::vector<VectorXd>());
            double cs = 0;
            for (int j = 0; j < seq_length; ++j)
            {
                VectorXd x = VectorXd(input_size);
                for (int i = 0; i < input_size; ++i)
                {
                    x[i] = r(mt);
                }
                data.inputs.back().push_back(x);

                VectorXd y = VectorXd(label_size);
                for (int i = 0; i < label_size; ++i)
                {
                    y[i] = (double)target;
                }
                data.labels.back().push_back(y);
            }
        }
        return data;
    }

    Dataset<std::vector<VectorXd>> Noise(int dataset_size, int seq_length, int input_size, int label_size)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> r(0, 1);
        /*
        outputs: y
        inputs: f, i, c, o
        if f
          cs = 0
        if i & c
          cs = 1
        if o
          y = cs
        else
          y = 0

        */
        Dataset<std::vector<VectorXd>> data;
        for (int i = 0; i < dataset_size; ++i)
        {
            data.inputs.emplace_back(std::vector<VectorXd>());
            data.labels.emplace_back(std::vector<VectorXd>());
            double cs = 0;
            for (int j = 0; j < seq_length; ++j)
            {
                VectorXd x = VectorXd(input_size);
                for (int i = 0; i < input_size; ++i)
                {
                    x[i] = r(mt);
                }
                data.inputs.back().push_back(x);

                VectorXd y = VectorXd(label_size);
                for (int i = 0; i < label_size; ++i)
                {
                    y[i] = r(mt);
                }
                data.labels.back().push_back(y);
            }
        }
        return data;
    }

    // From Elman's original paper. Seq length should be a multiple of three
    Dataset<std::vector<VectorXd>> ElmanXORSet(int dataset_size, int seq_length)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> r(0, 1);
        Dataset<std::vector<VectorXd>> data;
        VectorXd x(1);
        double a = 0, b = 0, y = 0;
        for (int i = 0; i < dataset_size; ++i)
        {
            data.inputs.emplace_back(std::vector<VectorXd>());
            data.labels.emplace_back(std::vector<VectorXd>());
            data.inputs.back().emplace_back(VectorXd(1));
            data.inputs.back().back().setZero();
            data.labels.back().emplace_back(VectorXd(1));
            data.labels.back().back().setZero();
            for (int j = 0; j < seq_length/3; ++j)
            {
                a = r(mt);
                b = r(mt);
                x << a;
                data.inputs.back().push_back(x);
                x << b;
                data.inputs.back().push_back(x);
                x << (double)(a != b);
                data.inputs.back().push_back(x);
                std::cout << a << " " << b << " " << x[0] << std::endl;
            }
            data.labels.back() = data.inputs.back();
            data.labels.back().erase(data.labels.back().begin());
            data.inputs.back().pop_back();
        }
        
        return data;
    }

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
            case GateType::NAND:
                output[0] = !(input[0] && input[1]);
                break;
            }
            data.inputs.push_back(input);
            data.labels.push_back(output);
        }
        return data;
    }
}