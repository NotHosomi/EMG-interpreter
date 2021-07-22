#include "BakedGenericLayer.h"
#include <iostream>

BakedGenericLayer::BakedGenericLayer(int input_size, int output_size) :
	INPUT_SIZE(input_size), OUTPUT_SIZE(output_size)
{}

void BakedGenericLayer::loadWeights(std::ifstream & file)
{
	std::for_each(w.data(), w.data() + w.size(), [&file](double& val)
		{
			file.read(reinterpret_cast<char*>(&val), sizeof(double));
		});
}

std::stringstream BakedGenericLayer::print()
{
	std::stringstream ss;
	IOFormat Fmt(4, 0, ", ", ";\n", "", "", "[", "]");
	ss << "w:\n" << w.format(Fmt) << "\n";
	return ss;
}

double BakedGenericLayer::sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double BakedGenericLayer::tangent(double x)
{
    return tanh(x);
}