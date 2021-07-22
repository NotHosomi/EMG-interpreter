#pragma once
#include "BakedGenericLayer.h"

class BakedDenseLayer : public BakedGenericLayer
{
public:
	BakedDenseLayer(int input_size, int output_size);

	VectorXd feedForward(VectorXd x_t) override;
};

