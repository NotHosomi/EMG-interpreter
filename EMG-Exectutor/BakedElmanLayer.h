#pragma once
#include "BakedGenericLayer.h"
class BakedElmanLayer : public BakedGenericLayer
{
public:
	BakedElmanLayer(int input_size, int output_size);

	VectorXd feedForward(VectorXd x_t) override;
private:
	VectorXd context_activations;
};

