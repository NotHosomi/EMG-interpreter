#include "BakedDenseLayer.h"
#include <QDebug>

BakedDenseLayer::BakedDenseLayer(int input_size, int output_size) :
	BakedGenericLayer(input_size, output_size)
{
	w = MatrixXd(OUTPUT_SIZE, INPUT_SIZE + 1);
}

VectorXd BakedDenseLayer::feedForward(VectorXd x_t)
{
	VectorXd in(INPUT_SIZE + 1);
	in << x_t, 1;
	VectorXd output = (w * in).unaryExpr(&BakedGenericLayer::sigmoid);
	return output;
}
