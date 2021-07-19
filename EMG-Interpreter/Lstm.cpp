#include "Lstm.h"
#include <iostream>
#include <fstream>

#include "Common.h"
using namespace Common;

Lstm::Lstm(int input_size, int output_size, double alpha) :
	GenericLayer(input_size, output_size, alpha)
{
	// init W matrices for gates
	f = MatrixXd::Random(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1); // + 1 (bias)
	i = MatrixXd::Random(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1); // + 1 (bias)
	c = MatrixXd::Random(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1); // + 1 (bias)
	o = MatrixXd::Random(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1); // + 1 (bias)

	// init Adam matrices
	rms_prop_f.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	rms_prop_i.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	rms_prop_c.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	rms_prop_o.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	momentum_f.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	momentum_i.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	momentum_c.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	momentum_o.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);

	clearCaches();
}

VectorXd Lstm::feedForward(VectorXd x_t)
{
	VectorXd in(INPUT_SIZE + OUTPUT_SIZE + 1); // + 1; (bias)
	VectorXd cs(OUTPUT_SIZE);
	VectorXd hs(OUTPUT_SIZE);

	// concatenate HS<t-1> and X<t>
	in << hs_history.back(), x_t, 1; // , 1; (bias)

	VectorXd fz = f * in;
	VectorXd forget = (fz).unaryExpr(&sigmoid);
	cs = forget.cwiseProduct(cs_history.back());

	// Create and apply the new Cell State candidate
	VectorXd iz = i * in;
	VectorXd ignore = (iz).unaryExpr(&sigmoid);
	VectorXd cz = c * in;
	VectorXd candidate = (cz).unaryExpr(&tangent);
	cs += ignore.cwiseProduct(candidate);

	// Create the new output
	VectorXd oz = o * in;
	VectorXd output = (oz).unaryExpr(&sigmoid);
	hs = output.cwiseProduct(cs.unaryExpr(&tangent));



	// Cycle memory
	// Cell IO
	x_history.emplace_back(x_t);
	cs_history.emplace_back(cs);
	hs_history.emplace_back(hs);
	// Gates
	f_history.emplace_back(forget);
	i_history.emplace_back(ignore);
	c_history.emplace_back(candidate);
	o_history.emplace_back(output);
	// Caches
	fz_history.emplace_back(fz);
	iz_history.emplace_back(iz);
	cz_history.emplace_back(cz);
	oz_history.emplace_back(oz);
	//std::cout << hs << std::endl;


	// DEBUGGING normalise to 1-0 output, for debugging with Cross entropy
	//return (hs.array() + 1) / 2;

	return hs;
}

VectorXd Lstm::backProp(VectorXd gradient, unsigned int t)
{
#ifdef PRINT_BP
	std::cout << "\nIn:\n" << x_history[t]
		<< "\nF\n" << f
		<< "\nI\n" << i
		<< "\nC\n" << c
		<< "\nO\n" << o
		<< "\ndelta Y\n" << gradient << std::endl;
#endif

	gradient += dhs;

	//VectorXd cs = cs_history[t];												// Comment notation note: (c = cs, g = c, z = activation pre act-func)

	// Find derivatives for the gates' outputs
	VectorXd de_do = gradient.cwiseProduct(cs_history[t].unaryExpr(&tangent));										// Error * tanh(ct) 
	VectorXd de_dcst = gradient.cwiseProduct(o_history[t].cwiseProduct(cs_history[t].unaryExpr(&dtangent))) + dcs;	// Error * o * dtan(ct)
	VectorXd de_df = de_dcst.cwiseProduct(cs_history[t - 1]);														// Error * o * dtan(ct) * ct-1
	VectorXd de_di = de_dcst.cwiseProduct(c_history[t]);														// Error * o * dtan(ct) * g 
	VectorXd de_dc = de_dcst.cwiseProduct(i_history[t]);														// Error * o * dtan(ct) * i 
	
	// concat the inputs at timestep T
	VectorXd in(INPUT_SIZE + OUTPUT_SIZE + 1); // + 1 (bias)
	in << hs_history[t-1], x_history[t], 1; //, 1 (bias)

	// +--- element-wise gradient ---+
	// de_dz = dsig(z) * error
	// de_dw = input * dsig(z) * error
	// de_dx = SUM_j( weight_j * dsig(z) * error )

	// Find derivatives for the gates' weights
	VectorXd de_dzo = oz_history[t].unaryExpr(&dsigmoid).cwiseProduct(de_do);
	MatrixXd de_dwo = de_dzo * in.transpose();

	VectorXd de_dzf = fz_history[t].unaryExpr(&dsigmoid).cwiseProduct(de_df);
	MatrixXd de_dwf = de_dzf * in.transpose();

	VectorXd de_dzi = iz_history[t].unaryExpr(&dsigmoid).cwiseProduct(de_di);
	MatrixXd de_dwi = de_dzi * in.transpose();

	VectorXd de_dzc = cz_history[t].unaryExpr(&dtangent).cwiseProduct(de_dc);
	MatrixXd de_dwc = de_dzc * in.transpose();




#ifdef PRINT_BP
	std::cout << "CS\n" << cs_history[t]
		<< "\nHS\n" << hs_history[t]
		<< "\ndelta CS\n" << dcs
		<< "\ndelta HS\n" << dhs
		<< "\ndelta F\n" << de_dwf
		<< "\ndelta I\n" << de_dwi
		<< "\ndelta C\n" << de_dwc
		<< "\ndelta O\n" << de_dwo << std::endl;
#endif


	// build net updates from the gradients
	tfu += de_dwf;
	tiu += de_dwi;
	tcu += de_dwc;
	tou += de_dwo;
	
	// gradients for previous cell
	dcs = de_dcst.cwiseProduct(f_history[t]);
	dhs = o.block(0, 0, OUTPUT_SIZE, OUTPUT_SIZE).transpose() * de_dzo + // https://eigen.tuxfamily.org/dox/group__TutorialMatrixArithmetic.html#title5
		f.block(0, 0, OUTPUT_SIZE, OUTPUT_SIZE).transpose() * de_dzf +
		i.block(0, 0, OUTPUT_SIZE, OUTPUT_SIZE).transpose() * de_dzi +
		c.block(0, 0, OUTPUT_SIZE, OUTPUT_SIZE).transpose() * de_dzc;
	// return input gradients
	return o.block(0, OUTPUT_SIZE, OUTPUT_SIZE, INPUT_SIZE).transpose() * de_dzo +
		f.block(0, OUTPUT_SIZE, OUTPUT_SIZE, INPUT_SIZE).transpose() * de_dzf +
		i.block(0, OUTPUT_SIZE, OUTPUT_SIZE, INPUT_SIZE).transpose() * de_dzi +
		c.block(0, OUTPUT_SIZE, OUTPUT_SIZE, INPUT_SIZE).transpose() * de_dzc;
}

void Lstm::applyUpdates()
{
#ifdef PRINT_UPDATES
	std::cout << "-----------------------------"
		<< "\nF\n" << f
		<< "\nI\n" << i
		<< "\nC\n" << c
		<< "\nO\n" << o
		<< "\n\nFu\n" << tfu
		<< "\nIu\n" << tiu
		<< "\nCu\n" << tcu
		<< "\nOu\n" << tou << std::endl;
	//evalUpdates(f, tfu, 'F');
	//evalUpdates(i, tiu, 'I');
	//evalUpdates(c, tcu, 'C');
	//evalUpdates(o, tiu, 'O');
#endif
	++adam_t;
	// Momentum (V)
	// Vdw = Beta1 * Vdw + (1 - Beta1) * dw
	momentum_f = 0.9 * momentum_f + 0.01 * tfu;
	momentum_i = 0.9 * momentum_i + 0.01 * tfu;
	momentum_c = 0.9 * momentum_c + 0.01 * tfu;
	momentum_o = 0.9 * momentum_o + 0.01 * tfu;
	// RMS Prop (S)
	// Sdw = Beta2 * Sdw + (1-Beta2) * dw^2
	rms_prop_f = 0.999 * rms_prop_f + 0.001 * tfu.cwiseProduct(tfu);
	rms_prop_i = 0.999 * rms_prop_i + 0.001 * tiu.cwiseProduct(tiu);
	rms_prop_c = 0.999 * rms_prop_c + 0.001 * tcu.cwiseProduct(tcu);
	rms_prop_o = 0.999 * rms_prop_o + 0.001 * tou.cwiseProduct(tou);

	// Bias correction
	double beta1_denom = (1 - pow(0.9, adam_t));
	double beta2_denom = (1 - pow(0.999, adam_t));
	MatrixXd Vfc = momentum_f / beta1_denom;
	MatrixXd Vic = momentum_i / beta1_denom;
	MatrixXd Vcc = momentum_c / beta1_denom;
	MatrixXd Voc = momentum_o / beta1_denom;
	MatrixXd Sfc = rms_prop_f / beta2_denom;
	MatrixXd Sic = rms_prop_i / beta2_denom;
	MatrixXd Scc = rms_prop_c / beta2_denom;
	MatrixXd Soc = rms_prop_o / beta2_denom;

#ifdef PRINT_UPDATES
	std::cout
		<< "\nB1_\n" << beta1_denom
		<< "\nB2_\n" << beta2_denom
		<< "\nAdam T\n" << adam_t
		<< "\nVfc\n" << Vfc
		<< "\nVic\n" << Vic
		<< "\nVcc\n" << Vcc
		<< "\nVoc\n" << Voc
		<< "\nSfc\n" << Sfc
		<< "\nSic\n" << Sic
		<< "\nScc\n" << Scc
		<< "\nSoc\n" << Soc
		<< "-----------------------------"
		<< std::endl;
#endif
	// apply update sums
	// w = w - alpha * Vdwc / (root(Sdwc) + eps)
	f -= (alpha * Vfc.array() / (sqrt(Sfc.array()) + 1e-8)).matrix();
	i -= (alpha * Vic.array() / (sqrt(Sic.array()) + 1e-8)).matrix();
	c -= (alpha * Vcc.array() / (sqrt(Scc.array()) + 1e-8)).matrix();
	o -= (alpha * Voc.array() / (sqrt(Soc.array()) + 1e-8)).matrix();

	// clear update buffers
	clearCaches();
}

void Lstm::print()
{
	IOFormat Fmt(4, 0, ", ", ";\n", "", "", "[", "]");
	std::cout << "f:\n" << f.format(Fmt) << std::endl;
	std::cout << "i:\n" << i.format(Fmt) << std::endl;
	std::cout << "c:\n" << c.format(Fmt) << std::endl;
	std::cout << "o:\n" << o.format(Fmt) << std::endl;
}

void Lstm::resize(size_t new_depth)
{
	clearCaches();
	new_depth++;
	f_history.reserve(new_depth);
	i_history.reserve(new_depth);
	c_history.reserve(new_depth);
	o_history.reserve(new_depth);
	fz_history.reserve(new_depth);
	iz_history.reserve(new_depth);
	cz_history.reserve(new_depth);
	oz_history.reserve(new_depth);
	x_history.reserve(new_depth);
	cs_history.reserve(new_depth);
	hs_history.reserve(new_depth);
}

void Lstm::loadWeights(std::ifstream& file)
{
	// sanity check matrix sizes
	f = MatrixXd(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	i = MatrixXd(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	c = MatrixXd(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	o = MatrixXd(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);

	std::for_each(f.data(), f.data() + f.size(), [&file](double& val)
		{
			file.read(reinterpret_cast<char*>(&val), sizeof(double));
		});
	//std::cout << "\nMatrix (" << i.cols() << "," << i.rows() << "), pos: " << file.tellg() << std::endl;
	std::for_each(i.data(), i.data() + i.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double));
			//std::cout << val << "\t";
		});
	//std::cout << "\nMatrix (" << c.cols() << "," << c.rows() << "), pos: " << file.tellg() << std::endl;
	std::for_each(c.data(), c.data() + c.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double));
			//std::cout << val << "\t";
		});
	//std::cout << "\nMatrix (" << o.cols() << "," << o.rows() << "), pos: " << file.tellg() << std::endl;
	std::for_each(o.data(), o.data() + o.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double));
			//std::cout << val << "\t";
		});
	//std::cout << "Layer end pos: " << file.tellg() << std::endl;
}

void Lstm::loadGates(MatrixXd forget, MatrixXd ignore, MatrixXd candidate, MatrixXd output)
{
	f = forget;
	i = ignore;
	c = candidate;
	o = output;
}

void Lstm::save(std::ofstream& file)
{
	char type = 'L';
	file.write(reinterpret_cast<char*>(&type), sizeof(char));
	file.write(reinterpret_cast<char*>(&OUTPUT_SIZE), sizeof(int));

	std::for_each(f.data(), f.data() + f.size(), [&file](double val)
		{ file.write(reinterpret_cast<char*>(&val), sizeof(double)); });
	std::for_each(i.data(), i.data() + i.size(), [&file](double val)
		{ file.write(reinterpret_cast<char*>(&val), sizeof(double)); });
	std::for_each(c.data(), c.data() + c.size(), [&file](double val)
		{ file.write(reinterpret_cast<char*>(&val), sizeof(double)); });
	std::for_each(o.data(), o.data() + o.size(), [&file](double val)
		{ file.write(reinterpret_cast<char*>(&val), sizeof(double)); });
}

// reset all LSTM memory, excluding weights
void Lstm::clearCaches()
{
	f_history.clear();
	i_history.clear();
	c_history.clear();
	o_history.clear();
	fz_history.clear();
	iz_history.clear();
	cz_history.clear();
	oz_history.clear();
	x_history.clear();
	cs_history.clear();
	hs_history.clear();

	// add a 0th tick (everything set to 0)
	VectorXd empt(INPUT_SIZE);
	empt.setZero();
	x_history.emplace_back(empt);

	empt = VectorXd(OUTPUT_SIZE);
	empt.setZero();
	cs_history.emplace_back(empt);
	hs_history.emplace_back(empt);

	f_history.emplace_back(empt);
	i_history.emplace_back(empt);
	c_history.emplace_back(empt);
	o_history.emplace_back(empt);
	fz_history.emplace_back(empt);
	iz_history.emplace_back(empt);
	cz_history.emplace_back(empt);
	oz_history.emplace_back(empt);

	// clear update buffers
	dcs.setZero(OUTPUT_SIZE);
	dhs.setZero(OUTPUT_SIZE);
	tfu.setZero(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1);
	tiu.setZero(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1); 
	tcu.setZero(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1); 
	tou.setZero(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1); 
}

// Debug tool for LstmDebugSeqA
void Lstm::evalUpdates(MatrixXd gate, MatrixXd updates, char name)
{
	VectorXd target(gate.size());
	switch (name)
	{
	case 'F': target << 0, -3, 0, 0, 0, 3;
		break;
	case 'I': target << 0, 0, 3, 0, 0, 0;
		break;
	case 'C': target << 0, 0, 0, 3, 0, 0;
		break;
	case 'O': target << 0, 0, 0, 0, 3, 0;
		break;
	}

	std::cout << name << "\n";
	for (int i = 0; i < gate.size(); ++i)
	{
		std::cout << " " << (int)(gate.data()[i] < target[i] && updates.data()[i] < 0);
	}
	std::cout << std::endl;
}

void Lstm::readWeightBuffer(const std::vector<double>& theta, int& pos)
{
	std::for_each(f.data(), f.data() + f.size(), [theta, &pos](double& val)
		{
			val = theta[pos];
			++pos;
		});
	std::for_each(i.data(), i.data() + i.size(), [theta, &pos](double& val)
		{
			val = theta[pos];
			++pos;
		});
	std::for_each(c.data(), c.data() + c.size(), [theta, &pos](double& val)
		{
			val = theta[pos];
			++pos;
		});
	std::for_each(o.data(), o.data() + o.size(), [theta, &pos](double& val)
		{
			val = theta[pos];
			++pos;
		});
}

void Lstm::writeWeightBuffer(std::vector<double>& theta, int& pos)
{
	std::for_each(f.data(), f.data() + f.size(), [&theta, &pos](double val)
		{
			theta.push_back(val);
			++pos;
		});
	std::for_each(i.data(), i.data() + i.size(), [&theta, &pos](double val)
		{
			theta.push_back(val);
			++pos;
		});
	std::for_each(c.data(), c.data() + c.size(), [&theta, &pos](double val)
		{
			theta.push_back(val);
			++pos;
		});
	std::for_each(o.data(), o.data() + o.size(), [&theta, &pos](double val)
		{
			theta.push_back(val);
			++pos;
		});
}

void Lstm::writeUpdateBuffer(VectorXd& theta, int& pos)
{
	std::for_each(tfu.data(), tfu.data() + tfu.size(), [&theta, &pos](double val)
		{
			theta[pos] = val;
			++pos;
		});
	std::for_each(tiu.data(), tiu.data() + tiu.size(), [&theta, &pos](double val)
		{
			theta[pos] = val;
			++pos;
		});
	std::for_each(tcu.data(), tcu.data() + tcu.size(), [&theta, &pos](double val)
		{
			theta[pos] = val;
			++pos;
		});
	std::for_each(tou.data(), tou.data() + tou.size(), [&theta, &pos](double val)
		{
			theta[pos] = val;
			++pos;
		});
}