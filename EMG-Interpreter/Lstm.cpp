#include "Lstm.h"
#include <iostream>
#include <fstream>

Lstm::Lstm(int input_size, int output_size, double eta) :
	INPUT_SIZE(input_size), OUTPUT_SIZE(output_size), eta(eta)
{
	// init W matrices for gates
	f = MatrixXd::Random(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1); // + 1 (bias)
	i = MatrixXd::Random(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1); // + 1 (bias)
	c = MatrixXd::Random(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1); // + 1 (bias)
	o = MatrixXd::Random(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1); // + 1 (bias)

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

	Gf.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	Gi.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	Gc.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	Go.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
}

Lstm::Lstm(std::ifstream& file, double eta) :
	eta(eta)
{
	file.read(reinterpret_cast<char*>(&INPUT_SIZE), sizeof(unsigned int));
	file.read(reinterpret_cast<char*>(&OUTPUT_SIZE), sizeof(unsigned int));

	f = MatrixXd(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	i = MatrixXd(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	c = MatrixXd(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	o = MatrixXd(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);

	std::for_each(f.data(), f.data() + f.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double)); });
	// incase the above doesn't work:
	//std::for_each(f.data(), f.data() + f.size(), [&file](double& val)
	//	{
	//		double temp;
	//		file.read(reinterpret_cast<char*>(&temp), sizeof(double));
	//		val = temp;
	//	});
	std::for_each(i.data(), i.data() + i.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double)); });
	std::for_each(c.data(), c.data() + c.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double)); });
	std::for_each(o.data(), o.data() + o.size(), [&file](double& val)
		{ file.read(reinterpret_cast<char*>(&val), sizeof(double)); });

	// same as normal constructor
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
	c_history.emplace_back(empt);
	o_history.emplace_back(empt);
	Gf.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	Gi.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	Gc.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
	Go.setZero(OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE + 1);
}

#pragma region UTILS

double Lstm::sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double Lstm::dsigmoid(double x)
{
	return (1 / (1 + exp(-x))) * (1 - (1 / (1 + exp(-x))));
}

double Lstm::tangent(double x)
{
	return tanh(x);
}

double Lstm::dtangent(double x)
{
	//return (1 / cosh(x)) * (1 / cosh(x)); // is cosh unsafe?

	// simplified tanh derivative
	double th = tanh(x);
	return 1 - th * th;
}

// clamps between -6 and 6 to prevent vanishing
// redundant?
double Lstm::m_clamp(double x)
{
	if (x > 6)
		return 6;
	if (x < -6)
		return -6;
	return x;
}

double Lstm::reciprocal(double x)
{
	return 1 / x;
}

// reset all LSTM memory, excluding weights
void Lstm::clearCaches()
{
	f_history.clear();
	i_history.clear();
	c_history.clear();
	o_history.clear();
	x_history.clear();
	cs_history.clear();
	hs_history.clear();

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
}
#pragma endregion

VectorXd Lstm::feedForward(VectorXd x_t)
{
	VectorXd in(INPUT_SIZE + OUTPUT_SIZE + 1); // + 1; (bias)
	VectorXd cs(OUTPUT_SIZE);
	VectorXd hs(OUTPUT_SIZE);

	// concatenate HS<t-1> and X<t>
	in << hs_history.back(), x_t, 1; // , 1; (bias)

	VectorXd forget = (f * in).unaryExpr(&sigmoid);
	cs = forget.cwiseProduct(cs_history.back());

	// Create and apply the new Cell State candidate
	VectorXd ignore = (i * in).unaryExpr(&sigmoid);
	VectorXd candidate = (c * in).unaryExpr(&tangent);
	cs += ignore.cwiseProduct(candidate);

	// Create the new output
	VectorXd output = (o * in).unaryExpr(&sigmoid);
	hs = output.cwiseProduct(cs.unaryExpr(&tangent));

	// Cycle memory
	x_history.emplace_back(x_t);
	cs_history.emplace_back(cs);
	hs_history.emplace_back(hs);
	f_history.emplace_back(forget);
	i_history.emplace_back(ignore);
	c_history.emplace_back(candidate);
	o_history.emplace_back(output);

	return hs;
}

double Lstm::backProp(std::vector<VectorXd> labels)
{
	// prep persistent backprop data
	double net_error = 0.0;
	// intercell gradients
	VectorXd dcs;
	VectorXd dhs;
	dcs.setZero(OUTPUT_SIZE);
	dhs.setZero(OUTPUT_SIZE);
	// Backprop update sums
	MatrixXd tfu;
	MatrixXd tiu;
	MatrixXd tcu;
	MatrixXd tou;
	tfu.setZero(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1); // + 1 (bias)
	tiu.setZero(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1); // + 1 (bias)
	tcu.setZero(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1); // + 1 (bias)
	tou.setZero(OUTPUT_SIZE, OUTPUT_SIZE + INPUT_SIZE + 1); // + 1 (bias)

	// calculate the recurrence limit
	int r_limit = labels.size() < hs_history.size() - 1 ? hs_history.size() - 1 - labels.size() : 1; // REVIEW THIS
	// Begin backprop
	for (int t = hs_history.size() - 1; t >= r_limit; --t)
	{
		VectorXd targets = labels[t-1]; // cell caches are all offset +1, t-1 here is just to counteract that
		VectorXd error = 2 * (targets - hs_history[t]);
		net_error += 2 * error.sum();



		// TODO: add an output net here



		error += dhs;

		VectorXd cs = cs_history[t];													// Comment notation note: (c = cs, g = c, z = activation pre act-func)

		// Find derivatives for the gates' outputs
		VectorXd de_do = error.cwiseProduct(cs_history[t].unaryExpr(&tangent));										// Error * tanh(ct) 
		VectorXd de_dcst = error.cwiseProduct(o_history[t].cwiseProduct(cs_history[t].unaryExpr(&dtangent))) + dcs;	// Error * o * dtan(ct)
		VectorXd de_df;
		//if (t > 0)
			de_df = de_dcst.cwiseProduct(cs_history[t - 1]);														// Error * o * dtan(ct) * ct-1  
		//else
		//	de_df = de_dcst * 0;																					// Error * o * dtan(ct) * ct-1 (all 0 at T:0)
		VectorXd de_di = de_dcst.cwiseProduct(c_history[t]);														// Error * o * dtan(ct) * g 
		VectorXd de_dc = de_dcst.cwiseProduct(i_history[t]);														// Error * o * dtan(ct) * i 
		
		// concat the inputs at timestep T
		VectorXd in(INPUT_SIZE + OUTPUT_SIZE + 1); // + 1 (bias)
		//if (t > 0)
			in << hs_history[t-1], x_history[t], 1; //, 1 (bias)
		//else
		//{
		//	VectorXd empt(OUTPUT_SIZE);
		//	empt.setZero();
		//	in << empt, x_history[t];
		//}

		// +--- element-wise gradient ---+
		// de_db = dsig(z) * error
		// de_dw = input * dsig(z) * error
		// de_dx = SUM_j( weight_j * dsig(z) * error )

		// refer to the G4G article for specific formula

		// Find derivatives for the gates' weights
		VectorXd de_dbo = (o * in).unaryExpr(&dsigmoid).cwiseProduct(de_do);
		MatrixXd de_dwo = de_dbo * in.transpose();

		VectorXd de_dbf = (f * in).unaryExpr(&dsigmoid).cwiseProduct(de_df);
		MatrixXd de_dwf = de_dbf * in.transpose();

		VectorXd de_dbi = (i * in).unaryExpr(&dsigmoid).cwiseProduct(de_di);
		MatrixXd de_dwi = de_dbi * in.transpose();

		VectorXd de_dbc = (c * in).unaryExpr(&dtangent).cwiseProduct(de_dc);
		MatrixXd de_dwc = de_dbc * in.transpose();

		// build net updates from the gradients
		tfu += de_dwf;
		tiu += de_dwi;
		tcu += de_dwc;
		tou += de_dwo;
		
		// gradients for previous cell
		dcs = de_dcst.cwiseProduct(f_history[t]);
		dhs = o.block(0, 0, OUTPUT_SIZE, OUTPUT_SIZE).transpose() * de_dbo + // https://eigen.tuxfamily.org/dox/group__TutorialMatrixArithmetic.html#title5
			f.block(0, 0, OUTPUT_SIZE, OUTPUT_SIZE).transpose() * de_dbf +
			i.block(0, 0, OUTPUT_SIZE, OUTPUT_SIZE).transpose() * de_dbi +
			c.block(0, 0, OUTPUT_SIZE, OUTPUT_SIZE).transpose() * de_dbc;
	}

	// update the gradients
	// TODO fix the momentum formula
	// other method: eta * gradient + alpha * prev_gradient
	// this method: 0.9 * prev_grad + 0.1 * gradient^2
	Gf = 0.9 * Gf + 0.1 * tfu.cwiseProduct(tfu);
	Gi = 0.9 * Gi + 0.1 * tiu.cwiseProduct(tiu);
	Gc = 0.9 * Gc + 0.1 * tcu.cwiseProduct(tcu);
	Go = 0.9 * Go + 0.1 * tou.cwiseProduct(tou);

	// apply update sums
	f += (eta / sqrt(Gf.array() + 1e-8) * tfu.array()).matrix();
	i += (eta / sqrt(Gi.array() + 1e-8) * tiu.array()).matrix();
	c += (eta / sqrt(Gc.array() + 1e-8) * tcu.array()).matrix();
	o += (eta / sqrt(Go.array() + 1e-8) * tou.array()).matrix();

	// clear update buffers
	clearCaches();

	return net_error;
}

void Lstm::printGates()
{
	IOFormat Fmt(4, 0, ", ", ";\n", "", "", "[", "]");
	std::cout << "f:\n" << f.format(Fmt) << std::endl;
	std::cout << "i:\n" << i.format(Fmt) << std::endl;
	std::cout << "c:\n" << c.format(Fmt) << std::endl;
	std::cout << "o:\n" << o.format(Fmt) << std::endl;
}

void Lstm::resize(int new_depth)
{
	clearCaches();
	new_depth++;
	f_history.reserve(new_depth);
	i_history.reserve(new_depth);
	c_history.reserve(new_depth);
	o_history.reserve(new_depth);
	x_history.reserve(new_depth);
	cs_history.reserve(new_depth);
	hs_history.reserve(new_depth);
}

// TODO: test this
bool Lstm::saveCell()
{
	printGates();
	std::string filename;
	std::cout << "\nSave to: ";
	std::cin >> filename;
	std::ofstream file;
	file.open("" + filename + ".dat", std::ios::binary | std::ios::out | std::ios::trunc);
	if (!file)
	{
		std::cout << "Failed to open " << filename << "_gates.dat" << std::endl;
		return false;
	}

	// all matrix dimensions should be homogenous
	int x = INPUT_SIZE;
	file.write(reinterpret_cast<char*>(&x), sizeof(int));
	int y = OUTPUT_SIZE;
	file.write(reinterpret_cast<char*>(&y), sizeof(int));

	std::for_each(f.data(), f.data() + f.size(), [&file](double val)
		{ file.write(reinterpret_cast<char*>(&val), sizeof(double)); });
	std::for_each(i.data(), i.data() + i.size(), [&file](double val)
		{ file.write(reinterpret_cast<char*>(&val), sizeof(double)); });
	std::for_each(c.data(), c.data() + c.size(), [&file](double val)
		{ file.write(reinterpret_cast<char*>(&val), sizeof(double)); });
	std::for_each(o.data(), o.data() + o.size(), [&file](double val)
		{ file.write(reinterpret_cast<char*>(&val), sizeof(double)); });


	std::cout << "\nSaved." << std::endl;
	file.close();
	return true;
}

void Lstm::loadGates(MatrixXd forget, MatrixXd ignore, MatrixXd candidate, MatrixXd output)
{
	f = forget;
	i = ignore;
	c = candidate;
	o = output;
}