#include "SignalParser.h"
#include "Serial.h"
#include <qmessagebox.h>
#include "Timer.h"
#include <string>
#include <sstream>
#include <fstream>

SignalParser::SignalParser(QWidget* window, int poll_time = 100, float sample_resolution = 10) :
	qWnd(window), POLL_TIME(poll_time), SAMPLE_TIME(1 / sample_resolution)
{}

SignalParser::~SignalParser()
{
	if (lstm)
	{
		delete lstm;
		lstm = nullptr;
	}
}

std::array<int, 3> SignalParser::getInput()
{
	std::array<int, 3> copy;
	inLock.lock();
	copy = inputs;
	inLock.unlock();
	return copy;
}

std::array<int, 5> SignalParser::getOutput()
{
	std::array<int, 5> copy;
	outLock.lock();
	copy = outputs;
	outLock.unlock();
	return copy;
}

bool SignalParser::loadModel(std::string model_name)
{
	if (lstm)
	{
		delete lstm;
		lstm = nullptr;
	}

	std::ifstream model_file(model_name + ".dat", std::ios::binary || std::ios::in);
	if (!model_file)
	{
		QMessageBox::warning(qWnd, "Invalid model", ("Failed to load model \"" + model_name + "\"").c_str());
		return false;
	}
	lstm = new DumbLstm(model_file);
	return true;
}

void SignalParser::runOn(std::string portname)
{
	disconnect_signal = false;
	Serial* port = new Serial(("\\\\.\\" + portname).c_str());

	if (!port->IsConnected())
	{
		QMessageBox::warning(qWnd, "COM port", ("Failed to connect to " + portname).c_str());
		return;
	}

	while (!disconnect_signal)
	{
		char buffer[256] = "";
		unsigned int buffer_size = 255;
		unsigned int length = 0;
		Timer tmr;

		tmr.mark();
		while (port->IsConnected())
		{
			length = port->ReadData(buffer, buffer_size);
			buffer[length] = 0;

			if (tmr.peek() > SAMPLE_TIME)
			{
				truncateBuffer(buffer, length);
				buildInputs(buffer);
				calcOutputs();
				tmr.mark();
			}
		}
	}
	disconnect_signal = false;
	delete port;
}

void SignalParser::disconnect()
{
	disconnect_signal = true;
}

bool SignalParser::truncateBuffer(char* buffer, unsigned int content_length)
{
	// find the most recent sample
	int end_index = -1;
	int start_index = -1;
	for (int i = content_length - 1; i >= 0; --i)
	{
		if (buffer[i] == '!')
		{
			if (end_index == -1)
			{
				end_index = i;
			}
			else
			{
				start_index = i + 1;
				break;
			}
		}
	}
	if (end_index == -1 || start_index == -1)
	{
		QMessageBox::warning(qWnd, "Serial", "Error: Bad buffer");
		return false;
	}
	int output_length = end_index - start_index;
	char temp[256] = "";
	for (int i = 0; i < output_length; ++i)
	{
		temp[i] = buffer[start_index + i];
	}
	for (int i = 0; i < output_length; ++i)
	{
		buffer[i] = temp[i];
	}
	buffer[output_length] = '\0';
	return true;
}

void SignalParser::buildInputs(char* buffer)
{
	// deconstruct the sample into values
	std::array<int, 3> sample;
	std::string value;
	std::stringstream samplestream(buffer);
	for (int i = 0; i < 3; ++i)
	{
		getline(samplestream, value, '-');
		sample[i] = std::stoi(value);
	}
	inLock.lock();
	inputs = sample;
	inLock.unlock();
}

void SignalParser::calcOutputs()
{
	Eigen::VectorXd input_vec(3);
	for (int i = 0; i < 3; ++i)
	{
		input_vec[i] = inputs[i] / 1023.0;
	}
	Eigen::VectorXd raw_out = lstm->feedForward(input_vec);
	raw_out *= 100;

	outLock.lock();
	for (int i = 0; i < 3; ++i)
	{
		outputs[i] = raw_out[i];
	}
	outLock.unlock();
}
