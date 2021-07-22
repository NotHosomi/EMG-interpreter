#include "SignalParser.h"
#include "Serial.h"
#include <qmessagebox.h>
#include "Timer.h"
#include <string>
#include <sstream>
#include <fstream>

SignalParser::SignalParser(QWidget* window, int poll_time = 100, float sample_resolution = 10) :
	qWnd(window), POLL_TIME(poll_time), SAMPLE_TIME(1 / sample_resolution) // or use TimeConstants.h???
{}

SignalParser::~SignalParser()
{
	rnn = nullptr;
}

void SignalParser::setPort(std::string port)
{
	portname = port;
}

// accessors for the UI - maybe these need changing?
std::array<int, 3> SignalParser::getInput()
{
	std::lock_guard<std::mutex> lg(inLock);
	return inputs;
}

void SignalParser::setModel(BakedNetwork* model)
{
	rnn = model;
}

void SignalParser::runOn()
{
	disconnect_signal = false;
	Serial* port = new Serial(("\\\\.\\" + portname).c_str());

	if (!port->IsConnected())
	{
		QMessageBox::warning(qWnd, "COM port", ("Failed to connect to " + portname).c_str());
		return;
	}

	char buffer[256] = "";
	unsigned int buffer_size = 255;
	unsigned int length = 0;
	Timer tmr;

	tmr.mark();
	while (port->IsConnected() && !disconnect_signal)
	{
		length = port->ReadData(buffer, buffer_size);
		buffer[length] = 0;

		if (tmr.peek() > SAMPLE_TIME)
		{
			truncateBuffer(buffer, length);
			buildInputs(buffer);
			tmr.mark();
		}
		Sleep(POLL_TIME);
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
		//QMessageBox::warning(qWnd, "Serial", "Error: Bad buffer");
		return false;
	}
	int sample_length = end_index - start_index;
	char temp[256] = "";
	for (int i = 0; i < sample_length; ++i)
	{
		temp[i] = buffer[start_index + i];
	}
	for (int i = 0; i < sample_length; ++i)
	{
		buffer[i] = temp[i];
	}
	buffer[sample_length] = '\0';
	return true;
}

// send the fresh sample to the net
void SignalParser::buildInputs(char* buffer)
{
	// deconstruct the sample into values
	std::array<int, 3> sample;
	std::string value;
	std::stringstream samplestream(buffer);
	for (int i = 0; i < 3; ++i)
	{
		getline(samplestream, value, '-');
		if (value == "")
		{
			// bad sample, just use the previous values
			sample = inputs; // don't need mutex because the only write is on the same thread
			break;
		}
		sample[i] = std::stoi(value);
	}
	inLock.lock();
	inputs = sample;
	inLock.unlock();

	Eigen::VectorXd input_vec(3);
	for (int i = 0; i < 3; ++i)
	{
		input_vec[i] = sample[i] / 1023.0;
	}
	// TODO: fixup the following
	rnn->addInput(input_vec);
}
