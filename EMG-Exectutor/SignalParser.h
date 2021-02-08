#pragma once
#include <array>
#include <string>
#include <mutex>
#include <qwidget.h>
#include "DumbLstm.h"

class SignalParser
{
public:
	SignalParser(QWidget* window, int poll_time, float sample_resolution);
	~SignalParser();

	void runOn(std::string portname);
	void disconnect();

	std::array<int, 3> getInput();
	std::array<int, 5> getOutput();

	bool loadModel(std::string model_name);
private:
	QWidget* qWnd;
	const int POLL_TIME; // ms
	const float SAMPLE_TIME;


	bool truncateBuffer(char* buffer, unsigned int content_length);
	void buildInputs(char* buffer);
	void calcOutputs();


	std::array<int, 3> inputs = std::array<int, 3>();
	std::array<int, 5> outputs = std::array<int, 5>();
	std::mutex inLock;
	std::mutex outLock;
	DumbLstm* lstm = nullptr;

	bool disconnect_signal = false;
};