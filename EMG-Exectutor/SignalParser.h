#pragma once
#include <array>
#include <string>
#include <mutex>
#include <qwidget.h>
#include "BakedNetwork.h"

class SignalParser
{
public:
	SignalParser(QWidget* window, int poll_time, float sample_resolution);
	~SignalParser();

	void setPort(std::string port);
	void runOn();
	void disconnect();

	std::array<int, 3> getInput();

	void setModel(BakedNetwork* model);
private:
	QWidget* qWnd;
	const int POLL_TIME; // ms
	const float SAMPLE_TIME;


	bool truncateBuffer(char* buffer, unsigned int content_length);
	void buildInputs(char* buffer);

	std::string portname = "";
	std::array<int, 3> inputs = { 0, 0, 0 };
	std::mutex inLock;
	BakedNetwork* rnn = nullptr;

	std::atomic<bool> disconnect_signal = false;
};