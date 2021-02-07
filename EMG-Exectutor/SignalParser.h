#pragma once
#include <array>
#include <string>
#include <mutex>
#include <qwidget.h>

class SignalParser
{
public:
	SignalParser(QWidget* window, int poll_time, float sample_resolution);

	void runOn(std::string portname);
	void disconnect();

	std::array<int, 3> getLatest();
private:
	QWidget* qWnd;
	const int POLL_TIME; // ms
	const float SAMPLE_TIME;


	bool truncateBuffer(char* buffer, unsigned int content_length);


	std::array<int, 3> latest = std::array<int, 3>();
	std::mutex outputLock;

	bool disconnect_signal = false;
};