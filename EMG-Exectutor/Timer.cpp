#include "Timer.h"

using namespace std::chrono;
Timer::Timer()
{
	t = steady_clock::now();
}

float Timer::mark()
{
	steady_clock::time_point old = t;
	t = steady_clock::now();
	duration<float> delta = t - old;
	return delta.count();
}

float Timer::peek() const
{
	return duration<float>(steady_clock::now() - t).count();
}