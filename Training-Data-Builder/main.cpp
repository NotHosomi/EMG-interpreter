#include <iostream>
#include <fstream>
#include <windows.h>
#include <string>
#include "Serial.h"
#include "Timer.h"

#define _CRT_SECURE_NO_WARNINGS

// disable to speed up testing
#define PICK_FILE 0
#define PICK_COM 1

// The Arduino writes about 70 per second
const int POLL_TIME = 10; // ms
const float SAMPLE_RESOLUTION = 10; // per sec
const float SAMPLE_TIME = 1 / SAMPLE_RESOLUTION; // in seconds

struct sample
{
    int inputs[3] = { 0, 0, 0 };
    bool targets[5] = { false, false, false, false, false };
};
bool truncateBuffer(char* buffer, unsigned int content_length)
{
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
        printf("Bad truncate buffer: ");
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

std::string genLabels()
{
    // FYUL, FTYJ
    std::string s;
    s += "-";
    s += GetAsyncKeyState(VK_SPACE) ? "1" : "0";
    s += "-";
    s += GetAsyncKeyState('F') ? "1" : "0";
    s += "-";
    s += GetAsyncKeyState('T') ? "1" : "0";
    s += "-";
    s += GetAsyncKeyState('Y') ? "1" : "0";
    s += "-";
    s += GetAsyncKeyState('J') ? "1" : "0";
    s += "!";
    return s;
}

void saveSample(const char* inputs, const std::string& labels, std::ofstream& file)
{
    std::string s(inputs);
    s += labels;
    file << s;
    std::cout << s << std::endl;
}

void run(std::ofstream& file, Serial* port)
{
    // Prep buffers
    char buffer[256] = "";
    unsigned int buffer_size = 255;
    unsigned int length = 0;
    Timer tmr;

    do
    {
        // Flush the buffer
        length = port->ReadData(buffer, buffer_size);
    } while (length == buffer_size);

    // Begin read cycle
    tmr.mark();
    while (port->IsConnected())
    {
        if (tmr.peek() > SAMPLE_TIME)
        {
            length = port->ReadData(buffer, buffer_size);
            buffer[length] = 0;
            truncateBuffer(buffer, length);
            saveSample(buffer, genLabels(), file);
            tmr.mark();
        }

        // Exit key
        if (GetAsyncKeyState(VK_ESCAPE))
        {
            break;
        }
        Sleep(POLL_TIME);
    }
    file << "\n";
}

int main() {

    //
    std::string filename;
    std::ofstream file;
#if PICK_FILE
    while (1)
    {
        std::cout << "Filename: ";
        std::cin >> filename;
        file.open("data/" + filename + ".emg", std::ios::app);
        if (file)
            break;
        std::cout << "Failed to open file " << filename << ".emg" << std::endl;
    }
#else
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    time_t tt = std::chrono::system_clock::to_time_t(now);
    tm time;
    localtime_s(&time, &tt);
    filename = std::to_string(time.tm_year + 1900) + "-"
        + std::to_string(time.tm_mon + 1) + "-"
        + std::to_string(time.tm_mday) + "_"
        + std::to_string(time.tm_hour) + "-"
        + std::to_string(time.tm_min) + "-"
        + std::to_string(time.tm_sec);
    file.open("data/" + filename + ".emg", std::ios::app);
    if (!file)
    {
        std::cout << "Failed to open file " << filename << ".emg" << std::endl;
        return 0xBAADDA7E;
    }
#endif
    std::cout << "Saving to file: " << filename << ".emg" << std::endl;

    // Open port
    Serial* port;
#if PICK_COM
    std::string portname;
    while (1)
    {
        std::cout << "Port: ";
        std::cin >> portname;
        std::cout << "Connecting..." << std::endl;
        port = new Serial( ("\\\\.\\" + portname).c_str());
        if (port->IsConnected())
        {
            std::cout << "Connected successfully" << std::endl;
            break;
        }
        std::cout << "\nFailed to open " << portname << "\nPlease try again..." << std::endl;
    }
#else
    std::cout << "Connecting..." << std::endl;
    port = new Serial("\\\\.\\com5");
    if (!port->IsConnected())
    {
        std::cout << "Failed to open open com5" << std::endl;
        return 0xBAADDA7E;
    }
    std::cout << "Connected successfully" << std::endl;
#endif

    std::string cmd = "";
    do
    {
        printf("Recording new sequence...\n");
        Sleep(1000);
        run(file, port);
        Sleep(1000);
        // clear input buffer
        std::cout << "\nPAUSED [cont/exit]" << std::endl;
        do
        {
            std::cin >> cmd;
        } while (cmd != "cont" && cmd != "exit");
    } while (cmd != "exit");

    delete port;
    file.close();
    return 0;
}