#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_EMGExectutor.h"
#include "SignalParser.h"
#include "BakedNetwork.h"
#include <qtimer.h>

class EMGExectutor : public QMainWindow
{
    Q_OBJECT

public:
    EMGExectutor(QWidget *parent = Q_NULLPTR);
    ~EMGExectutor();

private:
    Ui::EMGExectutorClass ui;
    QTimer* refresh;
    bool hasModel = false;
    bool isRunning = false;

    SignalParser* parser = nullptr;
    BakedNetwork* rnn = nullptr; 

    std::thread parse_thread;
    std::thread net_thread;

private slots:
    void connectPressed();
    void disconnectPressed();
    void updateBars();
    void loadModel();
};
