#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_EMGExectutor.h"
#include "SignalParser.h"
#include <qtimer.h>

class EMGExectutor : public QMainWindow
{
    Q_OBJECT

public:
    EMGExectutor(QWidget *parent = Q_NULLPTR);
    ~EMGExectutor();

private:
    Ui::EMGExectutorClass ui;
    SignalParser* parser = nullptr;
    QTimer* refresh;
    bool hasModel = false;

private slots:
    void connectPressed();
    void disconnectPressed();
    void updateBars();
    void loadModel();
};
