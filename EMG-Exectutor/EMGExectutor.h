#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_EMGExectutor.h"
#include "SignalParser.h"

class EMGExectutor : public QMainWindow
{
    Q_OBJECT

public:
    EMGExectutor(QWidget *parent = Q_NULLPTR);
    ~EMGExectutor();

private:
    Ui::EMGExectutorClass ui;

    SignalParser* parser = nullptr;

private slots:
    void connectPressed();
    void disconnectPressed();
};
