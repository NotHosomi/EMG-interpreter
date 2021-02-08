#include "EMGExectutor.h"
#include <QtCore>
#include <QtGui>
#include <QMessageBox>

EMGExectutor::EMGExectutor(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    parser = new SignalParser(this, 100, 10);
    QObject::connect(ui.BtnConnect, SIGNAL(clicked()), SLOT(connectPressed()));
    QObject::connect(ui.BtnDisconnect, SIGNAL(clicked()), SLOT(disconnectPressed()));
    QObject::connect(ui.BtnTempLoad, SIGNAL(clicked()), SLOT(loadModel()));

    refresh = new QTimer(this);
    QObject::connect(refresh, SIGNAL(timeout()), this, SLOT(updateBars()));
}

EMGExectutor::~EMGExectutor()
{
    if (parser != nullptr)
    {
        delete parser;
        parser = nullptr;
    }
}

void EMGExectutor::connectPressed()
{
    if (!hasModel)
    {
        QMessageBox::warning(this, "Invalid model", "No model has been loaded");
        return;
    }
    std::string portname = ui.textEdit_COMport->text().toStdString();
    std::thread parse_thread([this, portname] { this->parser->runOn(portname); });
    refresh->start(100);
}

void EMGExectutor::disconnectPressed()
{
    refresh->stop();
    parser->disconnect();
}

void EMGExectutor::updateBars()
{
    std::array<int, 3> inputs;
    inputs = parser->getInput();
    std::array<int, 5> outputs;
    outputs = parser->getOutput();
    ui.Bar_Input0->setValue(inputs[0]);
    ui.Bar_Input1->setValue(inputs[1]);
    ui.Bar_Input2->setValue(inputs[2]);
    ui.Bar_Output0->setValue(outputs[0]);
    ui.Bar_Output1->setValue(outputs[1]);
    ui.Bar_Output2->setValue(outputs[2]);
    ui.Bar_Output3->setValue(outputs[3]);
    ui.Bar_Output4->setValue(outputs[4]);
}

void EMGExectutor::loadModel()
{
    if (parser->loadModel(ui.textEdit_TempLoad->text().toStdString()))
    {
        hasModel = true;
    }
}

