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
    parser->disconnect();
    rnn->stop();

    parse_thread.join();
    if (parser != nullptr)
    {
        delete parser;
        parser = nullptr;
    }
    net_thread.join();
    if (rnn != nullptr)
    {
        delete rnn;
        rnn = nullptr;
    }
}

void EMGExectutor::connectPressed()
{
    // Open network and serial threads
    // network
    if (!hasModel)
    {
        QMessageBox::warning(this, "Connection Failed", "No model has been mounted");
        return;
    }
    net_thread = std::thread(&BakedNetwork::run, rnn);

    // serial
    //std::string portname = ui.textEdit_COMport->text().toStdString();
    //std::thread parse_thread([this, portname] { this->parser->runOn(portname); });
    parser->setPort(ui.textEdit_COMport->text().toStdString());
    parse_thread = std::thread(&SignalParser::runOn, parser);

    refresh->start(100);
}

void EMGExectutor::disconnectPressed()
{
    refresh->stop();
    parser->disconnect();
    rnn->stop();

    parse_thread.join();
    net_thread.join();
    hasModel = false;
}

void EMGExectutor::updateBars()
{
    if (parser == nullptr || rnn == nullptr)
        return;
    std::array<int, 3> inputs;
    inputs = parser->getInput();
    VectorXd outputs = rnn->getOutput();
    outputs *= 100;

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
    hasModel = false;
    if (rnn)
    {
        delete rnn;
        rnn = nullptr;
        parser->setModel(nullptr);
    }
    std::string model_name = ui.textEdit_TempLoad->text().toStdString();
    if (model_name == "")
    {
        QMessageBox::warning(this, "Invalid model", "No model has been specified");
        return;
    }
    std::ifstream model_file("nets/" + model_name + ".dat", std::ios::binary | std::ios::in);
    if (!model_file)
    {
        QMessageBox::warning(this, "Invalid model", ("Failed to load model \"nets/" + model_name + "\"").c_str());
        return;
    }
    rnn = new BakedNetwork(model_file);
    parser->setModel(rnn);
    hasModel = true;
}

