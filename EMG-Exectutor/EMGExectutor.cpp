#include "EMGExectutor.h"
#include <QtCore>
#include <QtGui>
#include <QMessageBox>

EMGExectutor::EMGExectutor(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    QObject::connect(ui.BtnConnect, SIGNAL(clicked()), SLOT(connectPressed()));
    QObject::connect(ui.BtnDisconnect, SIGNAL(clicked()), SLOT(disconnectPressed()));

    //parser = new SignalParser(parent, 100, 10);
    //std::thread parse_thread([this] { this->parser->runOn("COM5"); });
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
    QMessageBox::warning(this, "Title X", "WOW!");
   
}

void EMGExectutor::disconnectPressed()
{
    QMessageBox::warning(this, "Poggie woggie", "Nyaa!\t\t");
}
