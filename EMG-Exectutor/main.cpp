#include "EMGExectutor.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    EMGExectutor w;
    w.show();
    return a.exec();
}
