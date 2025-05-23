QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets charts

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

# Eigen is a header-only library, so we just need to add the include path
# Make sure to install Eigen or adjust the path accordingly
INCLUDEPATH += /usr/local/include/Eigen

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    mlp.cpp \
    layer.cpp \
    trainingworker.cpp \
    losscurvewidget.cpp

HEADERS += \
    mainwindow.h \
    mlp.h \
    layer.h \
    trainingworker.h \
    losscurvewidget.h

FORMS += \
    mainwindow.ui

TRANSLATIONS += \
    sensuser_en_US.ts
CONFIG += lrelease
CONFIG += embed_translations

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
