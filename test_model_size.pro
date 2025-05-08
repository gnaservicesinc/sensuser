QT += core gui

CONFIG += c++17 console
CONFIG -= app_bundle

INCLUDEPATH += /usr/local/include/Eigen

SOURCES += \
    test_model_size.cpp \
    mlp.cpp \
    layer.cpp

HEADERS += \
    mlp.h \
    layer.h

TARGET = test_model_size
