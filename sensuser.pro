QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets charts

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

## Internal MLP removed; Eigen not needed

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    trainingworker.cpp \
    losscurvewidget.cpp \
    noodlenet_backend.cpp

HEADERS += \
    mainwindow.h \
    trainingworker.h \
    losscurvewidget.h \
    optimizertypes.h \
    noodlenet_backend.h

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

# Link system-installed libnoodlenet by default
INCLUDEPATH += /usr/local/include
LIBS += -L/usr/local/lib -lnoodlenet

# Workaround for newer macOS SDKs where AGL is removed; strip if injected
LIBS -= -framework AGL
