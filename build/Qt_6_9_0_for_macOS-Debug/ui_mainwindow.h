/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.9.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QTabWidget *tabWidget;
    QWidget *tabSetupTraining;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout_3;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QLabel *label;
    QLabel *lblPositiveDir;
    QPushButton *btnLoadPositive;
    QLabel *label_2;
    QLabel *lblNegativeDir;
    QPushButton *btnLoadNegative;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout_2;
    QLabel *label_3;
    QSpinBox *sbHiddenNeurons;
    QLabel *label_4;
    QComboBox *cbHiddenActivation;
    QLabel *label_5;
    QDoubleSpinBox *sbBias;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_3;
    QLabel *label_6;
    QDoubleSpinBox *sbLearningRate;
    QLabel *label_7;
    QSpinBox *sbEpochs;
    QLabel *label_8;
    QSpinBox *sbBatchSize;
    QLabel *label_9;
    QCheckBox *cbShuffle;
    QGroupBox *groupBox_4;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *btnExportModel;
    QPushButton *btnImportModel;
    QSpacerItem *verticalSpacer;
    QVBoxLayout *verticalLayout_4;
    QGroupBox *groupBox_5;
    QVBoxLayout *verticalLayout_5;
    QLabel *lblCurrentImage;
    QLabel *lblImageInfo;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *btnPrevImage;
    QPushButton *btnNextImage;
    QLabel *lblPrediction;
    QGroupBox *groupBox_6;
    QVBoxLayout *verticalLayout_6;
    QHBoxLayout *horizontalLayout_4;
    QPushButton *btnTrain;
    QPushButton *btnEvaluate;
    QProgressBar *progressBar;
    QLabel *lblAccuracy;
    QWidget *tabInputLayer;
    QVBoxLayout *verticalLayout_7;
    QGraphicsView *gvInputLayer;
    QWidget *tabHiddenLayer;
    QVBoxLayout *verticalLayout_8;
    QGraphicsView *gvHiddenLayer;
    QWidget *tabOutputLayer;
    QVBoxLayout *verticalLayout_9;
    QGraphicsView *gvOutputLayer;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(1024, 768);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setObjectName("verticalLayout");
        tabWidget = new QTabWidget(centralwidget);
        tabWidget->setObjectName("tabWidget");
        tabSetupTraining = new QWidget();
        tabSetupTraining->setObjectName("tabSetupTraining");
        verticalLayout_2 = new QVBoxLayout(tabSetupTraining);
        verticalLayout_2->setObjectName("verticalLayout_2");
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName("horizontalLayout");
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setObjectName("verticalLayout_3");
        groupBox = new QGroupBox(tabSetupTraining);
        groupBox->setObjectName("groupBox");
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setObjectName("gridLayout");
        label = new QLabel(groupBox);
        label->setObjectName("label");

        gridLayout->addWidget(label, 0, 0, 1, 1);

        lblPositiveDir = new QLabel(groupBox);
        lblPositiveDir->setObjectName("lblPositiveDir");

        gridLayout->addWidget(lblPositiveDir, 0, 1, 1, 1);

        btnLoadPositive = new QPushButton(groupBox);
        btnLoadPositive->setObjectName("btnLoadPositive");

        gridLayout->addWidget(btnLoadPositive, 0, 2, 1, 1);

        label_2 = new QLabel(groupBox);
        label_2->setObjectName("label_2");

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        lblNegativeDir = new QLabel(groupBox);
        lblNegativeDir->setObjectName("lblNegativeDir");

        gridLayout->addWidget(lblNegativeDir, 1, 1, 1, 1);

        btnLoadNegative = new QPushButton(groupBox);
        btnLoadNegative->setObjectName("btnLoadNegative");

        gridLayout->addWidget(btnLoadNegative, 1, 2, 1, 1);


        verticalLayout_3->addWidget(groupBox);

        groupBox_2 = new QGroupBox(tabSetupTraining);
        groupBox_2->setObjectName("groupBox_2");
        gridLayout_2 = new QGridLayout(groupBox_2);
        gridLayout_2->setObjectName("gridLayout_2");
        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName("label_3");

        gridLayout_2->addWidget(label_3, 0, 0, 1, 1);

        sbHiddenNeurons = new QSpinBox(groupBox_2);
        sbHiddenNeurons->setObjectName("sbHiddenNeurons");
        sbHiddenNeurons->setMinimum(1);
        sbHiddenNeurons->setMaximum(1024);
        sbHiddenNeurons->setValue(128);

        gridLayout_2->addWidget(sbHiddenNeurons, 0, 1, 1, 1);

        label_4 = new QLabel(groupBox_2);
        label_4->setObjectName("label_4");

        gridLayout_2->addWidget(label_4, 1, 0, 1, 1);

        cbHiddenActivation = new QComboBox(groupBox_2);
        cbHiddenActivation->setObjectName("cbHiddenActivation");

        gridLayout_2->addWidget(cbHiddenActivation, 1, 1, 1, 1);

        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName("label_5");

        gridLayout_2->addWidget(label_5, 2, 0, 1, 1);

        sbBias = new QDoubleSpinBox(groupBox_2);
        sbBias->setObjectName("sbBias");
        sbBias->setMinimum(-10.000000000000000);
        sbBias->setMaximum(10.000000000000000);
        sbBias->setSingleStep(0.100000000000000);

        gridLayout_2->addWidget(sbBias, 2, 1, 1, 1);


        verticalLayout_3->addWidget(groupBox_2);

        groupBox_3 = new QGroupBox(tabSetupTraining);
        groupBox_3->setObjectName("groupBox_3");
        gridLayout_3 = new QGridLayout(groupBox_3);
        gridLayout_3->setObjectName("gridLayout_3");
        label_6 = new QLabel(groupBox_3);
        label_6->setObjectName("label_6");

        gridLayout_3->addWidget(label_6, 0, 0, 1, 1);

        sbLearningRate = new QDoubleSpinBox(groupBox_3);
        sbLearningRate->setObjectName("sbLearningRate");
        sbLearningRate->setDecimals(4);
        sbLearningRate->setMinimum(0.000100000000000);
        sbLearningRate->setMaximum(1.000000000000000);
        sbLearningRate->setSingleStep(0.001000000000000);
        sbLearningRate->setValue(0.010000000000000);

        gridLayout_3->addWidget(sbLearningRate, 0, 1, 1, 1);

        label_7 = new QLabel(groupBox_3);
        label_7->setObjectName("label_7");

        gridLayout_3->addWidget(label_7, 1, 0, 1, 1);

        sbEpochs = new QSpinBox(groupBox_3);
        sbEpochs->setObjectName("sbEpochs");
        sbEpochs->setMinimum(1);
        sbEpochs->setMaximum(1000);
        sbEpochs->setValue(100);

        gridLayout_3->addWidget(sbEpochs, 1, 1, 1, 1);

        label_8 = new QLabel(groupBox_3);
        label_8->setObjectName("label_8");

        gridLayout_3->addWidget(label_8, 2, 0, 1, 1);

        sbBatchSize = new QSpinBox(groupBox_3);
        sbBatchSize->setObjectName("sbBatchSize");
        sbBatchSize->setMinimum(1);
        sbBatchSize->setMaximum(100);
        sbBatchSize->setValue(10);

        gridLayout_3->addWidget(sbBatchSize, 2, 1, 1, 1);

        label_9 = new QLabel(groupBox_3);
        label_9->setObjectName("label_9");

        gridLayout_3->addWidget(label_9, 3, 0, 1, 1);

        cbShuffle = new QCheckBox(groupBox_3);
        cbShuffle->setObjectName("cbShuffle");
        cbShuffle->setChecked(true);

        gridLayout_3->addWidget(cbShuffle, 3, 1, 1, 1);


        verticalLayout_3->addWidget(groupBox_3);

        groupBox_4 = new QGroupBox(tabSetupTraining);
        groupBox_4->setObjectName("groupBox_4");
        horizontalLayout_3 = new QHBoxLayout(groupBox_4);
        horizontalLayout_3->setObjectName("horizontalLayout_3");
        btnExportModel = new QPushButton(groupBox_4);
        btnExportModel->setObjectName("btnExportModel");

        horizontalLayout_3->addWidget(btnExportModel);

        btnImportModel = new QPushButton(groupBox_4);
        btnImportModel->setObjectName("btnImportModel");

        horizontalLayout_3->addWidget(btnImportModel);


        verticalLayout_3->addWidget(groupBox_4);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Policy::Minimum, QSizePolicy::Policy::Expanding);

        verticalLayout_3->addItem(verticalSpacer);


        horizontalLayout->addLayout(verticalLayout_3);

        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setObjectName("verticalLayout_4");
        groupBox_5 = new QGroupBox(tabSetupTraining);
        groupBox_5->setObjectName("groupBox_5");
        verticalLayout_5 = new QVBoxLayout(groupBox_5);
        verticalLayout_5->setObjectName("verticalLayout_5");
        lblCurrentImage = new QLabel(groupBox_5);
        lblCurrentImage->setObjectName("lblCurrentImage");
        lblCurrentImage->setMinimumSize(QSize(512, 512));
        lblCurrentImage->setAlignment(Qt::AlignCenter);

        verticalLayout_5->addWidget(lblCurrentImage);

        lblImageInfo = new QLabel(groupBox_5);
        lblImageInfo->setObjectName("lblImageInfo");

        verticalLayout_5->addWidget(lblImageInfo);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName("horizontalLayout_2");
        btnPrevImage = new QPushButton(groupBox_5);
        btnPrevImage->setObjectName("btnPrevImage");

        horizontalLayout_2->addWidget(btnPrevImage);

        btnNextImage = new QPushButton(groupBox_5);
        btnNextImage->setObjectName("btnNextImage");

        horizontalLayout_2->addWidget(btnNextImage);


        verticalLayout_5->addLayout(horizontalLayout_2);

        lblPrediction = new QLabel(groupBox_5);
        lblPrediction->setObjectName("lblPrediction");

        verticalLayout_5->addWidget(lblPrediction);


        verticalLayout_4->addWidget(groupBox_5);

        groupBox_6 = new QGroupBox(tabSetupTraining);
        groupBox_6->setObjectName("groupBox_6");
        verticalLayout_6 = new QVBoxLayout(groupBox_6);
        verticalLayout_6->setObjectName("verticalLayout_6");
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName("horizontalLayout_4");
        btnTrain = new QPushButton(groupBox_6);
        btnTrain->setObjectName("btnTrain");

        horizontalLayout_4->addWidget(btnTrain);

        btnEvaluate = new QPushButton(groupBox_6);
        btnEvaluate->setObjectName("btnEvaluate");

        horizontalLayout_4->addWidget(btnEvaluate);


        verticalLayout_6->addLayout(horizontalLayout_4);

        progressBar = new QProgressBar(groupBox_6);
        progressBar->setObjectName("progressBar");
        progressBar->setValue(0);

        verticalLayout_6->addWidget(progressBar);

        lblAccuracy = new QLabel(groupBox_6);
        lblAccuracy->setObjectName("lblAccuracy");
        lblAccuracy->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);

        verticalLayout_6->addWidget(lblAccuracy);


        verticalLayout_4->addWidget(groupBox_6);


        horizontalLayout->addLayout(verticalLayout_4);


        verticalLayout_2->addLayout(horizontalLayout);

        tabWidget->addTab(tabSetupTraining, QString());
        tabInputLayer = new QWidget();
        tabInputLayer->setObjectName("tabInputLayer");
        verticalLayout_7 = new QVBoxLayout(tabInputLayer);
        verticalLayout_7->setObjectName("verticalLayout_7");
        gvInputLayer = new QGraphicsView(tabInputLayer);
        gvInputLayer->setObjectName("gvInputLayer");

        verticalLayout_7->addWidget(gvInputLayer);

        tabWidget->addTab(tabInputLayer, QString());
        tabHiddenLayer = new QWidget();
        tabHiddenLayer->setObjectName("tabHiddenLayer");
        verticalLayout_8 = new QVBoxLayout(tabHiddenLayer);
        verticalLayout_8->setObjectName("verticalLayout_8");
        gvHiddenLayer = new QGraphicsView(tabHiddenLayer);
        gvHiddenLayer->setObjectName("gvHiddenLayer");

        verticalLayout_8->addWidget(gvHiddenLayer);

        tabWidget->addTab(tabHiddenLayer, QString());
        tabOutputLayer = new QWidget();
        tabOutputLayer->setObjectName("tabOutputLayer");
        verticalLayout_9 = new QVBoxLayout(tabOutputLayer);
        verticalLayout_9->setObjectName("verticalLayout_9");
        gvOutputLayer = new QGraphicsView(tabOutputLayer);
        gvOutputLayer->setObjectName("gvOutputLayer");

        verticalLayout_9->addWidget(gvOutputLayer);

        tabWidget->addTab(tabOutputLayer, QString());

        verticalLayout->addWidget(tabWidget);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 1024, 24));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName("statusbar");
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MLP Image Classifier", nullptr));
        groupBox->setTitle(QCoreApplication::translate("MainWindow", "Data Loading", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "Positive Examples:", nullptr));
        lblPositiveDir->setText(QCoreApplication::translate("MainWindow", "Not set", nullptr));
        btnLoadPositive->setText(QCoreApplication::translate("MainWindow", "Load", nullptr));
        label_2->setText(QCoreApplication::translate("MainWindow", "Negative Examples:", nullptr));
        lblNegativeDir->setText(QCoreApplication::translate("MainWindow", "Not set", nullptr));
        btnLoadNegative->setText(QCoreApplication::translate("MainWindow", "Load", nullptr));
        groupBox_2->setTitle(QCoreApplication::translate("MainWindow", "Network Configuration", nullptr));
        label_3->setText(QCoreApplication::translate("MainWindow", "Hidden Neurons:", nullptr));
        label_4->setText(QCoreApplication::translate("MainWindow", "Hidden Activation:", nullptr));
        label_5->setText(QCoreApplication::translate("MainWindow", "Bias:", nullptr));
        groupBox_3->setTitle(QCoreApplication::translate("MainWindow", "Training Parameters", nullptr));
        label_6->setText(QCoreApplication::translate("MainWindow", "Learning Rate:", nullptr));
        label_7->setText(QCoreApplication::translate("MainWindow", "Epochs:", nullptr));
        label_8->setText(QCoreApplication::translate("MainWindow", "Batch Size:", nullptr));
        label_9->setText(QCoreApplication::translate("MainWindow", "Shuffle Data:", nullptr));
        cbShuffle->setText(QString());
        groupBox_4->setTitle(QCoreApplication::translate("MainWindow", "Model Management", nullptr));
        btnExportModel->setText(QCoreApplication::translate("MainWindow", "Export Model", nullptr));
        btnImportModel->setText(QCoreApplication::translate("MainWindow", "Import Model", nullptr));
        groupBox_5->setTitle(QCoreApplication::translate("MainWindow", "Current Image", nullptr));
        lblCurrentImage->setText(QCoreApplication::translate("MainWindow", "No image loaded", nullptr));
        lblImageInfo->setText(QString());
        btnPrevImage->setText(QCoreApplication::translate("MainWindow", "Previous", nullptr));
        btnNextImage->setText(QCoreApplication::translate("MainWindow", "Next", nullptr));
        lblPrediction->setText(QCoreApplication::translate("MainWindow", "No prediction", nullptr));
        groupBox_6->setTitle(QCoreApplication::translate("MainWindow", "Training & Evaluation", nullptr));
        btnTrain->setText(QCoreApplication::translate("MainWindow", "Train Model", nullptr));
        btnEvaluate->setText(QCoreApplication::translate("MainWindow", "Evaluate Model", nullptr));
        lblAccuracy->setText(QCoreApplication::translate("MainWindow", "No evaluation", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tabSetupTraining), QCoreApplication::translate("MainWindow", "Setup & Training", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tabInputLayer), QCoreApplication::translate("MainWindow", "Input Layer", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tabHiddenLayer), QCoreApplication::translate("MainWindow", "Hidden Layer", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tabOutputLayer), QCoreApplication::translate("MainWindow", "Output Layer", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
