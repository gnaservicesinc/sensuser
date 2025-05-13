#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include <QImage>
#include <QFileDialog>
#include <QMessageBox>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFile>
#include <QDir>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QTimer>
#include <QListWidget>
#include <QSpinBox>
#include <QComboBox>

#include "mlp.h"
#include "trainingworker.h"
#include "losscurvewidget.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    // Setup & Training tab
    void on_btnLoadPositive_clicked();
    void on_btnLoadNegative_clicked();
    void on_btnNextImage_clicked();
    void on_btnPrevImage_clicked();
    void on_btnTrain_clicked();
    void on_btnEvaluate_clicked();
    void on_btnExportModel_clicked();
    void on_btnImportModel_clicked();

    // Training worker signals
    void onTrainingProgressUpdated(int epoch, int totalEpochs, float loss);
    void onEpochCompleted(int epoch, float loss, float validationLoss);
    void onTrainingComplete(float finalLoss);
    void onEvaluationComplete(float accuracy, int truePositives, int trueNegatives, int falsePositives, int falseNegatives);

    // Tab changed
    void onTabChanged(int index);

    // Hidden layers configuration
    void onAddHiddenLayerClicked();
    void onRemoveHiddenLayerClicked();
    void onHiddenLayerValueChanged(int value);
    void onHiddenLayerSelectorChanged(int index);

private:
    Ui::MainWindow *ui;

    // MLP
    MLP* mlp;

    // Training worker
    QThread workerThread;
    TrainingWorker* worker;

    // Image data
    QString positiveDir;
    QString negativeDir;
    QStringList positiveImages;
    QStringList negativeImages;
    int currentImageIndex;
    QImage currentImage;
    bool isCurrentImagePositive;

    // Graphics scenes for visualization
    QGraphicsScene* inputLayerScene;
    QGraphicsScene* hiddenLayerScene;
    QGraphicsScene* outputLayerScene;

    // Loss curve widget
    LossCurveWidget* lossCurveWidget;

    // Hidden layers configuration
    QListWidget* hiddenLayersList;
    QPushButton* addHiddenLayerButton;
    QPushButton* removeHiddenLayerButton;
    std::vector<int> hiddenLayerSizes;

    // Hidden layer visualization selector
    QComboBox* hiddenLayerSelector;
    int currentHiddenLayerIndex;

    // Initialize UI
    void initializeUI();

    // Setup hidden layers UI
    void setupHiddenLayersUI();

    // Update hidden layers UI from model
    void updateHiddenLayersUIFromModel();

    // Create MLP from UI configuration
    void createMLPFromUIConfig();

    // Load images from directory
    void loadImagesFromDir(const QString& dir, QStringList& imageList);

    // Update current image display
    void updateCurrentImage();

    // Update layer visualizations
    void updateLayerVisualizations();

    // Update input layer visualization
    void updateInputLayerVisualization();

    // Update hidden layer visualization
    void updateHiddenLayerVisualization();

    // Update output layer visualization
    void updateOutputLayerVisualization();
};

#endif // MAINWINDOW_H
