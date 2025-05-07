#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <QDirIterator>
#include <QImageReader>
#include <QFileInfo>
#include <QDateTime>
#include <QStandardPaths>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , currentImageIndex(-1)
    , isCurrentImagePositive(false)
{
    ui->setupUi(this);

    // Initialize MLP with 512x512 input, 128 hidden neurons, and 1 output neuron
    mlp = new MLP(512 * 512, 128, 1, "sigmoid", "sigmoid");

    // Initialize worker thread
    worker = new TrainingWorker(mlp);
    worker->moveToThread(&workerThread);

    // Connect signals and slots
    connect(&workerThread, &QThread::finished, worker, &QObject::deleteLater);
    connect(worker, &TrainingWorker::progressUpdated, this, &MainWindow::onTrainingProgressUpdated);
    connect(worker, &TrainingWorker::trainingComplete, this, &MainWindow::onTrainingComplete);
    connect(worker, &TrainingWorker::evaluationComplete, this, &MainWindow::onEvaluationComplete);

    // Start worker thread
    workerThread.start();

    // Initialize UI
    initializeUI();

    // Connect tab changed signal
    connect(ui->tabWidget, &QTabWidget::currentChanged, this, &MainWindow::onTabChanged);
}

MainWindow::~MainWindow()
{
    // Stop worker thread
    worker->stop();
    workerThread.quit();
    workerThread.wait();

    // Clean up
    delete mlp;
    delete ui;

    // Clean up graphics scenes
    delete inputLayerScene;
    delete hiddenLayerScene;
    delete outputLayerScene;
}

void MainWindow::initializeUI()
{
    // Set window title
    setWindowTitle("MLP Image Classifier");

    // Initialize graphics scenes
    inputLayerScene = new QGraphicsScene(this);
    hiddenLayerScene = new QGraphicsScene(this);
    outputLayerScene = new QGraphicsScene(this);

    // Set scenes for graphics views
    ui->gvInputLayer->setScene(inputLayerScene);
    ui->gvHiddenLayer->setScene(hiddenLayerScene);
    ui->gvOutputLayer->setScene(outputLayerScene);

    // Initialize UI elements
    ui->lblPositiveDir->setText("Not set");
    ui->lblNegativeDir->setText("Not set");
    ui->lblCurrentImage->setText("No image loaded");
    ui->lblPrediction->setText("No prediction");
    ui->lblAccuracy->setText("No evaluation");

    // Set default values for training parameters
    ui->sbHiddenNeurons->setValue(128);
    ui->cbHiddenActivation->addItem("sigmoid");
    ui->cbHiddenActivation->addItem("relu");
    ui->cbHiddenActivation->addItem("tanh");
    ui->sbLearningRate->setValue(0.01);
    ui->sbEpochs->setValue(100);
    ui->sbBatchSize->setValue(10);
    ui->cbShuffle->setChecked(true);
    ui->sbBias->setValue(0.0);

    // Disable buttons that require data
    ui->btnTrain->setEnabled(false);
    ui->btnEvaluate->setEnabled(false);
    ui->btnNextImage->setEnabled(false);
    ui->btnPrevImage->setEnabled(false);

    // Set progress bar
    ui->progressBar->setValue(0);
    ui->progressBar->setVisible(false);
}

void MainWindow::loadImagesFromDir(const QString& dir, QStringList& imageList)
{
    imageList.clear();

    QDirIterator it(dir, QStringList() << "*.png" << "*.bmp", QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        QString filePath = it.next();
        QImageReader reader(filePath);
        if (reader.canRead()) {
            imageList.append(filePath);
        }
    }
}

void MainWindow::updateCurrentImage()
{
    if (currentImageIndex < 0 ||
        (isCurrentImagePositive && positiveImages.isEmpty()) ||
        (!isCurrentImagePositive && negativeImages.isEmpty())) {
        // No image to display
        ui->lblCurrentImage->setText("No image loaded");
        ui->lblCurrentImage->setPixmap(QPixmap());
        ui->lblPrediction->setText("No prediction");
        currentImage = QImage();
        return;
    }

    // Get current image path
    QString imagePath;
    if (isCurrentImagePositive) {
        imagePath = positiveImages.at(currentImageIndex);
    } else {
        imagePath = negativeImages.at(currentImageIndex);
    }

    // Load image
    QImageReader reader(imagePath);
    currentImage = reader.read();

    if (currentImage.isNull()) {
        ui->lblCurrentImage->setText("Failed to load image");
        ui->lblCurrentImage->setPixmap(QPixmap());
        ui->lblPrediction->setText("No prediction");
        return;
    }

    // Display image
    QPixmap pixmap = QPixmap::fromImage(currentImage.scaled(ui->lblCurrentImage->width(),
                                                           ui->lblCurrentImage->height(),
                                                           Qt::KeepAspectRatio,
                                                           Qt::SmoothTransformation));
    ui->lblCurrentImage->setPixmap(pixmap);
    ui->lblCurrentImage->setText("");

    // Update image info
    QFileInfo fileInfo(imagePath);
    QString imageInfo = QString("%1 (%2x%3) - %4")
                            .arg(fileInfo.fileName())
                            .arg(currentImage.width())
                            .arg(currentImage.height())
                            .arg(isCurrentImagePositive ? "Positive" : "Negative");
    ui->lblImageInfo->setText(imageInfo);

    // Make prediction
    float prediction = mlp->predict(currentImage);
    QString predictionText = QString("Prediction: %1 (Threshold: 0.5)")
                                .arg(prediction, 0, 'f', 4);
    ui->lblPrediction->setText(predictionText);

    // Update layer visualizations
    updateLayerVisualizations();
}

void MainWindow::updateLayerVisualizations()
{
    if (!currentImage.isNull()) {
        updateInputLayerVisualization();
        updateHiddenLayerVisualization();
        updateOutputLayerVisualization();
    }
}

void MainWindow::updateInputLayerVisualization()
{
    // Clear scene
    inputLayerScene->clear();

    // Add current image to scene
    QImage processedImage = currentImage;
    if (processedImage.format() != QImage::Format_Grayscale8) {
        processedImage = processedImage.convertToFormat(QImage::Format_Grayscale8);
    }

    if (processedImage.width() != 512 || processedImage.height() != 512) {
        processedImage = processedImage.scaled(512, 512, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    }

    QGraphicsPixmapItem* pixmapItem = inputLayerScene->addPixmap(QPixmap::fromImage(processedImage));
    inputLayerScene->setSceneRect(pixmapItem->boundingRect());
    ui->gvInputLayer->fitInView(inputLayerScene->sceneRect(), Qt::KeepAspectRatio);
}

void MainWindow::updateHiddenLayerVisualization()
{
    // Clear scene
    hiddenLayerScene->clear();

    if (currentImage.isNull()) {
        return;
    }

    // Get hidden layer activations
    Eigen::VectorXf input = mlp->preprocessImage(currentImage);
    mlp->forward(input);

    // Get hidden layer
    const Layer& hiddenLayer = mlp->getLayers()[0];
    const Eigen::VectorXf& activations = hiddenLayer.getLastOutput();

    // Determine grid size
    int hiddenSize = hiddenLayer.getOutputSize();
    int gridSize = static_cast<int>(std::ceil(std::sqrt(hiddenSize)));

    // Calculate cell size
    int cellSize = std::min(ui->gvHiddenLayer->width() / gridSize, ui->gvHiddenLayer->height() / gridSize);

    // Draw activations as a grid of colored squares
    for (int i = 0; i < hiddenSize; ++i) {
        int row = i / gridSize;
        int col = i % gridSize;

        // Map activation to color intensity (0-255)
        int intensity = static_cast<int>(activations(i) * 255);
        QColor color(intensity, intensity, intensity);

        // Create rectangle
        QGraphicsRectItem* rect = hiddenLayerScene->addRect(col * cellSize, row * cellSize, cellSize, cellSize);
        rect->setBrush(QBrush(color));
        rect->setPen(QPen(Qt::black));
    }

    // Set scene rect
    hiddenLayerScene->setSceneRect(0, 0, gridSize * cellSize, gridSize * cellSize);
    ui->gvHiddenLayer->fitInView(hiddenLayerScene->sceneRect(), Qt::KeepAspectRatio);
}

void MainWindow::updateOutputLayerVisualization()
{
    // Clear scene
    outputLayerScene->clear();

    if (currentImage.isNull()) {
        return;
    }

    // Get output layer
    const Layer& outputLayer = mlp->getLayers()[1];
    const Eigen::VectorXf& output = outputLayer.getLastOutput();
    const Eigen::VectorXf& z = outputLayer.getLastZ();

    // Create text items
    QString rawOutput = QString("Raw output (z): %1").arg(z(0), 0, 'f', 4);
    QString activatedOutput = QString("Activated output (sigmoid): %1").arg(output(0), 0, 'f', 4);
    QString thresholdInfo = QString("Classification threshold: 0.5");
    QString classification = QString("Classification: %1").arg(output(0) >= 0.5 ? "Object Detected" : "Object Not Detected");

    QGraphicsTextItem* rawOutputItem = outputLayerScene->addText(rawOutput);
    QGraphicsTextItem* activatedOutputItem = outputLayerScene->addText(activatedOutput);
    QGraphicsTextItem* thresholdInfoItem = outputLayerScene->addText(thresholdInfo);
    QGraphicsTextItem* classificationItem = outputLayerScene->addText(classification);

    // Position text items
    rawOutputItem->setPos(0, 0);
    activatedOutputItem->setPos(0, 30);
    thresholdInfoItem->setPos(0, 60);
    classificationItem->setPos(0, 90);

    // Add visualization of output as a bar
    int barWidth = 200;
    int barHeight = 30;
    int barY = 150;

    // Background bar (0 to 1 range)
    QGraphicsRectItem* backgroundBar = outputLayerScene->addRect(0, barY, barWidth, barHeight);
    backgroundBar->setBrush(QBrush(Qt::lightGray));

    // Output value bar
    int outputBarWidth = static_cast<int>(output(0) * barWidth);
    QGraphicsRectItem* outputBar = outputLayerScene->addRect(0, barY, outputBarWidth, barHeight);
    outputBar->setBrush(QBrush(output(0) >= 0.5 ? Qt::green : Qt::red));

    // Threshold line
    int thresholdX = static_cast<int>(0.5 * barWidth);
    QGraphicsLineItem* thresholdLine = outputLayerScene->addLine(thresholdX, barY - 10, thresholdX, barY + barHeight + 10);
    thresholdLine->setPen(QPen(Qt::black, 2));

    // Add labels
    QGraphicsTextItem* zeroLabel = outputLayerScene->addText("0.0");
    QGraphicsTextItem* halfLabel = outputLayerScene->addText("0.5");
    QGraphicsTextItem* oneLabel = outputLayerScene->addText("1.0");

    zeroLabel->setPos(0, barY + barHeight + 10);
    halfLabel->setPos(thresholdX - 10, barY + barHeight + 10);
    oneLabel->setPos(barWidth - 20, barY + barHeight + 10);

    // Set scene rect
    outputLayerScene->setSceneRect(0, 0, barWidth + 50, barY + barHeight + 50);
    ui->gvOutputLayer->fitInView(outputLayerScene->sceneRect(), Qt::KeepAspectRatio);
}

void MainWindow::on_btnLoadPositive_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, "Select Directory with Positive Examples",
                                                   QStandardPaths::writableLocation(QStandardPaths::PicturesLocation),
                                                   QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if (dir.isEmpty()) {
        return;
    }

    positiveDir = dir;
    ui->lblPositiveDir->setText(dir);

    // Load images from directory
    loadImagesFromDir(dir, positiveImages);

    // Update UI
    ui->btnTrain->setEnabled(!positiveImages.isEmpty());
    ui->btnEvaluate->setEnabled(!positiveImages.isEmpty() && !negativeImages.isEmpty());

    // Set current image to first positive image
    if (!positiveImages.isEmpty()) {
        currentImageIndex = 0;
        isCurrentImagePositive = true;
        ui->btnNextImage->setEnabled(positiveImages.size() > 1);
        ui->btnPrevImage->setEnabled(false);
        updateCurrentImage();
    }
}

void MainWindow::on_btnLoadNegative_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, "Select Directory with Negative Examples",
                                                   QStandardPaths::writableLocation(QStandardPaths::PicturesLocation),
                                                   QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if (dir.isEmpty()) {
        return;
    }

    negativeDir = dir;
    ui->lblNegativeDir->setText(dir);

    // Load images from directory
    loadImagesFromDir(dir, negativeImages);

    // Update UI
    ui->btnEvaluate->setEnabled(!positiveImages.isEmpty() && !negativeImages.isEmpty());

    // If no current image, set to first negative image
    if (currentImageIndex < 0 && !negativeImages.isEmpty()) {
        currentImageIndex = 0;
        isCurrentImagePositive = false;
        ui->btnNextImage->setEnabled(negativeImages.size() > 1);
        ui->btnPrevImage->setEnabled(false);
        updateCurrentImage();
    }
}

void MainWindow::on_btnNextImage_clicked()
{
    if (isCurrentImagePositive) {
        if (currentImageIndex < positiveImages.size() - 1) {
            currentImageIndex++;
            ui->btnPrevImage->setEnabled(true);
            ui->btnNextImage->setEnabled(currentImageIndex < positiveImages.size() - 1);
            updateCurrentImage();
        }
    } else {
        if (currentImageIndex < negativeImages.size() - 1) {
            currentImageIndex++;
            ui->btnPrevImage->setEnabled(true);
            ui->btnNextImage->setEnabled(currentImageIndex < negativeImages.size() - 1);
            updateCurrentImage();
        }
    }
}

void MainWindow::on_btnPrevImage_clicked()
{
    if (currentImageIndex > 0) {
        currentImageIndex--;
        ui->btnPrevImage->setEnabled(currentImageIndex > 0);
        ui->btnNextImage->setEnabled(true);
        updateCurrentImage();
    }
}

void MainWindow::on_btnTrain_clicked()
{
    // Update MLP with current parameters
    int hiddenNeurons = ui->sbHiddenNeurons->value();
    QString hiddenActivation = ui->cbHiddenActivation->currentText();

    // If hidden neurons changed, recreate MLP
    if (hiddenNeurons != mlp->getLayers()[0].getOutputSize()) {
        delete mlp;
        mlp = new MLP(512 * 512, hiddenNeurons, 1, hiddenActivation.toStdString(), "sigmoid");

        // Update worker
        worker->stop();
        delete worker;
        worker = new TrainingWorker(mlp);
        worker->moveToThread(&workerThread);

        // Connect signals and slots
        connect(worker, &TrainingWorker::progressUpdated, this, &MainWindow::onTrainingProgressUpdated);
        connect(worker, &TrainingWorker::trainingComplete, this, &MainWindow::onTrainingComplete);
        connect(worker, &TrainingWorker::evaluationComplete, this, &MainWindow::onEvaluationComplete);
    }

    // Set training parameters
    worker->setPositiveDir(positiveDir);
    worker->setNegativeDir(negativeDir);
    worker->setLearningRate(ui->sbLearningRate->value());
    worker->setEpochs(ui->sbEpochs->value());
    worker->setBatchSize(ui->sbBatchSize->value());
    worker->setShuffle(ui->cbShuffle->isChecked());

    // Disable UI elements during training
    ui->btnTrain->setEnabled(false);
    ui->btnEvaluate->setEnabled(false);
    ui->btnExportModel->setEnabled(false);
    ui->btnImportModel->setEnabled(false);
    ui->progressBar->setValue(0);
    ui->progressBar->setVisible(true);

    // Start training
    QMetaObject::invokeMethod(worker, "train", Qt::QueuedConnection);
}

void MainWindow::on_btnEvaluate_clicked()
{
    // Disable UI elements during evaluation
    ui->btnTrain->setEnabled(false);
    ui->btnEvaluate->setEnabled(false);
    ui->btnExportModel->setEnabled(false);
    ui->btnImportModel->setEnabled(false);

    // Set evaluation parameters
    worker->setPositiveDir(positiveDir);
    worker->setNegativeDir(negativeDir);

    // Start evaluation
    QMetaObject::invokeMethod(worker, "evaluate", Qt::QueuedConnection);
}

void MainWindow::on_btnExportModel_clicked()
{
    QString filePath = QFileDialog::getSaveFileName(this, "Export Model",
                                                   QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),
                                                   "JSON Files (*.json)");

    if (filePath.isEmpty()) {
        return;
    }

    // Add .json extension if not present
    if (!filePath.endsWith(".json", Qt::CaseInsensitive)) {
        filePath += ".json";
    }

    // Save model to JSON
    QJsonObject json = mlp->saveToJson();
    QJsonDocument doc(json);
    QByteArray jsonData = doc.toJson(QJsonDocument::Indented);

    // Write to file
    QFile file(filePath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(jsonData);
        file.close();
        QMessageBox::information(this, "Export Successful", "Model exported successfully.");
    } else {
        QMessageBox::critical(this, "Export Failed", "Failed to write model to file.");
    }
}

void MainWindow::on_btnImportModel_clicked()
{
    QString filePath = QFileDialog::getOpenFileName(this, "Import Model",
                                                   QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),
                                                   "JSON Files (*.json)");

    if (filePath.isEmpty()) {
        return;
    }

    // Read JSON file
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, "Import Failed", "Failed to open model file.");
        return;
    }

    QByteArray jsonData = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(jsonData);
    if (doc.isNull() || !doc.isObject()) {
        QMessageBox::critical(this, "Import Failed", "Invalid JSON format.");
        return;
    }

    // Load model from JSON
    if (mlp->loadFromJson(doc.object())) {
        QMessageBox::information(this, "Import Successful", "Model imported successfully.");

        // Update UI
        ui->sbHiddenNeurons->setValue(mlp->getLayers()[0].getOutputSize());
        ui->cbHiddenActivation->setCurrentText(QString::fromStdString(mlp->getLayers()[0].getActivationFunction()));

        // Update current image if available
        if (!currentImage.isNull()) {
            updateCurrentImage();
        }
    } else {
        QMessageBox::critical(this, "Import Failed", "Failed to load model from file. The model architecture may be incompatible.");
    }
}

void MainWindow::onTrainingProgressUpdated(int epoch, int totalEpochs, float loss)
{
    // Update progress bar
    ui->progressBar->setValue(static_cast<int>((static_cast<float>(epoch) / totalEpochs) * 100));

    // Update status bar
    statusBar()->showMessage(QString("Training: Epoch %1/%2, Loss: %3").arg(epoch).arg(totalEpochs).arg(loss, 0, 'f', 6));
}

void MainWindow::onTrainingComplete(float finalLoss)
{
    // Update UI
    ui->btnTrain->setEnabled(true);
    ui->btnEvaluate->setEnabled(!positiveImages.isEmpty() && !negativeImages.isEmpty());
    ui->btnExportModel->setEnabled(true);
    ui->btnImportModel->setEnabled(true);
    ui->progressBar->setVisible(false);

    // Update status bar
    statusBar()->showMessage(QString("Training complete. Final loss: %1").arg(finalLoss, 0, 'f', 6), 5000);

    // Update current image if available
    if (!currentImage.isNull()) {
        updateCurrentImage();
    }
}

void MainWindow::onEvaluationComplete(float accuracy, int truePositives, int trueNegatives, int falsePositives, int falseNegatives)
{
    // Update UI
    ui->btnTrain->setEnabled(true);
    ui->btnEvaluate->setEnabled(true);
    ui->btnExportModel->setEnabled(true);
    ui->btnImportModel->setEnabled(true);

    // Update accuracy label
    QString accuracyText = QString("Accuracy: %1%\nTrue Positives: %2\nTrue Negatives: %3\nFalse Positives: %4\nFalse Negatives: %5")
                              .arg(accuracy * 100.0, 0, 'f', 2)
                              .arg(truePositives)
                              .arg(trueNegatives)
                              .arg(falsePositives)
                              .arg(falseNegatives);
    ui->lblAccuracy->setText(accuracyText);

    // Update status bar
    statusBar()->showMessage(QString("Evaluation complete. Accuracy: %1%").arg(accuracy * 100.0, 0, 'f', 2), 5000);
}

void MainWindow::onTabChanged(int index)
{
    // Update visualizations when switching to visualization tabs
    if (index >= 1 && !currentImage.isNull()) {
        updateLayerVisualizations();
    }
}
