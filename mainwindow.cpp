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
    connect(worker, &TrainingWorker::epochCompleted, this, &MainWindow::onEpochCompleted);
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

    // The lossCurveWidget is owned by the tabWidget, so it will be deleted automatically
}

void MainWindow::initializeUI()
{
    // Set window title
    setWindowTitle("MLP Image Classifier");

    // Initialize loss curve widget
    lossCurveWidget = new LossCurveWidget();

    // Create a new tab for the loss curve widget
    ui->tabWidget->addTab(lossCurveWidget, "Loss Curve");

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
    ui->cbHiddenActivation->addItem("sigmoid");
    ui->cbHiddenActivation->addItem("relu");
    ui->cbHiddenActivation->addItem("tanh");
    ui->sbLearningRate->setValue(0.01);
    ui->sbEpochs->setValue(100);
    ui->sbBatchSize->setValue(10);
    ui->cbShuffle->setChecked(true);
    ui->sbBias->setValue(0.0);

    // Setup hidden layers configuration UI
    setupHiddenLayersUI();

    // Create and set up the hidden layer selector for visualization
    hiddenLayerSelector = new QComboBox();
    hiddenLayerSelector->addItem("Hidden Layer 1");
    hiddenLayerSelector->setStyleSheet("background-color: white; color: black; font-weight: bold;");
    connect(hiddenLayerSelector, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onHiddenLayerSelectorChanged);

    // Add the selector to the Hidden Layer visualization tab
    QWidget* hiddenLayerTab = ui->tabWidget->widget(2); // Assuming Hidden Layer tab is at index 2
    QVBoxLayout* hiddenLayerLayout = qobject_cast<QVBoxLayout*>(hiddenLayerTab->layout());
    if (!hiddenLayerLayout) {
        hiddenLayerLayout = new QVBoxLayout(hiddenLayerTab);
        hiddenLayerLayout->addWidget(ui->gvHiddenLayer);
    }

    QHBoxLayout* selectorLayout = new QHBoxLayout();
    QLabel* selectorLabel = new QLabel("Select Hidden Layer:");
    selectorLabel->setStyleSheet("color: white; font-weight: bold;");
    selectorLayout->addWidget(selectorLabel);
    selectorLayout->addWidget(hiddenLayerSelector);
    selectorLayout->addStretch();

    // Insert the selector layout at the top of the tab
    hiddenLayerLayout->insertLayout(0, selectorLayout);

    // Disable buttons that require data
    ui->btnTrain->setEnabled(false);
    ui->btnEvaluate->setEnabled(false);
    ui->btnNextImage->setEnabled(false);
    ui->btnPrevImage->setEnabled(false);

    // Set progress bar
    ui->progressBar->setValue(0);
    ui->progressBar->setVisible(false);

    // Initialize current hidden layer index
    currentHiddenLayerIndex = 0;
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

    // Get number of hidden layers
    int numHiddenLayers = mlp->getNumHiddenLayers();
    if (numHiddenLayers == 0) {
        // No hidden layers to visualize
        hiddenLayerScene->addText("No hidden layers in this model");
        return;
    }

    // Update the hidden layer selector if needed
    if (hiddenLayerSelector->count() != numHiddenLayers) {
        hiddenLayerSelector->clear();
        for (int i = 0; i < numHiddenLayers; ++i) {
            hiddenLayerSelector->addItem(QString("Hidden Layer %1").arg(i + 1));
        }
    }

    // Make sure the current index is valid
    if (currentHiddenLayerIndex >= numHiddenLayers) {
        currentHiddenLayerIndex = 0;
        hiddenLayerSelector->setCurrentIndex(currentHiddenLayerIndex);
    }

    // Get the selected hidden layer
    int layerIdx = currentHiddenLayerIndex;
    const Layer& hiddenLayer = mlp->getLayers()[layerIdx];
    const Eigen::VectorXf& activations = hiddenLayer.getLastOutput();

    // Add layer title with larger font and better visibility
    QGraphicsTextItem* layerTitle = hiddenLayerScene->addText(QString("Hidden Layer %1").arg(layerIdx + 1));
    layerTitle->setDefaultTextColor(Qt::white);
    QFont titleFont = layerTitle->font();
    titleFont.setPointSize(14);
    titleFont.setBold(true);
    layerTitle->setFont(titleFont);
    layerTitle->setPos(10, 10);

    // Add layer info
    QGraphicsTextItem* layerInfo = hiddenLayerScene->addText(
        QString("Neurons: %1, Activation: %2")
            .arg(hiddenLayer.getOutputSize())
            .arg(QString::fromStdString(hiddenLayer.getActivationFunction()))
    );
    layerInfo->setDefaultTextColor(Qt::white);
    layerInfo->setPos(10, layerTitle->boundingRect().height() + 20);

    // Determine grid size for this layer
    int hiddenSize = hiddenLayer.getOutputSize();
    int gridSize = static_cast<int>(std::ceil(std::sqrt(hiddenSize)));

    // Calculate cell size - make it larger for better visibility
    int cellSize = 30; // Larger size for better visibility

    // Create a background for the grid
    QGraphicsRectItem* gridBackground = hiddenLayerScene->addRect(
        10,
        layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + 30,
        gridSize * cellSize + 20,
        gridSize * cellSize + 20
    );
    gridBackground->setBrush(QBrush(QColor(60, 60, 60)));
    gridBackground->setPen(QPen(Qt::white));

    // Draw activations as a grid of colored squares
    for (int i = 0; i < hiddenSize; ++i) {
        int row = i / gridSize;
        int col = i % gridSize;

        // Map activation to color intensity (0-255)
        int intensity = static_cast<int>(activations(i) * 255);
        QColor color(intensity, intensity, intensity);

        // Create rectangle
        QGraphicsRectItem* rect = hiddenLayerScene->addRect(
            col * cellSize + 20,
            row * cellSize + layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + 40,
            cellSize - 2,
            cellSize - 2
        );
        rect->setBrush(QBrush(color));
        rect->setPen(QPen(Qt::black));

        // Add tooltip with neuron index and activation value
        QGraphicsTextItem* tooltip = hiddenLayerScene->addText(
            QString("Neuron %1: %2").arg(i).arg(activations(i), 0, 'f', 4)
        );
        tooltip->setVisible(false);
        tooltip->setDefaultTextColor(Qt::white);
        tooltip->setZValue(1); // Ensure it appears above other items

        // Store tooltip in rect's data
        rect->setData(0, QVariant::fromValue(tooltip));
    }

    // Add a legend
    QGraphicsRectItem* legendBackground = hiddenLayerScene->addRect(
        10,
        layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + gridSize * cellSize + 60,
        300,
        80
    );
    legendBackground->setBrush(QBrush(QColor(40, 40, 40)));
    legendBackground->setPen(QPen(Qt::white));

    QGraphicsTextItem* legendTitle = hiddenLayerScene->addText("Activation Legend:");
    legendTitle->setDefaultTextColor(Qt::white);
    legendTitle->setPos(20, layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + gridSize * cellSize + 70);

    // Create a gradient legend
    for (int i = 0; i < 256; ++i) {
        QGraphicsRectItem* legendItem = hiddenLayerScene->addRect(
            20 + i,
            layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + gridSize * cellSize + 100,
            1,
            20
        );
        legendItem->setBrush(QBrush(QColor(i, i, i)));
        legendItem->setPen(Qt::NoPen);
    }

    QGraphicsTextItem* zeroText = hiddenLayerScene->addText("0.0");
    zeroText->setDefaultTextColor(Qt::white);
    zeroText->setPos(20, layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + gridSize * cellSize + 125);

    QGraphicsTextItem* oneText = hiddenLayerScene->addText("1.0");
    oneText->setDefaultTextColor(Qt::white);
    oneText->setPos(270, layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + gridSize * cellSize + 125);

    // Set scene rect
    int sceneWidth = std::max(gridSize * cellSize + 40, 320);
    int sceneHeight = layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + gridSize * cellSize + 150;
    hiddenLayerScene->setSceneRect(0, 0, sceneWidth, sceneHeight);
    ui->gvHiddenLayer->fitInView(hiddenLayerScene->sceneRect(), Qt::KeepAspectRatio);
}

void MainWindow::onHiddenLayerSelectorChanged(int index)
{
    currentHiddenLayerIndex = index;
    updateHiddenLayerVisualization();
}

void MainWindow::updateOutputLayerVisualization()
{
    // Clear scene
    outputLayerScene->clear();

    if (currentImage.isNull()) {
        return;
    }

    // Get output layer (last layer in the network)
    const Layer& outputLayer = mlp->getLayers().back();
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

    // Add network architecture information
    int numHiddenLayers = mlp->getNumHiddenLayers();
    QString architectureInfo = QString("Network Architecture: %1 input → ").arg(mlp->getLayers()[0].getInputSize());

    if (numHiddenLayers > 0) {
        for (int i = 0; i < numHiddenLayers; ++i) {
            architectureInfo += QString("%1 → ").arg(mlp->getLayers()[i].getOutputSize());
        }
    }

    architectureInfo += QString("%1 output").arg(mlp->getLayers().back().getOutputSize());

    QGraphicsTextItem* architectureItem = outputLayerScene->addText(architectureInfo);
    architectureItem->setPos(0, barY + barHeight + 50);

    // Set scene rect
    outputLayerScene->setSceneRect(0, 0, barWidth + 50, barY + barHeight + 100);
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

void MainWindow::setupHiddenLayersUI()
{
    // Create a list widget for hidden layers
    hiddenLayersList = new QListWidget();

    // Create buttons for adding and removing hidden layers
    addHiddenLayerButton = new QPushButton("Add Hidden Layer");
    removeHiddenLayerButton = new QPushButton("Remove Hidden Layer");

    // Create a layout for the buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addWidget(addHiddenLayerButton);
    buttonLayout->addWidget(removeHiddenLayerButton);

    // Create a layout for the hidden layers configuration
    QVBoxLayout* hiddenLayersLayout = new QVBoxLayout();
    hiddenLayersLayout->addWidget(new QLabel("Hidden Layers:"));
    hiddenLayersLayout->addWidget(hiddenLayersList);
    hiddenLayersLayout->addLayout(buttonLayout);

    // Add the layout to the network configuration group box
    QLayout* oldLayout = ui->groupBox_2->layout();
    if (oldLayout) {
        // Remove the old hidden neurons spin box and label
        QLabel* hiddenNeuronsLabel = ui->groupBox_2->findChild<QLabel*>("label_3");
        if (hiddenNeuronsLabel) {
            hiddenNeuronsLabel->hide();
        }

        QSpinBox* hiddenNeuronsSpinBox = ui->groupBox_2->findChild<QSpinBox*>("sbHiddenNeurons");
        if (hiddenNeuronsSpinBox) {
            hiddenNeuronsSpinBox->hide();
        }

        // Add the new layout to the group box
        QGridLayout* gridLayout = qobject_cast<QGridLayout*>(oldLayout);
        if (gridLayout) {
            gridLayout->addLayout(hiddenLayersLayout, 0, 0, 1, 2);
        }
    }

    // Connect signals and slots
    connect(addHiddenLayerButton, &QPushButton::clicked, this, &MainWindow::onAddHiddenLayerClicked);
    connect(removeHiddenLayerButton, &QPushButton::clicked, this, &MainWindow::onRemoveHiddenLayerClicked);

    // Add a default hidden layer with 128 neurons
    hiddenLayerSizes = {128};
    updateHiddenLayersUIFromModel();
}

void MainWindow::updateHiddenLayersUIFromModel()
{
    // Clear the list widget
    hiddenLayersList->clear();

    // Set background color for the list widget to dark gray for better contrast
    hiddenLayersList->setStyleSheet("QListWidget { background-color: #2D2D30; }");

    // Add items for each hidden layer
    for (size_t i = 0; i < hiddenLayerSizes.size(); ++i) {
        QListWidgetItem* item = new QListWidgetItem();
        // Set a minimum height for the item to ensure it's clearly visible
        item->setSizeHint(QSize(item->sizeHint().width(), 40));
        hiddenLayersList->addItem(item);

        // Create a widget for the item
        QWidget* itemWidget = new QWidget();
        itemWidget->setStyleSheet("background-color: #3E3E42;"); // Slightly lighter than the list background
        QHBoxLayout* itemLayout = new QHBoxLayout(itemWidget);

        // Add a label with clear title and white text
        QLabel* label = new QLabel(QString("Hidden Layer %1:").arg(i + 1));
        label->setStyleSheet("color: white; font-weight: bold;");
        itemLayout->addWidget(label);

        // Add a spin box for the number of neurons
        QSpinBox* spinBox = new QSpinBox();
        spinBox->setMinimum(1);
        spinBox->setMaximum(1024);
        spinBox->setValue(hiddenLayerSizes[i]);
        spinBox->setProperty("layerIndex", static_cast<int>(i));
        // Style the spinbox for better visibility
        spinBox->setStyleSheet("background-color: white; color: black;");
        connect(spinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::onHiddenLayerValueChanged);
        itemLayout->addWidget(spinBox);

        // Add a label for "neurons" with white text
        QLabel* neuronsLabel = new QLabel("neurons");
        neuronsLabel->setStyleSheet("color: white;");
        itemLayout->addWidget(neuronsLabel);

        // Set the item widget
        hiddenLayersList->setItemWidget(item, itemWidget);
    }

    // Enable/disable the remove button based on the number of layers
    removeHiddenLayerButton->setEnabled(hiddenLayerSizes.size() > 1);
}

void MainWindow::onAddHiddenLayerClicked()
{
    // Add a new hidden layer with the same number of neurons as the last one
    int newLayerSize = hiddenLayerSizes.empty() ? 128 : hiddenLayerSizes.back();
    hiddenLayerSizes.push_back(newLayerSize);

    // Update the UI
    updateHiddenLayersUIFromModel();
}

void MainWindow::onRemoveHiddenLayerClicked()
{
    // Remove the last hidden layer
    if (!hiddenLayerSizes.empty()) {
        hiddenLayerSizes.pop_back();

        // Update the UI
        updateHiddenLayersUIFromModel();
    }
}

void MainWindow::onHiddenLayerValueChanged(int value)
{
    // Get the layer index from the sender
    QSpinBox* spinBox = qobject_cast<QSpinBox*>(sender());
    if (spinBox) {
        int layerIndex = spinBox->property("layerIndex").toInt();
        if (layerIndex >= 0 && layerIndex < static_cast<int>(hiddenLayerSizes.size())) {
            hiddenLayerSizes[layerIndex] = value;
        }
    }
}

void MainWindow::createMLPFromUIConfig()
{
    // Get the activation function
    QString hiddenActivation = ui->cbHiddenActivation->currentText();

    // Create a new MLP with the configured hidden layers
    delete mlp;
    mlp = new MLP(512 * 512, hiddenLayerSizes, 1, hiddenActivation.toStdString(), "sigmoid");

    // Update worker
    worker->stop();
    delete worker;
    worker = new TrainingWorker(mlp);
    worker->moveToThread(&workerThread);

    // Connect signals and slots
    connect(worker, &TrainingWorker::progressUpdated, this, &MainWindow::onTrainingProgressUpdated);
    connect(worker, &TrainingWorker::epochCompleted, this, &MainWindow::onEpochCompleted);
    connect(worker, &TrainingWorker::trainingComplete, this, &MainWindow::onTrainingComplete);
    connect(worker, &TrainingWorker::evaluationComplete, this, &MainWindow::onEvaluationComplete);
}

void MainWindow::on_btnTrain_clicked()
{
    // Create a new MLP with the current configuration
    createMLPFromUIConfig();

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

    // Clear the loss curve
    lossCurveWidget->clearPlot();

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
                                                   "Binary Model Files (*.senm);;JSON Files (*.json)");

    if (filePath.isEmpty()) {
        return;
    }

    bool success = false;

    // Check which format was selected based on the file extension
    if (filePath.endsWith(".senm", Qt::CaseInsensitive)) {
        // Save model to binary format
        success = mlp->saveToBinary(filePath);

        if (success) {
            QMessageBox::information(this, "Export Successful", "Model exported successfully in binary format.");
        } else {
            QMessageBox::critical(this, "Export Failed", "Failed to write model to binary file.");
        }
    } else {
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
            QMessageBox::information(this, "Export Successful", "Model exported successfully in JSON format.");
            success = true;
        } else {
            QMessageBox::critical(this, "Export Failed", "Failed to write model to JSON file.");
        }
    }
}

void MainWindow::on_btnImportModel_clicked()
{
    QString filePath = QFileDialog::getOpenFileName(this, "Import Model",
                                                   QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),
                                                   "All Supported Files (*.senm *.json);;Binary Model Files (*.senm);;JSON Files (*.json)");

    if (filePath.isEmpty()) {
        return;
    }

    bool success = false;

    // Check file extension to determine format
    if (filePath.endsWith(".senm", Qt::CaseInsensitive)) {
        // Load model from binary format
        success = mlp->loadFromBinary(filePath);

        if (success) {
            QMessageBox::information(this, "Import Successful", "Model imported successfully from binary format.");

            // Update UI with the loaded model's configuration
            hiddenLayerSizes = mlp->getHiddenLayerSizes();
            updateHiddenLayersUIFromModel();

            // Set activation function
            if (mlp->getNumHiddenLayers() > 0) {
                ui->cbHiddenActivation->setCurrentText(QString::fromStdString(mlp->getLayers()[0].getActivationFunction()));
            }

            // Update current image if available
            if (!currentImage.isNull()) {
                updateCurrentImage();
            }
        } else {
            QMessageBox::critical(this, "Import Failed", "Failed to load model from binary file. The file may be corrupted or incompatible.");
        }
    } else {
        // Assume JSON format
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
            QMessageBox::information(this, "Import Successful", "Model imported successfully from JSON format.");

            // Update UI with the loaded model's configuration
            hiddenLayerSizes = mlp->getHiddenLayerSizes();
            updateHiddenLayersUIFromModel();

            // Set activation function
            if (mlp->getNumHiddenLayers() > 0) {
                ui->cbHiddenActivation->setCurrentText(QString::fromStdString(mlp->getLayers()[0].getActivationFunction()));
            }

            // Update current image if available
            if (!currentImage.isNull()) {
                updateCurrentImage();
            }

            success = true;
        } else {
            QMessageBox::critical(this, "Import Failed", "Failed to load model from JSON file. The model architecture may be incompatible.");
        }
    }
}

void MainWindow::onTrainingProgressUpdated(int epoch, int totalEpochs, float loss)
{
    // Update progress bar
    ui->progressBar->setValue(static_cast<int>((static_cast<float>(epoch) / totalEpochs) * 100));

    // Update status bar
    statusBar()->showMessage(QString("Training: Epoch %1/%2, Loss: %3").arg(epoch).arg(totalEpochs).arg(loss, 0, 'f', 6));
}

void MainWindow::onEpochCompleted(int epoch, float loss, float validationLoss)
{
    // Update the loss curve widget
    lossCurveWidget->addDataPoint(epoch, loss, validationLoss);
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
