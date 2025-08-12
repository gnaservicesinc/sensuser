#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "optimizertypes.h"
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
    ui->setupUi(static_cast<QMainWindow*>(this));

    // Initialize backend (libnoodlenet)
    nnBackend = new NoodleNetBackend();

    // Initialize worker thread
    worker = new TrainingWorker(nnBackend);
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
    connect(ui->tabWidget, SIGNAL(currentChanged(int)), this, SLOT(onTabChanged(int)));
    // New Model button
    connect(ui->btnNewModel, &QPushButton::clicked, this, &MainWindow::on_btnNewModel_clicked);
}

MainWindow::~MainWindow()
{
    // Free backend
    delete nnBackend;
    // Stop worker thread
    worker->stop();
    workerThread.quit();
    workerThread.wait();

    // Clean up
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
    inputLayerScene = new QGraphicsScene(static_cast<QObject*>(this));
    hiddenLayerScene = new QGraphicsScene(static_cast<QObject*>(this));
    outputLayerScene = new QGraphicsScene(static_cast<QObject*>(this));

    // Set scenes for graphics views
    ui->gvInputLayer->setScene(inputLayerScene);
    ui->gvHiddenLayer->setScene(hiddenLayerScene);
    ui->gvOutputLayer->setScene(outputLayerScene);

    // Initialize UI elements
    ui->lblPositiveDir->setText("Not set");
    ui->lblNegativeDir->setText("Not set");
    if (ui->lblValidationDir) ui->lblValidationDir->setText("Not set");
    ui->lblCurrentImage->setText("No image loaded");
    ui->lblPrediction->setText("No prediction");
    ui->lblAccuracy->setText("No evaluation");

    // Set default values for training parameters
    ui->cbHiddenActivation->addItem("sigmoid");
    ui->cbHiddenActivation->addItem("relu");
    ui->cbHiddenActivation->addItem("tanh");
    ui->cbHiddenActivation->addItem("leaky_relu");

    // Set tooltips for activation functions
    ui->cbHiddenActivation->setItemData(0, "Classic 'S' shaped curve. Good for binary classification outputs.", Qt::ToolTipRole);
    ui->cbHiddenActivation->setItemData(1, "Outputs the input if positive, otherwise zero. Computationally efficient and helps mitigate vanishing gradients.", Qt::ToolTipRole);
    ui->cbHiddenActivation->setItemData(2, "Similar to Sigmoid but zero-centered (-1 to 1). Often preferred over Sigmoid in hidden layers.", Qt::ToolTipRole);
    ui->cbHiddenActivation->setItemData(3, "A variant of ReLU that allows a small, non-zero gradient when the unit is not active, preventing 'dying ReLU' problems.", Qt::ToolTipRole);

    // Set tooltips for optimizer options
    ui->cbOptimizer->setItemData(0, "Standard Stochastic Gradient Descent. The simplest optimizer, it updates weights based on the current gradient and learning rate. Can be slow to converge.", Qt::ToolTipRole);
    ui->cbOptimizer->setItemData(1, "Root Mean Square Propagation. Adapts the learning rate for each weight by dividing by a moving average of recent squared gradients. Good for non-stationary problems.", Qt::ToolTipRole);
    ui->cbOptimizer->setItemData(2, "Adaptive Moment Estimation. Combines the ideas of RMSprop and momentum. It uses moving averages of both the gradient and its squared value to adapt the learning rate for each weight. Often converges fastest.", Qt::ToolTipRole);

    ui->sbLearningRate->setValue(0.01);
    ui->sbEpochs->setValue(100);
    ui->sbBatchSize->setValue(10);
    ui->cbShuffle->setChecked(true);
    ui->cbOptimizer->setCurrentIndex(0); // Default to SGD
    ui->sbBias->setValue(0.0);

    // Setup hidden layers configuration UI
    setupHiddenLayersUI();

    // Create and set up the hidden layer selector for visualization
    hiddenLayerSelector = new QComboBox();
    hiddenLayerSelector->addItem("Hidden Layer 1");
    hiddenLayerSelector->setStyleSheet("background-color: white; color: black; font-weight: bold;");
    connect(hiddenLayerSelector, SIGNAL(currentIndexChanged(int)),
            this, SLOT(onHiddenLayerSelectorChanged(int)));

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

    // Visualization panel (mirrors Bookish options)
    QGroupBox* visGroup = new QGroupBox("Visualization Options", hiddenLayerTab);
    visGroup->setStyleSheet("QGroupBox { color: white; font-weight: bold; } QLabel { color: white; }");
    QGridLayout* visLayout = new QGridLayout(visGroup);
    QLabel* modeLabel = new QLabel("Mode:"); visModeCombo = new QComboBox(visGroup); visModeCombo->addItem("weights"); visModeCombo->addItem("heatmap");
    QLabel* scaleLabel = new QLabel("Scaling:"); visScaleCombo = new QComboBox(visGroup); visScaleCombo->addItem("minmax"); visScaleCombo->addItem("symmetric zero");
    visBiasCheck = new QCheckBox("Include biases", visGroup);
    visStatsCheck = new QCheckBox("Include stats", visGroup);
    visRawCheck = new QCheckBox("Raw weights (non-square)", visGroup);
    QPushButton* exportBtn = new QPushButton("Export Visualizations…", visGroup);
    connect(exportBtn, &QPushButton::clicked, this, &MainWindow::onExportVisualizationsClicked);
    connect(visModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int){ updateHiddenLayerVisualization(); });
    connect(visScaleCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int){ updateHiddenLayerVisualization(); });
    connect(visRawCheck, &QCheckBox::toggled, this, [this](bool){ updateHiddenLayerVisualization(); });

    int r=0; visLayout->addWidget(modeLabel, r,0); visLayout->addWidget(visModeCombo, r,1); r++;
    visLayout->addWidget(scaleLabel, r,0); visLayout->addWidget(visScaleCombo, r,1); r++;
    visLayout->addWidget(visBiasCheck, r,0,1,2); r++;
    visLayout->addWidget(visStatsCheck, r,0,1,2); r++;
    visLayout->addWidget(visRawCheck, r,0,1,2); r++;
    visLayout->addWidget(exportBtn, r,0,1,2);
    hiddenLayerLayout->insertWidget(1, visGroup);

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
    modelLocked = false;
    trainedSinceLastSave = false;
}

void MainWindow::onExportVisualizationsClicked() {
    if (!nnBackend || !nnBackend->hasModel()) {
        QMessageBox::warning(this, "Export", "No model loaded.");
        return;
    }
    QString outDir = QFileDialog::getExistingDirectory(this, "Select Output Directory");
    if (outDir.isEmpty()) return;
    NN_VisMode mode = (visModeCombo && visModeCombo->currentIndex()==1) ? NN_VIS_MODE_HEATMAP : NN_VIS_MODE_WEIGHTS;
    NN_VisScale scale = (visScaleCombo && visScaleCombo->currentIndex()==1) ? NN_VIS_SCALE_SYM_ZERO : NN_VIS_SCALE_MINMAX;
    bool includeBias = visBiasCheck && visBiasCheck->isChecked();
    bool includeStats = visStatsCheck && visStatsCheck->isChecked();
    bool raw = visRawCheck && visRawCheck->isChecked();
    int sel = currentHiddenLayerIndex; if (sel < 0) sel = 0; if (sel >= nnBackend->numHidden()) sel = nnBackend->numHidden()-1;
    bool ok = nnBackend->exportVisualizations(outDir, mode, scale, includeBias, includeStats, raw, sel);
    if (ok) QMessageBox::information(this, "Export", "Visualizations exported.");
    else QMessageBox::critical(this, "Export", "Export failed.");
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
    try {
        if (nnBackend && nnBackend->hasModel()) {
            float prediction = nnBackend->predict(currentImage);
            QString predictionText = QString("Prediction: %1 (Threshold: 0.5)")
                                    .arg(prediction, 0, 'f', 4);
            ui->lblPrediction->setText(predictionText);
        } else {
            ui->lblPrediction->setText("No model available");
        }
    } catch (const std::exception& e) {
        // Handle any exceptions that might occur during prediction
        qWarning() << "Exception in prediction:" << e.what();
        ui->lblPrediction->setText("Error in prediction");
    } catch (...) {
        // Catch any other exceptions
        qWarning() << "Unknown exception in prediction";
        ui->lblPrediction->setText("Error in prediction");
    }

    // Update layer visualizations
    updateLayerVisualizations();
}

void MainWindow::updateLayerVisualizations()
{
    try {
        if (!currentImage.isNull()) {
            updateInputLayerVisualization();
            updateHiddenLayerVisualization();
            updateOutputLayerVisualization();
        }
    } catch (const std::exception& e) {
        // Handle any exceptions that might occur during visualization
        qWarning() << "Exception in updateLayerVisualizations:" << e.what();
    } catch (...) {
        // Catch any other exceptions
        qWarning() << "Unknown exception in updateLayerVisualizations";
    }
}

void MainWindow::updateInputLayerVisualization()
{
    try {
        // Clear scene
        if (inputLayerScene) {
            inputLayerScene->clear();
        } else {
            return; // Safety check
        }

        if (currentImage.isNull()) {
            return;
        }

        // Add current image to scene
        QImage processedImage = currentImage;
        if (processedImage.format() != QImage::Format_Grayscale8) {
            processedImage = processedImage.convertToFormat(QImage::Format_Grayscale8);
        }

        if (processedImage.width() != 512 || processedImage.height() != 512) {
            processedImage = processedImage.scaled(512, 512, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
        }

        QGraphicsPixmapItem* pixmapItem = inputLayerScene->addPixmap(QPixmap::fromImage(processedImage));
        if (pixmapItem) {
            inputLayerScene->setSceneRect(pixmapItem->boundingRect());
            if (ui && ui->gvInputLayer) {
                ui->gvInputLayer->fitInView(inputLayerScene->sceneRect(), Qt::KeepAspectRatio);
            }
        }
    } catch (const std::exception& e) {
        // Handle any exceptions that might occur during visualization
        qWarning() << "Exception in updateInputLayerVisualization:" << e.what();
    } catch (...) {
        // Catch any other exceptions
        qWarning() << "Unknown exception in updateInputLayerVisualization";
    }
}

void MainWindow::updateHiddenLayerVisualization()
{
    // Clear scene
    if (hiddenLayerScene) {
        hiddenLayerScene->clear();
    } else {
        return; // Safety check
    }

    if (currentImage.isNull() || !nnBackend || !nnBackend->hasModel()) {
        return;
    }
    try {
        // Get number of hidden layers from backend
        int numHiddenLayers = nnBackend->numHidden();
        if (numHiddenLayers == 0) {
            // No hidden layers to visualize
            hiddenLayerScene->addText("No hidden layers in this model");
            return;
        }

        // Update the hidden layer selector if it exists
        if (hiddenLayerSelector) {
            // Block signals temporarily to prevent recursive calls
            hiddenLayerSelector->blockSignals(true);

            hiddenLayerSelector->clear();
            for (int i = 0; i < numHiddenLayers; ++i) {
                hiddenLayerSelector->addItem(QString("Hidden Layer %1").arg(i + 1));
            }

            // Make sure the current index is valid
            if (currentHiddenLayerIndex >= numHiddenLayers) {
                currentHiddenLayerIndex = 0;
            }

            if (hiddenLayerSelector->count() > 0) {
                hiddenLayerSelector->setCurrentIndex(currentHiddenLayerIndex);
            }

            hiddenLayerSelector->blockSignals(false);
        }
    } catch (const std::exception& e) {
        // Handle any exceptions that might occur during processing
        qWarning() << "Exception in updateHiddenLayerVisualization:" << e.what();
        return;
    } catch (...) {
        // Catch any other exceptions
        qWarning() << "Unknown exception in updateHiddenLayerVisualization";
        return;
    }

    try {
        // Safety checks
        if (currentHiddenLayerIndex < 0 || !hiddenLayerScene) return;
        int layerIdx = currentHiddenLayerIndex;
        if (layerIdx < 0 || layerIdx >= nnBackend->numHidden()) { hiddenLayerScene->addText("Invalid layer index"); return; }

        // Determine visualization options from UI
        NN_VisMode mode = (visModeCombo && visModeCombo->currentIndex()==1) ? NN_VIS_MODE_HEATMAP : NN_VIS_MODE_WEIGHTS;
        NN_VisScale scale = (visScaleCombo && visScaleCombo->currentIndex()==1) ? NN_VIS_SCALE_SYM_ZERO : NN_VIS_SCALE_MINMAX;
        bool raw = visRawCheck && visRawCheck->isChecked();

        // Render via backend/libnoodlenet
        QImage img;
        if (!nnBackend->renderHiddenLayerVisualization(layerIdx, mode, scale, raw, img)) {
            hiddenLayerScene->addText("Visualization not available");
            return;
        }

        // Clear scene and draw image with a simple title/info
        hiddenLayerScene->clear();
        QGraphicsTextItem* layerTitle = hiddenLayerScene->addText(QString("Hidden Layer %1").arg(layerIdx + 1));
        layerTitle->setDefaultTextColor(Qt::white);
        QFont titleFont = layerTitle->font(); titleFont.setPointSize(14); titleFont.setBold(true); layerTitle->setFont(titleFont); layerTitle->setPos(10, 10);
        int inDim=0,outDim=0; nnBackend->layerDims(layerIdx, inDim, outDim);
        auto actEnum = nnBackend->hiddenActivation(layerIdx);
        const char* actName = (actEnum==NN_ACTIVATION_FUNCTION_TANH?"tanh": actEnum==NN_ACTIVATION_FUNCTION_RELU?"relu": actEnum==NN_ACTIVATION_FUNCTION_LEAKY_RELU?"leaky_relu":"sigmoid");
        QGraphicsTextItem* layerInfo = hiddenLayerScene->addText(QString("Neurons: %1, Activation: %2").arg(outDim).arg(actName));
        layerInfo->setDefaultTextColor(Qt::white);
        layerInfo->setPos(10, layerTitle->boundingRect().height() + 20);
        QGraphicsPixmapItem* pix = hiddenLayerScene->addPixmap(QPixmap::fromImage(img.scaled(512, 512, Qt::KeepAspectRatio, Qt::SmoothTransformation)));
        pix->setPos(10, layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + 30);
        hiddenLayerScene->setSceneRect(hiddenLayerScene->itemsBoundingRect());
        if (ui && ui->gvHiddenLayer) ui->gvHiddenLayer->fitInView(hiddenLayerScene->sceneRect(), Qt::KeepAspectRatio);
    } catch (const std::exception& e) {
        qWarning() << "Exception in updateHiddenLayerVisualization (rendering):" << e.what();
        return;
    } catch (...) {
        qWarning() << "Unknown exception in updateHiddenLayerVisualization (rendering)";
        return;
    }

    try {
        // Safety checks
        if (!hiddenLayerScene) {
            return;
        }

        // Get references to the title and info items we created earlier
        QList<QGraphicsItem*> items = hiddenLayerScene->items();
        QGraphicsTextItem* layerTitle = nullptr;
        QGraphicsTextItem* layerInfo = nullptr;

        for (QGraphicsItem* item : items) {
            QGraphicsTextItem* textItem = dynamic_cast<QGraphicsTextItem*>(item);
            if (textItem) {
                QString text = textItem->toPlainText();
                if (text.startsWith("Hidden Layer")) {
                    layerTitle = textItem;
                } else if (text.startsWith("Neurons:")) {
                    layerInfo = textItem;
                }

                if (layerTitle && layerInfo) {
                    break;
                }
            }
        }

        if (!layerTitle || !layerInfo) {
            return; // Can't proceed without these items
        }

        // Get the grid size from earlier
        int hiddenSize = 0;
        int gridSize = 0;

        if (currentHiddenLayerIndex >= 0) {
            int inDim2=0,outDim2=0; nnBackend->layerDims(currentHiddenLayerIndex, inDim2, outDim2);
            hiddenSize = outDim2;
            gridSize = static_cast<int>(std::ceil(std::sqrt(hiddenSize)));
        } else {
            return; // Can't proceed without valid layer info
        }

        // Calculate cell size - make it larger for better visibility
        int cellSize = 30; // Larger size for better visibility

        // Add a legend (0..255 grayscale) for reference
        QGraphicsRectItem* legendBackground = hiddenLayerScene->addRect(
            10,
            layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + 512 + 60,
            300,
            80
        );
        legendBackground->setBrush(QBrush(QColor(40, 40, 40)));
        legendBackground->setPen(QPen(Qt::white));

        QGraphicsTextItem* legendTitle = hiddenLayerScene->addText("Activation Legend:");
        legendTitle->setDefaultTextColor(Qt::white);
        legendTitle->setPos(20, layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + 512 + 70);

        // Create a gradient legend
        for (int i = 0; i < 256; ++i) {
            QGraphicsRectItem* legendItem = hiddenLayerScene->addRect(
                20 + i,
                layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + 512 + 100,
                1,
                20
            );
            legendItem->setBrush(QBrush(QColor(i, i, i)));
            legendItem->setPen(Qt::NoPen);
        }

        QGraphicsTextItem* zeroText = hiddenLayerScene->addText("0.0");
        zeroText->setDefaultTextColor(Qt::white);
        zeroText->setPos(20, layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + 512 + 125);

        QGraphicsTextItem* oneText = hiddenLayerScene->addText("1.0");
        oneText->setDefaultTextColor(Qt::white);
        oneText->setPos(270, layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + 512 + 125);

        // Set scene rect
        int sceneWidth = std::max(512 + 40, 320);
        int sceneHeight = layerTitle->boundingRect().height() + layerInfo->boundingRect().height() + 512 + 150;
        hiddenLayerScene->setSceneRect(0, 0, sceneWidth, sceneHeight);

        if (ui && ui->gvHiddenLayer) {
            ui->gvHiddenLayer->fitInView(hiddenLayerScene->sceneRect(), Qt::KeepAspectRatio);
        }
    } catch (const std::exception& e) {
        // Handle any exceptions that might occur during legend creation
        qWarning() << "Exception in updateHiddenLayerVisualization (legend):" << e.what();
    } catch (...) {
        // Catch any other exceptions
        qWarning() << "Unknown exception in updateHiddenLayerVisualization (legend)";
    }
}

void MainWindow::onHiddenLayerSelectorChanged(int index)
{
    currentHiddenLayerIndex = index;
    updateHiddenLayerVisualization();
}

void MainWindow::updateOutputLayerVisualization()
{
    try {
        // Clear scene
        if (outputLayerScene) {
            outputLayerScene->clear();
        } else {
            return; // Safety check
        }

    if (currentImage.isNull() || !nnBackend || !nnBackend->hasModel()) {
        return;
    }

        // Compute output activations and raw pre-activation (z) via backend
        int L = nnBackend->numWeightLayers();
        std::vector<float> out; std::vector<float> z;
        if (L < 0 || !nnBackend->computeActivations(currentImage, L, out) || out.empty() ||
            !nnBackend->computePreActivations(currentImage, L, z) || z.empty()) {
            outputLayerScene->addText("No output available");
            return;
        }

        // Create text items
        QString rawOutput = QString("Raw output (z): %1").arg(z[0], 0, 'f', 4);
        QString activatedOutput = QString("Output probability: %1").arg(out[0], 0, 'f', 4);
        QString thresholdInfo = QString("Classification threshold: 0.5");
        QString classification = QString("Classification: %1").arg(out[0] >= 0.5 ? "Object Detected" : "Object Not Detected");

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

        // Output value bar - clamp output to [0, 1] range
        float outputValue = std::max(0.0f, std::min(1.0f, out[0]));
        int outputBarWidth = static_cast<int>(outputValue * barWidth);
        QGraphicsRectItem* outputBar = outputLayerScene->addRect(0, barY, outputBarWidth, barHeight);
        outputBar->setBrush(QBrush(outputValue >= 0.5 ? Qt::green : Qt::red));

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
        int numHiddenLayers = nnBackend->numHidden();
        QString architectureInfo = QString("Network Architecture: %1 input → ").arg(nnBackend->inputSize());
        for (int i = 0; i < numHiddenLayers; ++i) {
            architectureInfo += QString("%1 → ").arg(nnBackend->hiddenSize(i));
        }
        architectureInfo += QString("%1 output").arg(nnBackend->outputSize());

        QGraphicsTextItem* architectureItem = outputLayerScene->addText(architectureInfo);
        architectureItem->setPos(0, barY + barHeight + 50);

        // Set scene rect
        outputLayerScene->setSceneRect(0, 0, barWidth + 50, barY + barHeight + 100);

        if (ui && ui->gvOutputLayer) {
            ui->gvOutputLayer->fitInView(outputLayerScene->sceneRect(), Qt::KeepAspectRatio);
        }
    } catch (const std::exception& e) {
        // Handle any exceptions that might occur during visualization
        qWarning() << "Exception in updateOutputLayerVisualization:" << e.what();
    } catch (...) {
        // Catch any other exceptions
        qWarning() << "Unknown exception in updateOutputLayerVisualization";
    }
}

void MainWindow::on_btnLoadPositive_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(static_cast<QWidget*>(this), "Select Directory with Positive Examples",
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
    QString dir = QFileDialog::getExistingDirectory(static_cast<QWidget*>(this), "Select Directory with Negative Examples",
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
    connect(addHiddenLayerButton, SIGNAL(clicked()), this, SLOT(onAddHiddenLayerClicked()));
    connect(removeHiddenLayerButton, SIGNAL(clicked()), this, SLOT(onRemoveHiddenLayerClicked()));

    // Add a default hidden layer with 128 neurons and sigmoid activation
    hiddenLayerSizes = {128};
    hiddenLayerActivations = {"sigmoid"};
    updateHiddenLayersUIFromModel();
}

void MainWindow::updateHiddenLayersUIFromModel()
{
    // Ensure hiddenLayerActivations vector is properly sized
    while (hiddenLayerActivations.size() < hiddenLayerSizes.size()) {
        hiddenLayerActivations.push_back("sigmoid"); // Default activation
    }
    while (hiddenLayerActivations.size() > hiddenLayerSizes.size()) {
        hiddenLayerActivations.pop_back();
    }

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
        itemLayout->setContentsMargins(10, 5, 10, 5); // Add some padding

        // Add a label with clear title and white text
        QLabel* label = new QLabel(QString("Hidden Layer %1:").arg(i + 1));
        label->setStyleSheet("color: white; font-weight: bold;");
        label->setMinimumWidth(100); // Ensure the label has enough width to display the text
        itemLayout->addWidget(label);

        // Add a spin box for the number of neurons
        QSpinBox* spinBox = new QSpinBox();
        spinBox->setMinimum(1);
        spinBox->setMaximum(1024);
        spinBox->setValue(hiddenLayerSizes[i]);
        spinBox->setProperty("layerIndex", static_cast<int>(i));
        // Style the spinbox for better visibility
        spinBox->setStyleSheet("background-color: white; color: black;");
        connect(spinBox, SIGNAL(valueChanged(int)), this, SLOT(onHiddenLayerValueChanged(int)));
        itemLayout->addWidget(spinBox);

        // Add a label for "neurons" with white text
        QLabel* neuronsLabel = new QLabel("neurons");
        neuronsLabel->setStyleSheet("color: white;");
        itemLayout->addWidget(neuronsLabel);

        // Add activation function combo box
        QComboBox* activationCombo = new QComboBox();
        activationCombo->addItem("sigmoid");
        activationCombo->addItem("relu");
        activationCombo->addItem("tanh");
        activationCombo->addItem("leaky_relu");

        // Set the current activation function for this layer
        if (i < hiddenLayerActivations.size()) {
            activationCombo->setCurrentText(QString::fromStdString(hiddenLayerActivations[i]));
        } else {
            activationCombo->setCurrentText("sigmoid"); // Default
        }

        activationCombo->setProperty("layerIndex", static_cast<int>(i));
        activationCombo->setStyleSheet("background-color: white; color: black;");
        connect(activationCombo, SIGNAL(currentTextChanged(QString)), this, SLOT(onHiddenLayerActivationChanged(QString)));
        itemLayout->addWidget(activationCombo);

        // Set the item widget
        hiddenLayersList->setItemWidget(item, itemWidget);
    }

    // Enable/disable the remove button based on the number of layers
    removeHiddenLayerButton->setEnabled(hiddenLayerSizes.size() > 1);
}

void MainWindow::onAddHiddenLayerClicked()
{
    if (modelLocked) { QMessageBox::warning(this, "Locked", "Network configuration is locked after loading or starting training."); return; }
    // Add a new hidden layer with the same number of neurons as the last one
    int newLayerSize = hiddenLayerSizes.empty() ? 128 : hiddenLayerSizes.back();
    hiddenLayerSizes.push_back(newLayerSize);

    // Add default activation function for the new layer
    std::string newActivation = hiddenLayerActivations.empty() ? "sigmoid" : hiddenLayerActivations.back();
    hiddenLayerActivations.push_back(newActivation);

    // Update the UI
    updateHiddenLayersUIFromModel();

    // Update the hidden layer selector in the visualization tab
    if (hiddenLayerSelector) {
        hiddenLayerSelector->clear();
        for (size_t i = 0; i < hiddenLayerSizes.size(); ++i) {
            hiddenLayerSelector->addItem(QString("Hidden Layer %1").arg(i + 1));
        }
    }
}

void MainWindow::onRemoveHiddenLayerClicked()
{
    if (modelLocked) { QMessageBox::warning(this, "Locked", "Network configuration is locked after loading or starting training."); return; }
    // Remove the last hidden layer
    if (!hiddenLayerSizes.empty()) {
        hiddenLayerSizes.pop_back();

        // Also remove the corresponding activation function
        if (!hiddenLayerActivations.empty()) {
            hiddenLayerActivations.pop_back();
        }

        // Update the UI
        updateHiddenLayersUIFromModel();

        // Update the hidden layer selector in the visualization tab
        if (hiddenLayerSelector) {
            hiddenLayerSelector->clear();
            for (size_t i = 0; i < hiddenLayerSizes.size(); ++i) {
                hiddenLayerSelector->addItem(QString("Hidden Layer %1").arg(i + 1));
            }
        }
    }
}

void MainWindow::onHiddenLayerValueChanged(int value)
{
    if (modelLocked) { QMessageBox::warning(this, "Locked", "Network configuration is locked after loading or starting training."); return; }
    // Get the layer index from the sender
    QSpinBox* spinBox = qobject_cast<QSpinBox*>(static_cast<QObject*>(sender()));
    if (spinBox) {
        int layerIndex = spinBox->property("layerIndex").toInt();
        if (layerIndex >= 0 && layerIndex < static_cast<int>(hiddenLayerSizes.size())) {
            hiddenLayerSizes[layerIndex] = value;
        }
    }
}

void MainWindow::onHiddenLayerActivationChanged(const QString& activation)
{
    if (modelLocked) { QMessageBox::warning(this, "Locked", "Network configuration is locked after loading or starting training."); return; }
    // Get the layer index from the sender
    QComboBox* comboBox = qobject_cast<QComboBox*>(static_cast<QObject*>(sender()));
    if (comboBox) {
        int layerIndex = comboBox->property("layerIndex").toInt();
        if (layerIndex >= 0 && layerIndex < static_cast<int>(hiddenLayerActivations.size())) {
            hiddenLayerActivations[layerIndex] = activation.toStdString();
        }
    }
}

void MainWindow::createModelFromUIConfig()
{
    // Ensure we have activation functions for all layers
    while (hiddenLayerActivations.size() < hiddenLayerSizes.size()) {
        hiddenLayerActivations.push_back("sigmoid"); // Default for missing activations
    }

    // Create libnoodlenet model for prediction/IO
    if (nnBackend) {
        nnBackend->createModel(512 * 512, hiddenLayerSizes, hiddenLayerActivations, NN_ACTIVATION_FUNCTION_SIGMOID);
    }
}

void MainWindow::on_btnTrain_clicked()
{
    // If no model yet, create from current configuration; else refuse edits
    if (!nnBackend || !nnBackend->hasModel()) {
        createModelFromUIConfig();
    } else if (!modelLocked) {
        // Lock if user attempts to change config while a model exists
        setModelLocked(true);
    }

    // Set training parameters
    worker->setPositiveDir(positiveDir);
    worker->setNegativeDir(negativeDir);
    worker->setLearningRate(ui->sbLearningRate->value());
    worker->setEpochs(ui->sbEpochs->value());
    worker->setBatchSize(ui->sbBatchSize->value());
    worker->setShuffle(ui->cbShuffle->isChecked());
    worker->setValidationDir(validationDir);

    // Set optimizer type based on combo box selection
    OptimizerType selectedOptimizer = static_cast<OptimizerType>(ui->cbOptimizer->currentIndex());
    worker->setOptimizer(selectedOptimizer);

    // Persist training dirs and locked training params into model metadata
    if (nnBackend && nnBackend->hasModel()) {
        nnBackend->setDataDirs(positiveDir, negativeDir, validationDir);
        nnBackend->setLockedTrainingParams(ui->sbBatchSize->value(), ui->sbLearningRate->value(), ui->cbShuffle->isChecked(), selectedOptimizer);
        setModelLocked(true);
        trainedSinceLastSave = true;
    }

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

    if (filePath.endsWith(".senm", Qt::CaseInsensitive)) {
        // Ensure metadata is stored before save
        if (nnBackend && nnBackend->hasModel()) {
            nnBackend->setDataDirs(positiveDir, negativeDir, validationDir);
            // If we have locked params (after first train or loaded), persist them too
            nnBackend->setLockedTrainingParams(ui->sbBatchSize->value(), ui->sbLearningRate->value(), ui->cbShuffle->isChecked(), static_cast<OptimizerType>(ui->cbOptimizer->currentIndex()));
        }
        bool success = nnBackend && nnBackend->hasModel() && nnBackend->saveModel(filePath);
        if (success) QMessageBox::information(this, "Export Successful", "Model exported successfully in binary format.");
        else QMessageBox::critical(this, "Export Failed", "Failed to write model to binary file.");
        if (success) trainedSinceLastSave = false;
    } else {
        QMessageBox::critical(this, "Export Failed", "Only binary .senm export is supported.");
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

    if (filePath.endsWith(".senm", Qt::CaseInsensitive)) {
        bool loaded = nnBackend && nnBackend->loadModel(filePath);
        if (loaded) {
            // Update UI with the loaded model's configuration
            hiddenLayerSizes.clear();
            hiddenLayerActivations.clear();
            int H = nnBackend->numHidden();
            for (int i = 0; i < H; ++i) {
                hiddenLayerSizes.push_back(nnBackend->hiddenSize(i));
                ActivationFunction af = nnBackend->hiddenActivation(i);
                std::string name = (af == NN_ACTIVATION_FUNCTION_TANH) ? "tanh" :
                                   (af == NN_ACTIVATION_FUNCTION_RELU) ? "relu" :
                                   (af == NN_ACTIVATION_FUNCTION_LEAKY_RELU) ? "leaky_relu" : "sigmoid";
                hiddenLayerActivations.push_back(name);
            }
            updateHiddenLayersUIFromModel();
            if (hiddenLayerSelector) {
                hiddenLayerSelector->clear();
                for (size_t i = 0; i < hiddenLayerSizes.size(); ++i) {
                    hiddenLayerSelector->addItem(QString("Hidden Layer %1").arg(i + 1));
                }
            }
            QMessageBox::information(this, "Import Successful", "Model imported successfully from binary format.");
            if (!currentImage.isNull()) updateCurrentImage();
            // Load any saved data dirs and update UI labels
            QString p, n, v; if (nnBackend->getDataDirs(p, n, v)) {
                if (!p.isEmpty()) { positiveDir = p; ui->lblPositiveDir->setText(p); loadImagesFromDir(p, positiveImages); }
                if (!n.isEmpty()) { negativeDir = n; ui->lblNegativeDir->setText(n); loadImagesFromDir(n, negativeImages); }
                if (!v.isEmpty()) { validationDir = v; ui->lblValidationDir->setText(v); }
                ui->btnTrain->setEnabled(!positiveDir.isEmpty());
                ui->btnEvaluate->setEnabled(!positiveDir.isEmpty());
            }
            // Apply locked training params if present
            applyLockedParamsFromModel();
            setModelLocked(true);
            trainedSinceLastSave = false;
        } else {
            QMessageBox::critical(this, "Import Failed", "Failed to load model from binary file.");
        }
    } else {
        QMessageBox::critical(this, "Import Failed", "Only binary .senm models are supported.");
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
    trainedSinceLastSave = true;

    // Update current image if available
    if (!currentImage.isNull()) {
        updateCurrentImage();
    }
}

void MainWindow::setModelLocked(bool locked) {
    modelLocked = locked;
    // Disable network configuration controls when locked
    if (addHiddenLayerButton) addHiddenLayerButton->setEnabled(!locked);
    if (removeHiddenLayerButton) removeHiddenLayerButton->setEnabled(!locked);
    if (hiddenLayersList) {
        for (int i = 0; i < hiddenLayersList->count(); ++i) {
            QWidget* w = hiddenLayersList->itemWidget(hiddenLayersList->item(i));
            if (w) w->setEnabled(!locked);
        }
    }
    // Also disable simple config widgets if present
    if (ui->sbHiddenNeurons) ui->sbHiddenNeurons->setEnabled(!locked);
    if (ui->cbHiddenActivation) ui->cbHiddenActivation->setEnabled(!locked);
    if (ui->sbBias) ui->sbBias->setEnabled(!locked);
    // Disable training params that must remain fixed across additional steps
    if (ui->sbLearningRate) ui->sbLearningRate->setEnabled(!locked);
    if (ui->sbBatchSize) ui->sbBatchSize->setEnabled(!locked);
    if (ui->cbShuffle) ui->cbShuffle->setEnabled(!locked);
    if (ui->cbOptimizer) ui->cbOptimizer->setEnabled(!locked);
    // Keep epochs editable for additional training steps
}

void MainWindow::applyLockedParamsFromModel() {
    if (!nnBackend || !nnBackend->hasModel()) return;
    int bs=0; float lr=0.0f; bool sh=false; OptimizerType opt=OptimizerType::SGD;
    if (nnBackend->getLockedTrainingParams(bs, lr, sh, opt)) {
        if (ui->sbBatchSize) ui->sbBatchSize->setValue(bs);
        if (ui->sbLearningRate) ui->sbLearningRate->setValue(lr);
        if (ui->cbShuffle) ui->cbShuffle->setChecked(sh);
        if (ui->cbOptimizer) ui->cbOptimizer->setCurrentIndex(static_cast<int>(opt));
    }
}

void MainWindow::on_btnNewModel_clicked() {
    if (!nnBackend) return;
    if (nnBackend->hasModel() && trainedSinceLastSave) {
        auto ret = QMessageBox::question(this, "Start New Model",
                                         "You have unsaved training progress. Do you want to export the current model before starting a new one?",
                                         QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel,
                                         QMessageBox::Yes);
        if (ret == QMessageBox::Cancel) return;
        if (ret == QMessageBox::Yes) {
            on_btnExportModel_clicked();
            if (trainedSinceLastSave) {
                // export failed or canceled
                return;
            }
        }
    }
    // Drop current model
    nnBackend->resetModel();
    setModelLocked(false);
    trainedSinceLastSave = false;
    // Clear data dirs and UI labels
    positiveDir.clear(); negativeDir.clear(); validationDir.clear();
    positiveImages.clear(); negativeImages.clear();
    ui->lblPositiveDir->setText("Not set");
    ui->lblNegativeDir->setText("Not set");
    if (ui->lblValidationDir) ui->lblValidationDir->setText("Not set");
    ui->btnTrain->setEnabled(false);
    ui->btnEvaluate->setEnabled(false);
    // Reset hidden layers to one default layer
    hiddenLayerSizes.clear(); hiddenLayerActivations.clear();
    hiddenLayerSizes.push_back(128); hiddenLayerActivations.push_back("sigmoid");
    updateHiddenLayersUIFromModel();
    if (hiddenLayerSelector) {
        hiddenLayerSelector->clear(); hiddenLayerSelector->addItem("Hidden Layer 1");
        currentHiddenLayerIndex = 0;
    }
    // Clear scenes
    inputLayerScene->clear(); hiddenLayerScene->clear(); outputLayerScene->clear();
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
        try {
            updateLayerVisualizations();
        } catch (const std::exception& e) {
            qWarning() << "Exception during tab change visualization update:" << e.what();
        } catch (...) {
            qWarning() << "Unknown exception during tab change visualization update";
        }
    }
}

void MainWindow::on_btnLoadValidation_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(static_cast<QWidget*>(this), "Select Directory with Validation Examples",
                                                   QStandardPaths::writableLocation(QStandardPaths::PicturesLocation),
                                                   QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (dir.isEmpty()) return;
    validationDir = dir;
    if (ui->lblValidationDir) ui->lblValidationDir->setText(dir);
}
