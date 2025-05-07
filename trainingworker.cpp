#include "trainingworker.h"
#include <QDirIterator>
#include <QFileInfo>
#include <QImageReader>
#include <QDebug>
#include <algorithm>
#include <random>

TrainingWorker::TrainingWorker(MLP* mlp, QObject* parent)
    : QObject(parent), mlp(mlp), learningRate(0.01f), epochs(100), batchSize(10), shuffle(true), stopRequested(false)
{
}

void TrainingWorker::setPositiveDir(const QString& dir)
{
    QMutexLocker locker(&mutex);
    positiveDir = dir;
}

void TrainingWorker::setNegativeDir(const QString& dir)
{
    QMutexLocker locker(&mutex);
    negativeDir = dir;
}

void TrainingWorker::setLearningRate(float rate)
{
    QMutexLocker locker(&mutex);
    learningRate = rate;
}

void TrainingWorker::setEpochs(int epochs)
{
    QMutexLocker locker(&mutex);
    this->epochs = epochs;
}

void TrainingWorker::setBatchSize(int size)
{
    QMutexLocker locker(&mutex);
    batchSize = size;
}

void TrainingWorker::setShuffle(bool shuffle)
{
    QMutexLocker locker(&mutex);
    this->shuffle = shuffle;
}

void TrainingWorker::stop()
{
    QMutexLocker locker(&mutex);
    stopRequested = true;
    condition.wakeAll();
}

std::vector<QImage> TrainingWorker::loadImages(const QString& dir)
{
    std::vector<QImage> images;
    
    QDirIterator it(dir, QStringList() << "*.png" << "*.bmp", QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        QString filePath = it.next();
        QImageReader reader(filePath);
        QImage image = reader.read();
        
        if (!image.isNull()) {
            images.push_back(image);
        } else {
            qWarning() << "Failed to load image:" << filePath << reader.errorString();
        }
    }
    
    return images;
}

void TrainingWorker::train()
{
    QMutexLocker locker(&mutex);
    
    // Reset stop flag
    stopRequested = false;
    
    // Load positive examples
    std::vector<QImage> positiveImages = loadImages(positiveDir);
    if (positiveImages.empty()) {
        qWarning() << "No positive images found in" << positiveDir;
        emit trainingComplete(0.0f);
        return;
    }
    
    // Prepare training data
    std::vector<Eigen::VectorXf> inputs;
    std::vector<Eigen::VectorXf> targets;
    
    // Process positive examples
    for (const auto& image : positiveImages) {
        inputs.push_back(mlp->preprocessImage(image));
        targets.push_back(Eigen::VectorXf::Ones(1));
    }
    
    // Process negative examples if available
    if (!negativeDir.isEmpty()) {
        std::vector<QImage> negativeImages = loadImages(negativeDir);
        for (const auto& image : negativeImages) {
            inputs.push_back(mlp->preprocessImage(image));
            targets.push_back(Eigen::VectorXf::Zero(1));
        }
    }
    
    // Training loop
    float totalLoss = 0.0f;
    int totalBatches = (inputs.size() + batchSize - 1) / batchSize;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle data if requested
        if (shuffle) {
            std::random_device rd;
            std::mt19937 g(rd());
            
            std::vector<int> indices(inputs.size());
            for (int i = 0; i < indices.size(); ++i) {
                indices[i] = i;
            }
            std::shuffle(indices.begin(), indices.end(), g);
            
            std::vector<Eigen::VectorXf> shuffledInputs;
            std::vector<Eigen::VectorXf> shuffledTargets;
            
            for (int idx : indices) {
                shuffledInputs.push_back(inputs[idx]);
                shuffledTargets.push_back(targets[idx]);
            }
            
            inputs = shuffledInputs;
            targets = shuffledTargets;
        }
        
        // Train on batches
        totalLoss = 0.0f;
        for (int i = 0; i < inputs.size(); i += batchSize) {
            int batchEnd = std::min(i + batchSize, static_cast<int>(inputs.size()));
            float batchLoss = 0.0f;
            
            for (int j = i; j < batchEnd; ++j) {
                batchLoss += mlp->train(inputs[j], targets[j], learningRate);
            }
            
            totalLoss += batchLoss;
            
            // Check if stop requested
            if (stopRequested) {
                emit trainingComplete(totalLoss / inputs.size());
                return;
            }
        }
        
        // Calculate average loss
        float avgLoss = totalLoss / inputs.size();
        
        // Emit progress
        emit progressUpdated(epoch + 1, epochs, avgLoss);
    }
    
    // Training complete
    emit trainingComplete(totalLoss / inputs.size());
}

void TrainingWorker::evaluate()
{
    QMutexLocker locker(&mutex);
    
    // Load positive examples
    std::vector<QImage> positiveImages = loadImages(positiveDir);
    if (positiveImages.empty()) {
        qWarning() << "No positive images found in" << positiveDir;
        emit evaluationComplete(0.0f, 0, 0, 0, 0);
        return;
    }
    
    // Load negative examples
    std::vector<QImage> negativeImages = loadImages(negativeDir);
    if (negativeImages.empty()) {
        qWarning() << "No negative images found in" << negativeDir;
        emit evaluationComplete(0.0f, 0, 0, 0, 0);
        return;
    }
    
    // Evaluate on positive examples
    int truePositives = 0;
    int falseNegatives = 0;
    
    for (const auto& image : positiveImages) {
        float prediction = mlp->predict(image);
        if (prediction >= 0.5f) {
            truePositives++;
        } else {
            falseNegatives++;
        }
    }
    
    // Evaluate on negative examples
    int trueNegatives = 0;
    int falsePositives = 0;
    
    for (const auto& image : negativeImages) {
        float prediction = mlp->predict(image);
        if (prediction < 0.5f) {
            trueNegatives++;
        } else {
            falsePositives++;
        }
    }
    
    // Calculate accuracy
    int totalSamples = positiveImages.size() + negativeImages.size();
    float accuracy = static_cast<float>(truePositives + trueNegatives) / totalSamples;
    
    // Emit evaluation results
    emit evaluationComplete(accuracy, truePositives, trueNegatives, falsePositives, falseNegatives);
}
