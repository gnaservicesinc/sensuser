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
    clearLossHistory();
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

QVector<QPointF> TrainingWorker::getTrainingLossHistory() const
{
    // Use a non-const mutex for locking
    QMutexLocker locker(const_cast<QMutex*>(&mutex));
    return m_trainingLossHistory;
}

QVector<QPointF> TrainingWorker::getValidationLossHistory() const
{
    // Use a non-const mutex for locking
    QMutexLocker locker(const_cast<QMutex*>(&mutex));
    return m_validationLossHistory;
}

void TrainingWorker::clearLossHistory()
{
    QMutexLocker locker(&mutex);
    m_trainingLossHistory.clear();
    m_validationLossHistory.clear();
    // Note: This method is already properly protected with a mutex lock
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
    // Local variables to store thread-safe copies of the parameters
    QString localPositiveDir;
    QString localNegativeDir;
    float localLearningRate;
    int localEpochs;
    int localBatchSize;
    bool localShuffle;

    // Get parameters under mutex lock
    {
        QMutexLocker locker(&mutex);

        // Reset stop flag
        stopRequested = false;

        // Make local copies of all parameters
        localPositiveDir = positiveDir;
        localNegativeDir = negativeDir;
        localLearningRate = learningRate;
        localEpochs = epochs;
        localBatchSize = batchSize;
        localShuffle = shuffle;

        // Clear loss history at the start of training
        m_trainingLossHistory.clear();
        m_validationLossHistory.clear();
    }

    // Load positive examples - done outside the mutex lock
    std::vector<QImage> positiveImages = loadImages(localPositiveDir);
    if (positiveImages.empty()) {
        qWarning() << "No positive images found in" << localPositiveDir;
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
    if (!localNegativeDir.isEmpty()) {
        std::vector<QImage> negativeImages = loadImages(localNegativeDir);
        for (const auto& image : negativeImages) {
            inputs.push_back(mlp->preprocessImage(image));
            targets.push_back(Eigen::VectorXf::Zero(1));
        }
    }

    // Training loop
    float totalLoss = 0.0f;

    for (int epoch = 0; epoch < localEpochs; ++epoch) {
        // Check if stop requested before each epoch
        {
            QMutexLocker locker(&mutex);
            if (stopRequested) {
                emit trainingComplete(totalLoss / inputs.size());
                return;
            }
        }

        // Shuffle data if requested
        if (localShuffle) {
            std::random_device rd;
            std::mt19937 g(rd());

            std::vector<int> indices(inputs.size());
            for (size_t i = 0; i < indices.size(); ++i) {
                indices[i] = static_cast<int>(i);
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
        for (size_t i = 0; i < inputs.size(); i += localBatchSize) {
            size_t batchEnd = std::min(i + static_cast<size_t>(localBatchSize), inputs.size());
            float batchLoss = 0.0f;

            for (size_t j = i; j < batchEnd; ++j) {
                batchLoss += mlp->train(inputs[j], targets[j], localLearningRate);
            }

            totalLoss += batchLoss;

            // Check if stop requested periodically
            {
                QMutexLocker locker(&mutex);
                if (stopRequested) {
                    emit trainingComplete(totalLoss / inputs.size());
                    return;
                }
            }
        }

        // Calculate average loss
        float avgLoss = totalLoss / inputs.size();

        // Store loss history - need to lock mutex for this
        {
            QMutexLocker locker(&mutex);
            m_trainingLossHistory.append(QPointF(epoch + 1, avgLoss));
        }

        // Emit progress and epoch completion - done outside the mutex lock
        emit progressUpdated(epoch + 1, localEpochs, avgLoss);
        emit epochCompleted(epoch + 1, avgLoss);
    }

    // Training complete
    emit trainingComplete(totalLoss / inputs.size());
}

void TrainingWorker::evaluate()
{
    // Local variables to store thread-safe copies of the parameters
    QString localPositiveDir;
    QString localNegativeDir;

    // Get parameters under mutex lock
    {
        QMutexLocker locker(&mutex);
        localPositiveDir = positiveDir;
        localNegativeDir = negativeDir;
    }

    // Load positive examples - done outside the mutex lock
    std::vector<QImage> positiveImages = loadImages(localPositiveDir);
    if (positiveImages.empty()) {
        qWarning() << "No positive images found in" << localPositiveDir;
        emit evaluationComplete(0.0f, 0, 0, 0, 0);
        return;
    }

    // Load negative examples
    std::vector<QImage> negativeImages = loadImages(localNegativeDir);
    if (negativeImages.empty()) {
        qWarning() << "No negative images found in" << localNegativeDir;
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
