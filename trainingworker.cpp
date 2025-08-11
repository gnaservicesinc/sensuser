#include "trainingworker.h"
#include <QDirIterator>
#include <QFileInfo>
#include <QImageReader>
#include <QDebug>
#include <QThread>
#include <algorithm>
#include <random>

TrainingWorker::TrainingWorker(NoodleNetBackend* backend, QObject* parent)
    : QObject(parent), backend(backend), learningRate(0.01f), epochs(100), batchSize(10), shuffle(true),
      optimizer(OptimizerType::SGD), stopRequested(false)
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

void TrainingWorker::setValidationDir(const QString& dir)
{
    QMutexLocker locker(&mutex);
    validationDir = dir;
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

void TrainingWorker::setOptimizer(OptimizerType optimizer)
{
    QMutexLocker locker(&mutex);
    this->optimizer = optimizer;
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
    QString localValidationDir;
    float localLearningRate;
    int localEpochs;
    int localBatchSize;
    bool localShuffle;
    OptimizerType localOptimizer;

    // Get parameters under mutex lock
    {
        QMutexLocker locker(&mutex);

        // Reset stop flag
        stopRequested = false;

        // Make local copies of all parameters
        localPositiveDir = positiveDir;
        localNegativeDir = negativeDir;
        localValidationDir = validationDir;
        localLearningRate = learningRate;
        localEpochs = epochs;
        localBatchSize = batchSize;
        localShuffle = shuffle;
        localOptimizer = optimizer;

        // Clear loss history at the start of training
        m_trainingLossHistory.clear();
        m_validationLossHistory.clear();
    }

    // Training loop (lib-backed): call one step at a time to emit progress
    float totalLoss = 0.0f;
    // set optimizer on backend based on selected optimizer
    if (backend) {
        backend->setOptimizer(localOptimizer);
    }
    for (int epoch = 0; epoch < localEpochs; ++epoch) {
        // Check if stop requested before each epoch
        {
            QMutexLocker locker(&mutex);
            if (stopRequested) {
                emit trainingComplete(totalLoss);
                return;
            }
        }

        // One training step using libnoodlenet
        float lastLoss = 0.0f;
        bool ok = (backend && backend->trainFromDirs(localPositiveDir,
                                                     localNegativeDir,
                                                     localValidationDir,
                                                     /*steps*/1,
                                                     localBatchSize,
                                                     localLearningRate,
                                                     /*l1*/0.0f,
                                                     /*l2*/0.0f,
                                                     lastLoss));
        if (!ok) {
            qWarning() << "Training step failed (libnoodlenet).";
        }

        float avgLoss = lastLoss;
        // Compute validation loss (sample from available dirs if both set)
        float valLoss = -1.0f;
        if (backend && !localPositiveDir.isEmpty() && !localNegativeDir.isEmpty()) {
            // Sample up to N images from each dir and compute BCE
            const int maxPerClass = 64;
            int count = 0;
            double sumLoss = 0.0;
            // Prefer validation dir if provided; otherwise fallback to training dirs
            QString posDirForVal = !localValidationDir.isEmpty() ? (localValidationDir + "/pos") : localPositiveDir;
            QString negDirForVal = !localValidationDir.isEmpty() ? (localValidationDir + "/neg") : localNegativeDir;
            // Iterate positives
            int posCount = 0;
            QDirIterator itp(posDirForVal, QStringList() << "*.png" << "*.bmp" << "*.jpg" << "*.jpeg", QDir::Files, QDirIterator::Subdirectories);
            while (itp.hasNext() && posCount < maxPerClass) {
                QString p = itp.next();
                QImageReader r(p); QImage img = r.read(); if (img.isNull()) continue;
                float prob = backend->predict(img);
                if (prob >= 0.0f) { double l = -(log(std::max(1e-7f, prob))); sumLoss += l; count++; }
                posCount++;
            }
            // Iterate negatives
            int negCount = 0;
            QDirIterator itn(negDirForVal, QStringList() << "*.png" << "*.bmp" << "*.jpg" << "*.jpeg", QDir::Files, QDirIterator::Subdirectories);
            while (itn.hasNext() && negCount < maxPerClass) {
                QString p = itn.next();
                QImageReader r(p); QImage img = r.read(); if (img.isNull()) continue;
                float prob = backend->predict(img);
                if (prob >= 0.0f) { double l = -(log(std::max(1e-7f, 1.0f - prob))); sumLoss += l; count++; }
                negCount++;
            }
            if (count > 0) valLoss = (float)(sumLoss / count);
        }
        totalLoss = avgLoss;

        // Store loss history - need to lock mutex for this
        {
            QMutexLocker locker(&mutex);
            m_trainingLossHistory.append(QPointF(epoch + 1, avgLoss));
        }

        // Emit progress and epoch completion - done outside the mutex lock
        emit progressUpdated(epoch + 1, localEpochs, avgLoss);
        emit epochCompleted(epoch + 1, avgLoss, valLoss);
    }

    // Training complete
    emit trainingComplete(totalLoss);
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

    // Evaluate using libnoodlenet
    float accuracy = 0.0f;
    int tp=0, tn=0, fp=0, fn=0;
    bool ok = (backend && backend->evaluateDirs(localPositiveDir, localNegativeDir, accuracy, tp, tn, fp, fn));
    if (!ok) {
        emit evaluationComplete(0.0f, 0, 0, 0, 0);
        return;
    }
    emit evaluationComplete(accuracy, tp, tn, fp, fn);
}
