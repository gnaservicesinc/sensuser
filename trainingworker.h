#ifndef TRAININGWORKER_H
#define TRAININGWORKER_H

#include "mlp.h"
#include <QObject>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QDir>
#include <QImage>
#include <vector>

/**
 * @brief The TrainingWorker class handles training the MLP in a separate thread
 */
class TrainingWorker : public QObject
{
    Q_OBJECT
    
public:
    /**
     * @brief TrainingWorker constructor
     * @param mlp Pointer to the MLP to train
     * @param parent Parent object
     */
    explicit TrainingWorker(MLP* mlp, QObject* parent = nullptr);
    
    /**
     * @brief Set the positive examples directory
     * @param dir Directory containing positive examples
     */
    void setPositiveDir(const QString& dir);
    
    /**
     * @brief Set the negative examples directory
     * @param dir Directory containing negative examples
     */
    void setNegativeDir(const QString& dir);
    
    /**
     * @brief Set the learning rate
     * @param rate Learning rate
     */
    void setLearningRate(float rate);
    
    /**
     * @brief Set the number of epochs
     * @param epochs Number of epochs
     */
    void setEpochs(int epochs);
    
    /**
     * @brief Set the batch size
     * @param size Batch size
     */
    void setBatchSize(int size);
    
    /**
     * @brief Set whether to shuffle data each epoch
     * @param shuffle Whether to shuffle data
     */
    void setShuffle(bool shuffle);
    
    /**
     * @brief Stop training
     */
    void stop();
    
public slots:
    /**
     * @brief Start training
     */
    void train();
    
    /**
     * @brief Evaluate the model on the test set
     */
    void evaluate();
    
signals:
    /**
     * @brief Signal emitted when training progress is updated
     * @param epoch Current epoch
     * @param totalEpochs Total number of epochs
     * @param loss Current loss
     */
    void progressUpdated(int epoch, int totalEpochs, float loss);
    
    /**
     * @brief Signal emitted when training is complete
     * @param finalLoss Final loss
     */
    void trainingComplete(float finalLoss);
    
    /**
     * @brief Signal emitted when evaluation is complete
     * @param accuracy Accuracy of the model
     * @param truePositives Number of true positives
     * @param trueNegatives Number of true negatives
     * @param falsePositives Number of false positives
     * @param falseNegatives Number of false negatives
     */
    void evaluationComplete(float accuracy, int truePositives, int trueNegatives, int falsePositives, int falseNegatives);
    
private:
    MLP* mlp;
    QString positiveDir;
    QString negativeDir;
    float learningRate;
    int epochs;
    int batchSize;
    bool shuffle;
    bool stopRequested;
    
    QMutex mutex;
    QWaitCondition condition;
    
    /**
     * @brief Load images from a directory
     * @param dir Directory to load images from
     * @return Vector of loaded images
     */
    std::vector<QImage> loadImages(const QString& dir);
};

#endif // TRAININGWORKER_H
