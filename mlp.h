#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include "optimizertypes.h"
#include <vector>
#include <string>
#include </usr/local/include/Eigen/Dense>
#include <QImage>
#include <QJsonObject>
#include <QFile>
#include <QDataStream>
#include <QByteArray>
#include <QJsonDocument>
#include <QMutex>
#include <QMutexLocker>
#include <QReadWriteLock>
#include <QReadLocker>
#include <QWriteLocker>

/**
 * @brief The MLP class represents a Multi-Layer Perceptron neural network
 */
class MLP
{
public:
    /**
     * @brief MLP constructor with a single hidden layer (for backward compatibility)
     * @param inputSize Number of input neurons
     * @param hiddenSize Number of neurons in the hidden layer
     * @param outputSize Number of output neurons
     * @param hiddenActivation Activation function for the hidden layer
     * @param outputActivation Activation function for the output layer
     */
    MLP(int inputSize, int hiddenSize, int outputSize,
        const std::string& hiddenActivation = "sigmoid",
        const std::string& outputActivation = "sigmoid");

    /**
     * @brief MLP constructor with multiple hidden layers
     * @param inputSize Number of input neurons
     * @param hiddenSizes Vector of neuron counts for each hidden layer
     * @param outputSize Number of output neurons
     * @param hiddenActivation Activation function for all hidden layers
     * @param outputActivation Activation function for the output layer
     */
    MLP(int inputSize, const std::vector<int>& hiddenSizes, int outputSize,
        const std::string& hiddenActivation = "sigmoid",
        const std::string& outputActivation = "sigmoid");

    /**
     * @brief MLP constructor with multiple hidden layers and different activation functions per layer
     * @param inputSize Number of input neurons
     * @param hiddenSizes Vector of neuron counts for each hidden layer
     * @param outputSize Number of output neurons
     * @param hiddenActivations Vector of activation functions for each hidden layer
     * @param outputActivation Activation function for the output layer
     */
    MLP(int inputSize, const std::vector<int>& hiddenSizes, int outputSize,
        const std::vector<std::string>& hiddenActivations,
        const std::string& outputActivation = "sigmoid");

    /**
     * @brief Forward pass through the network (thread-safe)
     * @param input Input values
     * @return Output values
     */
    Eigen::VectorXf forward(const Eigen::VectorXf& input);

    /**
     * @brief Forward pass through the network without locking (for internal use)
     * @param input Input values
     * @return Output values
     */
    Eigen::VectorXf forwardUnsafe(const Eigen::VectorXf& input);

    /**
     * @brief Train the network on a single example (SGD version for backward compatibility)
     * @param input Input values
     * @param target Target values
     * @param learningRate Learning rate for weight updates
     * @return Loss value
     */
    float train(const Eigen::VectorXf& input, const Eigen::VectorXf& target, float learningRate);

    /**
     * @brief Train the network on a single example with optimizer support
     * @param input Input values
     * @param target Target values
     * @param learningRate Learning rate for weight updates
     * @param optimizer Optimizer type to use
     * @param timestep Current timestep (for Adam bias correction)
     * @return Loss value
     */
    float train(const Eigen::VectorXf& input, const Eigen::VectorXf& target, float learningRate,
                OptimizerType optimizer, int timestep = 1);

    /**
     * @brief Train the network on a single example with granular locking (for training loops)
     * @param input Input values
     * @param target Target values
     * @param learningRate Learning rate for weight updates
     * @param optimizer Optimizer type to use
     * @param timestep Current timestep (for Adam bias correction)
     * @return Loss value
     */
    float trainWithGranularLocking(const Eigen::VectorXf& input, const Eigen::VectorXf& target,
                                   float learningRate, OptimizerType optimizer, int timestep = 1);

    /**
     * @brief Reset optimizer state for all layers
     */
    void resetOptimizerState();

    /**
     * @brief Set Adam hyperparameters for all layers
     * @param beta1 First moment decay rate (default: 0.9)
     * @param beta2 Second moment decay rate (default: 0.999)
     * @param epsilon Small constant for numerical stability (default: 1e-8)
     */
    void setAdamHyperparameters(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    /**
     * @brief Preprocess an image for input to the network
     * @param image Input image
     * @return Preprocessed image as a vector
     */
    Eigen::VectorXf preprocessImage(const QImage& image);

    /**
     * @brief Predict whether an image contains the target object (thread-safe)
     * @param image Input image
     * @return Probability that the image contains the target object
     */
    float predict(const QImage& image);

    /**
     * @brief Get a thread-safe copy of the layers for visualization
     * @return Copy of the layers vector
     */
    std::vector<Layer> getLayersCopy() const;

    /**
     * @brief Try to get a copy of the layers without blocking (for UI)
     * @param layersCopy Output parameter for the layers copy
     * @param timeoutMs Maximum time to wait in milliseconds
     * @return True if successful, false if timeout or model is busy
     */
    bool tryGetLayersCopy(std::vector<Layer>& layersCopy, int timeoutMs = 100) const;

    /**
     * @brief Check if the model is currently busy (non-blocking)
     * @return True if model is busy with training or other write operations
     */
    bool isBusy() const;

    /**
     * @brief Get the layers of the network (thread-safe)
     * @return Vector of layers
     */
    const std::vector<Layer>& getLayers() const;

    /**
     * @brief Get the number of hidden layers (thread-safe)
     * @return Number of hidden layers
     */
    int getNumHiddenLayers() const;

    /**
     * @brief Get the sizes of all hidden layers
     * @return Vector of hidden layer sizes
     */
    std::vector<int> getHiddenLayerSizes() const;

    /**
     * @brief Save the model to a JSON object
     * @return JSON object containing the model
     */
    QJsonObject saveToJson() const;

    /**
     * @brief Load the model from a JSON object
     * @param json JSON object containing the model
     * @return True if successful, false otherwise
     */
    bool loadFromJson(const QJsonObject& json);

    /**
     * @brief Save the model to a binary file
     * @param filePath Path to save the model to
     * @return True if successful, false otherwise
     */
    bool saveToBinary(const QString& filePath) const;

    /**
     * @brief Load the model from a binary file
     * @param filePath Path to load the model from
     * @return True if successful, false otherwise
     */
    bool loadFromBinary(const QString& filePath);

private:
    // Constants for binary file format
    static const quint32 MAGIC_NUMBER = 0x4D4E4553; // "SENM" in little-endian (S E N M)
    static const quint32 MAGIC_NUMBER_REVERSED = 0x53454E4D; // "SENM" in big-endian (M N E S)
    static const quint8 FORMAT_VERSION = 0x02; // Incremented to support multiple hidden layers
    std::vector<Layer> layers;
    int inputSize;
    int outputSize;
    std::vector<int> hiddenSizes; // Store sizes of all hidden layers

    // Adam optimizer state
    int adamTimestep; // Global timestep counter for Adam bias correction

    // Thread safety
    mutable QReadWriteLock rwLock; // Protects all MLP operations with read/write semantics
    mutable QMutex trainingMutex; // Separate mutex for training operations

    /**
     * @brief Calculate the loss for a single example
     * @param output Output values
     * @param target Target values
     * @return Loss value
     */
    float calculateLoss(const Eigen::VectorXf& output, const Eigen::VectorXf& target) const;

    /**
     * @brief Calculate the gradient of the loss function
     * @param output Output values
     * @param target Target values
     * @return Gradient of the loss function
     */
    Eigen::VectorXf calculateLossGradient(const Eigen::VectorXf& output, const Eigen::VectorXf& target) const;
};

#endif // MLP_H
