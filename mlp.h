#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include <vector>
#include <string>
#include </usr/local/include/Eigen/Dense>
#include <QImage>
#include <QJsonObject>
#include <QFile>
#include <QDataStream>
#include <QByteArray>
#include <QJsonDocument>

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
     * @brief Forward pass through the network
     * @param input Input values
     * @return Output values
     */
    Eigen::VectorXf forward(const Eigen::VectorXf& input);

    /**
     * @brief Train the network on a single example
     * @param input Input values
     * @param target Target values
     * @param learningRate Learning rate for weight updates
     * @return Loss value
     */
    float train(const Eigen::VectorXf& input, const Eigen::VectorXf& target, float learningRate);

    /**
     * @brief Preprocess an image for input to the network
     * @param image Input image
     * @return Preprocessed image as a vector
     */
    Eigen::VectorXf preprocessImage(const QImage& image);

    /**
     * @brief Predict whether an image contains the target object
     * @param image Input image
     * @return Probability that the image contains the target object
     */
    float predict(const QImage& image);

    /**
     * @brief Get the layers of the network
     * @return Vector of layers
     */
    const std::vector<Layer>& getLayers() const { return layers; }

    /**
     * @brief Get the number of hidden layers
     * @return Number of hidden layers
     */
    int getNumHiddenLayers() const { return layers.size() - 1; }

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
