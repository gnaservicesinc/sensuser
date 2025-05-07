#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include <functional>
#include <string>

/**
 * @brief The Layer class represents a single layer in a neural network
 */
class Layer
{
public:
    /**
     * @brief Layer constructor
     * @param inputSize Number of input neurons
     * @param outputSize Number of output neurons
     * @param activationFunction Activation function to use
     */
    Layer(int inputSize, int outputSize, const std::string& activationFunction = "sigmoid");
    
    /**
     * @brief Forward pass through the layer
     * @param input Input values
     * @return Output values after activation
     */
    Eigen::VectorXf forward(const Eigen::VectorXf& input);
    
    /**
     * @brief Backward pass through the layer
     * @param outputGradient Gradient from the next layer
     * @param learningRate Learning rate for weight updates
     * @return Gradient to pass to the previous layer
     */
    Eigen::VectorXf backward(const Eigen::VectorXf& outputGradient, float learningRate);
    
    /**
     * @brief Get the weights of the layer
     * @return Weight matrix
     */
    const Eigen::MatrixXf& getWeights() const { return weights; }
    
    /**
     * @brief Set the weights of the layer
     * @param newWeights New weight matrix
     */
    void setWeights(const Eigen::MatrixXf& newWeights) { weights = newWeights; }
    
    /**
     * @brief Get the biases of the layer
     * @return Bias vector
     */
    const Eigen::VectorXf& getBiases() const { return biases; }
    
    /**
     * @brief Set the biases of the layer
     * @param newBiases New bias vector
     */
    void setBiases(const Eigen::VectorXf& newBiases) { biases = newBiases; }
    
    /**
     * @brief Get the activation function name
     * @return Activation function name
     */
    const std::string& getActivationFunction() const { return activationFunctionName; }
    
    /**
     * @brief Get the input size of the layer
     * @return Input size
     */
    int getInputSize() const { return inputSize; }
    
    /**
     * @brief Get the output size of the layer
     * @return Output size
     */
    int getOutputSize() const { return outputSize; }
    
    /**
     * @brief Get the last input to the layer
     * @return Last input
     */
    const Eigen::VectorXf& getLastInput() const { return lastInput; }
    
    /**
     * @brief Get the last output from the layer
     * @return Last output
     */
    const Eigen::VectorXf& getLastOutput() const { return lastOutput; }
    
    /**
     * @brief Get the last pre-activation values
     * @return Last pre-activation values
     */
    const Eigen::VectorXf& getLastZ() const { return lastZ; }
    
private:
    int inputSize;
    int outputSize;
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    std::string activationFunctionName;
    
    // Activation functions
    std::function<float(float)> activation;
    std::function<float(float)> activationDerivative;
    
    // Cache for backpropagation
    Eigen::VectorXf lastInput;
    Eigen::VectorXf lastZ;
    Eigen::VectorXf lastOutput;
    
    // Initialize activation functions based on name
    void initializeActivationFunctions();
};

#endif // LAYER_H
