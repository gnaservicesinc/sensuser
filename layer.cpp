#include "layer.h"
#include <cmath>
#include <random>

Layer::Layer(int inputSize, int outputSize, const std::string& activationFunction)
    : inputSize(inputSize), outputSize(outputSize), activationFunctionName(activationFunction)
{
    // Initialize weights with Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f / (inputSize + outputSize));
    std::uniform_real_distribution<float> dis(-limit, limit);
    
    weights = Eigen::MatrixXf(outputSize, inputSize);
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights(i, j) = dis(gen);
        }
    }
    
    // Initialize biases to zero
    biases = Eigen::VectorXf::Zero(outputSize);
    
    // Initialize activation functions
    initializeActivationFunctions();
}

void Layer::initializeActivationFunctions()
{
    if (activationFunctionName == "sigmoid") {
        // Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
        activation = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
        // Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
        activationDerivative = [](float x) {
            float sigmoid = 1.0f / (1.0f + std::exp(-x));
            return sigmoid * (1.0f - sigmoid);
        };
    }
    else if (activationFunctionName == "relu") {
        // ReLU activation function: f(x) = max(0, x)
        activation = [](float x) { return std::max(0.0f, x); };
        // Derivative of ReLU: f'(x) = 1 if x > 0, 0 otherwise
        activationDerivative = [](float x) { return x > 0.0f ? 1.0f : 0.0f; };
    }
    else if (activationFunctionName == "tanh") {
        // Tanh activation function: f(x) = tanh(x)
        activation = [](float x) { return std::tanh(x); };
        // Derivative of tanh: f'(x) = 1 - tanh^2(x)
        activationDerivative = [](float x) {
            float tanhx = std::tanh(x);
            return 1.0f - tanhx * tanhx;
        };
    }
    else {
        // Default to sigmoid if unknown activation function
        activationFunctionName = "sigmoid";
        activation = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
        activationDerivative = [](float x) {
            float sigmoid = 1.0f / (1.0f + std::exp(-x));
            return sigmoid * (1.0f - sigmoid);
        };
    }
}

Eigen::VectorXf Layer::forward(const Eigen::VectorXf& input)
{
    // Cache input for backpropagation
    lastInput = input;
    
    // Calculate weighted sum: z = Wx + b
    lastZ = weights * input + biases;
    
    // Apply activation function element-wise
    lastOutput = Eigen::VectorXf(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        lastOutput(i) = activation(lastZ(i));
    }
    
    return lastOutput;
}

Eigen::VectorXf Layer::backward(const Eigen::VectorXf& outputGradient, float learningRate)
{
    // Calculate gradient of activation function
    Eigen::VectorXf activationGradient(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        activationGradient(i) = activationDerivative(lastZ(i));
    }
    
    // Element-wise multiplication of output gradient and activation gradient
    Eigen::VectorXf delta = outputGradient.array() * activationGradient.array();
    
    // Calculate gradient for the previous layer
    Eigen::VectorXf inputGradient = weights.transpose() * delta;
    
    // Update weights: W -= learning_rate * delta * input^T
    weights -= learningRate * (delta * lastInput.transpose());
    
    // Update biases: b -= learning_rate * delta
    biases -= learningRate * delta;
    
    return inputGradient;
}
