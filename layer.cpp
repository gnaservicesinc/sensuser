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

    // Initialize optimizer state variables to zero
    m_weights = Eigen::MatrixXf::Zero(outputSize, inputSize);
    m_biases = Eigen::VectorXf::Zero(outputSize);
    v_weights = Eigen::MatrixXf::Zero(outputSize, inputSize);
    v_biases = Eigen::VectorXf::Zero(outputSize);

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
    else if (activationFunctionName == "leaky_relu") {
        // Leaky ReLU activation function: f(x) = x if x > 0, 0.01*x otherwise
        activation = [](float x) { return x > 0.0f ? x : 0.01f * x; };
        // Derivative of Leaky ReLU: f'(x) = 1 if x > 0, 0.01 otherwise
        activationDerivative = [](float x) { return x > 0.0f ? 1.0f : 0.01f; };
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
    // Use SGD optimizer for backward compatibility
    return backward(outputGradient, learningRate, OptimizerType::SGD, 1);
}

Eigen::VectorXf Layer::backward(const Eigen::VectorXf& outputGradient, float learningRate,
                               OptimizerType optimizer, int timestep)
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

    // Apply the selected optimizer
    switch (optimizer) {
        case OptimizerType::SGD:
            applySGD(delta, learningRate);
            break;
        case OptimizerType::RMSprop:
            applyRMSprop(delta, learningRate);
            break;
        case OptimizerType::Adam:
            applyAdam(delta, learningRate, timestep);
            break;
    }

    return inputGradient;
}

void Layer::resetOptimizerState()
{
    // Reset all optimizer state variables to zero
    m_weights.setZero();
    m_biases.setZero();
    v_weights.setZero();
    v_biases.setZero();
}

void Layer::applySGD(const Eigen::VectorXf& delta, float learningRate)
{
    // Standard SGD update
    weights -= learningRate * (delta * lastInput.transpose());
    biases -= learningRate * delta;
}

void Layer::applyRMSprop(const Eigen::VectorXf& delta, float learningRate)
{
    // RMSprop hyperparameters
    const float beta = 0.9f;
    const float epsilon = 1e-8f;

    // Calculate weight gradients
    Eigen::MatrixXf weightGradients = delta * lastInput.transpose();

    // Update second moment for weights: v = beta * v + (1 - beta) * gradient^2
    v_weights = beta * v_weights + (1.0f - beta) * weightGradients.array().square().matrix();

    // Update second moment for biases
    v_biases = beta * v_biases + (1.0f - beta) * delta.array().square().matrix();

    // Update weights: W -= learning_rate * gradient / sqrt(v + epsilon)
    weights -= learningRate * (weightGradients.array() / (v_weights.array().sqrt() + epsilon)).matrix();

    // Update biases: b -= learning_rate * gradient / sqrt(v + epsilon)
    biases -= learningRate * (delta.array() / (v_biases.array().sqrt() + epsilon)).matrix();
}

void Layer::applyAdam(const Eigen::VectorXf& delta, float learningRate, int timestep)
{
    // Adam hyperparameters
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;

    // Calculate weight gradients
    Eigen::MatrixXf weightGradients = delta * lastInput.transpose();

    // Update first moment for weights: m = beta1 * m + (1 - beta1) * gradient
    m_weights = beta1 * m_weights + (1.0f - beta1) * weightGradients;

    // Update second moment for weights: v = beta2 * v + (1 - beta2) * gradient^2
    v_weights = beta2 * v_weights + (1.0f - beta2) * weightGradients.array().square().matrix();

    // Update first moment for biases
    m_biases = beta1 * m_biases + (1.0f - beta1) * delta;

    // Update second moment for biases
    v_biases = beta2 * v_biases + (1.0f - beta2) * delta.array().square().matrix();

    // Bias correction
    float beta1_t = std::pow(beta1, timestep);
    float beta2_t = std::pow(beta2, timestep);

    Eigen::MatrixXf m_hat_weights = m_weights / (1.0f - beta1_t);
    Eigen::MatrixXf v_hat_weights = v_weights / (1.0f - beta2_t);

    Eigen::VectorXf m_hat_biases = m_biases / (1.0f - beta1_t);
    Eigen::VectorXf v_hat_biases = v_biases / (1.0f - beta2_t);

    // Update weights: W -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    weights -= learningRate * (m_hat_weights.array() / (v_hat_weights.array().sqrt() + epsilon)).matrix();

    // Update biases: b -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    biases -= learningRate * (m_hat_biases.array() / (v_hat_biases.array().sqrt() + epsilon)).matrix();
}
