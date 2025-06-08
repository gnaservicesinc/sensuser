#include "layer.h"
#include "optimizertypes.h"
#include <iostream>
#include <vector>
#include <random>
#include </usr/local/include/Eigen/Dense>

/**
 * Simple test to verify Adam optimizer fix
 * This test creates a simple layer and tests Adam updates
 */
int main() {
    std::cout << "Testing Adam Optimizer Fix..." << std::endl;

    // Create a simple layer: 2 inputs, 3 outputs
    Layer layer(2, 3, "sigmoid");

    // Set Adam hyperparameters (using standard values)
    layer.setAdamHyperparameters(0.9f, 0.999f, 1e-8f);
    
    // Test data
    Eigen::VectorXf input(2);
    input << 1.0f, 0.5f;

    // Reset optimizer state before training
    layer.resetOptimizerState();
    
    float learningRate = 0.01f;
    int timesteps = 10;

    std::cout << "Testing Adam optimizer with " << timesteps << " timesteps..." << std::endl;
    std::cout << "Learning rate: " << learningRate << std::endl;

    // Get initial weights for comparison
    Eigen::MatrixXf initialWeights = layer.getWeights();
    Eigen::VectorXf initialBiases = layer.getBiases();

    std::cout << "Initial weights norm: " << initialWeights.norm() << std::endl;
    std::cout << "Initial biases norm: " << initialBiases.norm() << std::endl;

    // Simulate training steps
    for (int timestep = 1; timestep <= timesteps; ++timestep) {
        // Forward pass
        Eigen::VectorXf output = layer.forward(input);

        // Create a dummy gradient (simulating backprop)
        Eigen::VectorXf gradient(3);
        gradient << 0.1f, -0.05f, 0.08f;

        // Apply Adam update
        layer.backward(gradient, learningRate, OptimizerType::Adam, timestep);

        // Check weights after update
        Eigen::MatrixXf currentWeights = layer.getWeights();
        Eigen::VectorXf currentBiases = layer.getBiases();

        std::cout << "Timestep " << timestep
                  << ": weights norm = " << currentWeights.norm()
                  << ", biases norm = " << currentBiases.norm() << std::endl;
    }
    
    // Final check
    Eigen::MatrixXf finalWeights = layer.getWeights();
    Eigen::VectorXf finalBiases = layer.getBiases();

    std::cout << "\nFinal weights norm: " << finalWeights.norm() << std::endl;
    std::cout << "Final biases norm: " << finalBiases.norm() << std::endl;

    // Check if weights changed (indicating Adam is working)
    float weightChange = (finalWeights - initialWeights).norm();
    float biasChange = (finalBiases - initialBiases).norm();

    std::cout << "Weight change: " << weightChange << std::endl;
    std::cout << "Bias change: " << biasChange << std::endl;

    bool success = (weightChange > 1e-6f && biasChange > 1e-6f);

    if (success) {
        std::cout << "\n✓ Adam optimizer test PASSED! Weights and biases updated correctly." << std::endl;
    } else {
        std::cout << "\n✗ Adam optimizer test FAILED! Weights and biases did not update properly." << std::endl;
    }

    return success ? 0 : 1;
}
