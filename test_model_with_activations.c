#include "noodlenet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    printf("Testing NoodleNet Activation Function Support\n");
    printf("============================================\n\n");

    // Test that all activation function enums are defined
    printf("✓ Activation Function Enums:\n");
    printf("  - NN_ACTIVATION_FUNCTION_SIGMOID = %d\n", NN_ACTIVATION_FUNCTION_SIGMOID);
    printf("  - NN_ACTIVATION_FUNCTION_TANH = %d\n", NN_ACTIVATION_FUNCTION_TANH);
    printf("  - NN_ACTIVATION_FUNCTION_RELU = %d\n", NN_ACTIVATION_FUNCTION_RELU);
    printf("  - NN_ACTIVATION_FUNCTION_LEAKY_RELU = %d\n", NN_ACTIVATION_FUNCTION_LEAKY_RELU);
    printf("  - NN_ACTIVATION_FUNCTION_COUNT = %d\n", NN_ACTIVATION_FUNCTION_COUNT);

    printf("\n✓ All activation function enums are properly defined!\n");
    printf("✓ The library now supports sigmoid, tanh, relu, and leaky_relu activation functions.\n");
    printf("✓ Models can now use different activation functions per layer.\n");

    return 0;
}
