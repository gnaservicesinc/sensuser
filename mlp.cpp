#include "mlp.h"
#include <cmath>
#include <QJsonArray>
#include <QJsonDocument>

MLP::MLP(int inputSize, int hiddenSize, int outputSize,
         const std::string& hiddenActivation, const std::string& outputActivation)
    : inputSize(inputSize), outputSize(outputSize)
{
    // Store hidden layer size
    hiddenSizes = {hiddenSize};

    // Create layers
    layers.push_back(Layer(inputSize, hiddenSize, hiddenActivation));
    layers.push_back(Layer(hiddenSize, outputSize, outputActivation));
}

MLP::MLP(int inputSize, const std::vector<int>& hiddenSizes, int outputSize,
         const std::string& hiddenActivation, const std::string& outputActivation)
    : inputSize(inputSize), hiddenSizes(hiddenSizes), outputSize(outputSize)
{
    if (hiddenSizes.empty()) {
        // If no hidden layers, create a direct input-to-output layer
        layers.push_back(Layer(inputSize, outputSize, outputActivation));
    } else {
        // Create first hidden layer (input to first hidden)
        layers.push_back(Layer(inputSize, hiddenSizes[0], hiddenActivation));

        // Create additional hidden layers
        for (size_t i = 1; i < hiddenSizes.size(); ++i) {
            layers.push_back(Layer(hiddenSizes[i-1], hiddenSizes[i], hiddenActivation));
        }

        // Create output layer (last hidden to output)
        layers.push_back(Layer(hiddenSizes.back(), outputSize, outputActivation));
    }
}

std::vector<int> MLP::getHiddenLayerSizes() const {
    return hiddenSizes;
}

Eigen::VectorXf MLP::forward(const Eigen::VectorXf& input)
{
    Eigen::VectorXf current = input;

    // Forward pass through each layer
    for (auto& layer : layers) {
        current = layer.forward(current);
    }

    return current;
}

float MLP::train(const Eigen::VectorXf& input, const Eigen::VectorXf& target, float learningRate)
{
    // Forward pass
    Eigen::VectorXf output = forward(input);

    // Calculate loss
    float loss = calculateLoss(output, target);

    // Calculate loss gradient
    Eigen::VectorXf gradient = calculateLossGradient(output, target);

    // Backward pass
    for (int i = layers.size() - 1; i >= 0; --i) {
        gradient = layers[i].backward(gradient, learningRate);
    }

    return loss;
}

float MLP::calculateLoss(const Eigen::VectorXf& output, const Eigen::VectorXf& target) const
{
    // Binary cross-entropy loss
    float loss = 0.0f;
    for (int i = 0; i < output.size(); ++i) {
        // Clip output to avoid log(0)
        float clippedOutput = std::max(1e-7f, std::min(1.0f - 1e-7f, output(i)));
        loss -= target(i) * std::log(clippedOutput) + (1.0f - target(i)) * std::log(1.0f - clippedOutput);
    }
    return loss;
}

Eigen::VectorXf MLP::calculateLossGradient(const Eigen::VectorXf& output, const Eigen::VectorXf& target) const
{
    // Gradient of binary cross-entropy loss
    Eigen::VectorXf gradient(output.size());
    for (int i = 0; i < output.size(); ++i) {
        // Clip output to avoid division by zero
        float clippedOutput = std::max(1e-7f, std::min(1.0f - 1e-7f, output(i)));
        gradient(i) = -target(i) / clippedOutput + (1.0f - target(i)) / (1.0f - clippedOutput);
    }
    return gradient;
}

Eigen::VectorXf MLP::preprocessImage(const QImage& image)
{
    // Convert to grayscale and resize if necessary
    QImage processedImage = image;
    if (processedImage.format() != QImage::Format_Grayscale8) {
        processedImage = processedImage.convertToFormat(QImage::Format_Grayscale8);
    }

    if (processedImage.width() != 512 || processedImage.height() != 512) {
        processedImage = processedImage.scaled(512, 512, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    }

    // Convert to vector and normalize
    Eigen::VectorXf input(512 * 512);
    for (int y = 0; y < 512; ++y) {
        for (int x = 0; x < 512; ++x) {
            // Normalize pixel value to [0, 1]
            input(y * 512 + x) = static_cast<float>(qGray(processedImage.pixel(x, y))) / 255.0f;
        }
    }

    return input;
}

float MLP::predict(const QImage& image)
{
    // Preprocess image
    Eigen::VectorXf input = preprocessImage(image);

    // Forward pass
    Eigen::VectorXf output = forward(input);

    // Return probability
    return output(0);
}

QJsonObject MLP::saveToJson() const
{
    QJsonObject json;

    // Save architecture
    QJsonObject architecture;
    architecture["input_neurons"] = inputSize;
    architecture["output_neurons"] = outputSize;

    // Save hidden layers configuration
    QJsonArray hiddenLayersArray;
    int numHiddenLayers = getNumHiddenLayers();

    for (int i = 0; i < numHiddenLayers; ++i) {
        QJsonObject hiddenLayer;
        hiddenLayer["neurons"] = layers[i].getOutputSize();
        hiddenLayer["activation"] = QString::fromStdString(layers[i].getActivationFunction());
        hiddenLayersArray.append(hiddenLayer);
    }

    architecture["hidden_layers"] = hiddenLayersArray;
    architecture["output_activation"] = QString::fromStdString(layers.back().getActivationFunction());

    json["architecture"] = architecture;

    // Save weights and biases
    QJsonObject weights;
    QJsonObject biases;

    // Save weights and biases for each layer
    for (size_t i = 0; i < layers.size(); ++i) {
        const Layer& layer = layers[i];

        // Create a name for this layer's weights and biases
        QString layerName;
        if (i == 0) {
            layerName = "input_to_hidden1";
        } else if (i == layers.size() - 1) {
            if (numHiddenLayers == 0) {
                layerName = "input_to_output";
            } else {
                layerName = QString("hidden%1_to_output").arg(numHiddenLayers);
            }
        } else {
            layerName = QString("hidden%1_to_hidden%2").arg(i).arg(i + 1);
        }

        // Save weights
        QJsonArray layerWeights;
        for (int row = 0; row < layer.getOutputSize(); ++row) {
            QJsonArray rowArray;
            for (int col = 0; col < layer.getInputSize(); ++col) {
                rowArray.append(layer.getWeights()(row, col));
            }
            layerWeights.append(rowArray);
        }
        weights[layerName] = layerWeights;

        // Save biases
        QJsonArray layerBiases;
        for (int j = 0; j < layer.getOutputSize(); ++j) {
            layerBiases.append(layer.getBiases()(j));
        }

        // Name for biases
        QString biasName;
        if (i == layers.size() - 1) {
            biasName = "output";
        } else {
            biasName = QString("hidden%1").arg(i + 1);
        }

        biases[biasName] = layerBiases;
    }

    json["weights"] = weights;
    json["biases"] = biases;

    return json;
}

bool MLP::loadFromJson(const QJsonObject& json)
{
    // Check if the JSON object has the required fields
    if (!json.contains("architecture") || !json.contains("weights") || !json.contains("biases")) {
        return false;
    }

    // Load architecture
    QJsonObject architecture = json["architecture"].toObject();
    if (architecture["input_neurons"].toInt() != inputSize ||
        architecture["output_neurons"].toInt() != outputSize) {
        return false;
    }

    // Get hidden layers configuration
    QJsonArray hiddenLayersArray = architecture["hidden_layers"].toArray();

    // Extract hidden layer sizes and activations
    std::vector<int> newHiddenSizes;
    std::vector<std::string> hiddenActivations;

    for (int i = 0; i < hiddenLayersArray.size(); ++i) {
        QJsonObject hiddenLayer = hiddenLayersArray[i].toObject();
        newHiddenSizes.push_back(hiddenLayer["neurons"].toInt());
        hiddenActivations.push_back(hiddenLayer["activation"].toString().toStdString());
    }

    QString outputActivation = architecture["output_activation"].toString();

    // Recreate the network with the specified hidden layers
    layers.clear();
    hiddenSizes = newHiddenSizes;

    if (hiddenSizes.empty()) {
        // If no hidden layers, create a direct input-to-output layer
        layers.push_back(Layer(inputSize, outputSize, outputActivation.toStdString()));
    } else {
        // Create first hidden layer (input to first hidden)
        layers.push_back(Layer(inputSize, hiddenSizes[0], hiddenActivations[0]));

        // Create additional hidden layers
        for (size_t i = 1; i < hiddenSizes.size(); ++i) {
            layers.push_back(Layer(hiddenSizes[i-1], hiddenSizes[i], hiddenActivations[i]));
        }

        // Create output layer (last hidden to output)
        layers.push_back(Layer(hiddenSizes.back(), outputSize, outputActivation.toStdString()));
    }

    // Load weights and biases
    QJsonObject weights = json["weights"].toObject();
    QJsonObject biases = json["biases"].toObject();

    // Load weights and biases for each layer
    for (size_t i = 0; i < layers.size(); ++i) {
        Layer& layer = layers[i];

        // Create a name for this layer's weights and biases
        QString layerName;
        if (i == 0) {
            layerName = "input_to_hidden1";
        } else if (i == layers.size() - 1) {
            if (hiddenSizes.empty()) {
                layerName = "input_to_output";
            } else {
                layerName = QString("hidden%1_to_output").arg(hiddenSizes.size());
            }
        } else {
            layerName = QString("hidden%1_to_hidden%2").arg(i).arg(i + 1);
        }

        // Load weights
        if (!weights.contains(layerName)) {
            return false;
        }

        QJsonArray layerWeights = weights[layerName].toArray();
        Eigen::MatrixXf weightMatrix(layer.getOutputSize(), layer.getInputSize());

        for (int row = 0; row < layer.getOutputSize(); ++row) {
            QJsonArray rowArray = layerWeights[row].toArray();
            for (int col = 0; col < layer.getInputSize(); ++col) {
                weightMatrix(row, col) = rowArray[col].toDouble();
            }
        }
        layer.setWeights(weightMatrix);

        // Name for biases
        QString biasName;
        if (i == layers.size() - 1) {
            biasName = "output";
        } else {
            biasName = QString("hidden%1").arg(i + 1);
        }

        // Load biases
        if (!biases.contains(biasName)) {
            return false;
        }

        QJsonArray layerBiases = biases[biasName].toArray();
        Eigen::VectorXf biasVector(layer.getOutputSize());

        for (int j = 0; j < layer.getOutputSize(); ++j) {
            biasVector(j) = layerBiases[j].toDouble();
        }
        layer.setBiases(biasVector);
    }

    return true;
}

bool MLP::saveToBinary(const QString& filePath) const
{
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly)) {
        return false;
    }

    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::LittleEndian);
    stream.setFloatingPointPrecision(QDataStream::SinglePrecision);

    // Write magic number and version (using little-endian format for new models)
    stream << MAGIC_NUMBER << FORMAT_VERSION;

    // Create JSON metadata
    QJsonObject metadata;

    // Save architecture
    QJsonObject architecture;
    architecture["input_neurons"] = inputSize;
    architecture["output_neurons"] = outputSize;

    // Save hidden layers configuration
    QJsonArray hiddenLayersArray;
    int numHiddenLayers = getNumHiddenLayers();

    for (int i = 0; i < numHiddenLayers; ++i) {
        QJsonObject hiddenLayer;
        hiddenLayer["neurons"] = layers[i].getOutputSize();
        hiddenLayer["activation"] = QString::fromStdString(layers[i].getActivationFunction());
        hiddenLayersArray.append(hiddenLayer);
    }

    architecture["hidden_layers"] = hiddenLayersArray;

    // Save output layer activation
    architecture["output_activation"] = QString::fromStdString(layers.back().getActivationFunction());

    metadata["architecture"] = architecture;
    metadata["data_precision"] = "float";

    // Convert metadata to JSON string
    QJsonDocument doc(metadata);
    QByteArray jsonData = doc.toJson(QJsonDocument::Compact);

    // Write metadata length and metadata
    stream << static_cast<quint32>(jsonData.size());
    stream.writeRawData(jsonData.constData(), jsonData.size());

    // Write weights and biases as binary data for all layers
    for (size_t i = 0; i < layers.size(); ++i) {
        const Layer& layer = layers[i];
        const Eigen::MatrixXf& weights = layer.getWeights();
        const Eigen::VectorXf& biases = layer.getBiases();

        // Write weights
        for (int row = 0; row < weights.rows(); ++row) {
            for (int col = 0; col < weights.cols(); ++col) {
                stream << static_cast<float>(weights(row, col));
            }
        }

        // Write biases
        for (int j = 0; j < biases.size(); ++j) {
            stream << static_cast<float>(biases(j));
        }
    }

    file.close();
    return true;
}

bool MLP::loadFromBinary(const QString& filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }

    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::LittleEndian);
    stream.setFloatingPointPrecision(QDataStream::SinglePrecision);

    // Read and verify magic number and version
    quint32 magicNumber;
    quint8 formatVersion;
    stream >> magicNumber >> formatVersion;

    // Accept both magic number formats for compatibility
    if (magicNumber != MAGIC_NUMBER && magicNumber != MAGIC_NUMBER_REVERSED) {
        file.close();
        return false;
    }

    // Read metadata length and metadata
    quint32 jsonLength;
    stream >> jsonLength;

    QByteArray jsonData = file.read(jsonLength);
    if (jsonData.size() != static_cast<int>(jsonLength)) {
        file.close();
        return false;
    }

    // Parse metadata
    QJsonDocument doc = QJsonDocument::fromJson(jsonData);
    if (doc.isNull() || !doc.isObject()) {
        file.close();
        return false;
    }

    QJsonObject metadata = doc.object();

    // Verify architecture
    QJsonObject architecture = metadata["architecture"].toObject();
    if (architecture["input_neurons"].toInt() != inputSize ||
        architecture["output_neurons"].toInt() != outputSize) {
        file.close();
        return false;
    }

    // Verify data precision
    QString dataPrecision = metadata["data_precision"].toString();
    if (dataPrecision != "float") {
        file.close();
        return false;
    }

    // Get hidden layers configuration
    QJsonArray hiddenLayersArray = architecture["hidden_layers"].toArray();

    // Handle different format versions
    if (formatVersion == 0x01) {
        // Old format with single hidden layer
        if (hiddenLayersArray.size() != 1) {
            file.close();
            return false;
        }

        int hiddenSize = hiddenLayersArray[0].toObject()["neurons"].toInt();
        QString hiddenActivation = hiddenLayersArray[0].toObject()["activation"].toString();
        QString outputActivation = architecture["output_activation"].toString();

        // Recreate the network with a single hidden layer
        layers.clear();
        hiddenSizes = {hiddenSize};
        layers.push_back(Layer(inputSize, hiddenSize, hiddenActivation.toStdString()));
        layers.push_back(Layer(hiddenSize, outputSize, outputActivation.toStdString()));

        // Read weights and biases

        // Layer 0 weights (input to hidden)
        Eigen::MatrixXf inputToHiddenWeights(hiddenSize, inputSize);
        for (int i = 0; i < hiddenSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                float value;
                stream >> value;
                inputToHiddenWeights(i, j) = value;
            }
        }
        layers[0].setWeights(inputToHiddenWeights);

        // Layer 0 biases (hidden)
        Eigen::VectorXf hiddenBiases(hiddenSize);
        for (int i = 0; i < hiddenSize; ++i) {
            float value;
            stream >> value;
            hiddenBiases(i) = value;
        }
        layers[0].setBiases(hiddenBiases);

        // Layer 1 weights (hidden to output)
        Eigen::MatrixXf hiddenToOutputWeights(outputSize, hiddenSize);
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < hiddenSize; ++j) {
                float value;
                stream >> value;
                hiddenToOutputWeights(i, j) = value;
            }
        }
        layers[1].setWeights(hiddenToOutputWeights);

        // Layer 1 biases (output)
        Eigen::VectorXf outputBiases(outputSize);
        for (int i = 0; i < outputSize; ++i) {
            float value;
            stream >> value;
            outputBiases(i) = value;
        }
        layers[1].setBiases(outputBiases);
    }
    else if (formatVersion == 0x02) {
        // New format with multiple hidden layers

        // Extract hidden layer sizes and activations
        std::vector<int> newHiddenSizes;
        std::vector<std::string> hiddenActivations;

        for (int i = 0; i < hiddenLayersArray.size(); ++i) {
            QJsonObject hiddenLayer = hiddenLayersArray[i].toObject();
            newHiddenSizes.push_back(hiddenLayer["neurons"].toInt());
            hiddenActivations.push_back(hiddenLayer["activation"].toString().toStdString());
        }

        QString outputActivation = architecture["output_activation"].toString();

        // Recreate the network with the specified hidden layers
        layers.clear();
        hiddenSizes = newHiddenSizes;

        if (hiddenSizes.empty()) {
            // If no hidden layers, create a direct input-to-output layer
            layers.push_back(Layer(inputSize, outputSize, outputActivation.toStdString()));
        } else {
            // Create first hidden layer (input to first hidden)
            layers.push_back(Layer(inputSize, hiddenSizes[0], hiddenActivations[0]));

            // Create additional hidden layers
            for (size_t i = 1; i < hiddenSizes.size(); ++i) {
                layers.push_back(Layer(hiddenSizes[i-1], hiddenSizes[i], hiddenActivations[i]));
            }

            // Create output layer (last hidden to output)
            layers.push_back(Layer(hiddenSizes.back(), outputSize, outputActivation.toStdString()));
        }

        // Read weights and biases for all layers
        for (size_t i = 0; i < layers.size(); ++i) {
            Layer& layer = layers[i];
            int outputSize = layer.getOutputSize();
            int inputSize = layer.getInputSize();

            // Read weights
            Eigen::MatrixXf weights(outputSize, inputSize);
            for (int row = 0; row < outputSize; ++row) {
                for (int col = 0; col < inputSize; ++col) {
                    float value;
                    stream >> value;
                    weights(row, col) = value;
                }
            }
            layer.setWeights(weights);

            // Read biases
            Eigen::VectorXf biases(outputSize);
            for (int j = 0; j < outputSize; ++j) {
                float value;
                stream >> value;
                biases(j) = value;
            }
            layer.setBiases(biases);
        }
    }
    else {
        // Unsupported format version
        file.close();
        return false;
    }

    file.close();
    return true;
}
