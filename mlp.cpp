#include "mlp.h"
#include <cmath>
#include <QJsonArray>
#include <QJsonDocument>

MLP::MLP(int inputSize, int hiddenSize, int outputSize,
         const std::string& hiddenActivation, const std::string& outputActivation)
    : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize)
{
    // Create layers
    layers.push_back(Layer(inputSize, hiddenSize, hiddenActivation));
    layers.push_back(Layer(hiddenSize, outputSize, outputActivation));
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

    QJsonArray hiddenLayers;
    QJsonObject hiddenLayer;
    hiddenLayer["neurons"] = hiddenSize;
    hiddenLayer["activation"] = QString::fromStdString(layers[0].getActivationFunction());
    hiddenLayers.append(hiddenLayer);
    architecture["hidden_layers"] = hiddenLayers;

    architecture["output_neurons"] = outputSize;
    architecture["output_activation"] = QString::fromStdString(layers[1].getActivationFunction());

    json["architecture"] = architecture;

    // Save weights and biases
    QJsonObject weights;
    QJsonArray inputToHidden;
    for (int i = 0; i < layers[0].getOutputSize(); ++i) {
        QJsonArray row;
        for (int j = 0; j < layers[0].getInputSize(); ++j) {
            row.append(layers[0].getWeights()(i, j));
        }
        inputToHidden.append(row);
    }
    weights["input_to_hidden1"] = inputToHidden;

    QJsonArray hiddenToOutput;
    for (int i = 0; i < layers[1].getOutputSize(); ++i) {
        QJsonArray row;
        for (int j = 0; j < layers[1].getInputSize(); ++j) {
            row.append(layers[1].getWeights()(i, j));
        }
        hiddenToOutput.append(row);
    }
    weights["hidden1_to_output"] = hiddenToOutput;

    json["weights"] = weights;

    // Save biases
    QJsonObject biases;
    QJsonArray hidden1Biases;
    for (int i = 0; i < layers[0].getOutputSize(); ++i) {
        hidden1Biases.append(layers[0].getBiases()(i));
    }
    biases["hidden1"] = hidden1Biases;

    QJsonArray outputBiases;
    for (int i = 0; i < layers[1].getOutputSize(); ++i) {
        outputBiases.append(layers[1].getBiases()(i));
    }
    biases["output"] = outputBiases;

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

    QJsonArray hiddenLayers = architecture["hidden_layers"].toArray();
    if (hiddenLayers.size() != 1 || hiddenLayers[0].toObject()["neurons"].toInt() != hiddenSize) {
        return false;
    }

    // Load weights
    QJsonObject weights = json["weights"].toObject();
    QJsonArray inputToHidden = weights["input_to_hidden1"].toArray();
    Eigen::MatrixXf inputToHiddenWeights(hiddenSize, inputSize);
    for (int i = 0; i < hiddenSize; ++i) {
        QJsonArray row = inputToHidden[i].toArray();
        for (int j = 0; j < inputSize; ++j) {
            inputToHiddenWeights(i, j) = row[j].toDouble();
        }
    }
    layers[0].setWeights(inputToHiddenWeights);

    QJsonArray hiddenToOutput = weights["hidden1_to_output"].toArray();
    Eigen::MatrixXf hiddenToOutputWeights(outputSize, hiddenSize);
    for (int i = 0; i < outputSize; ++i) {
        QJsonArray row = hiddenToOutput[i].toArray();
        for (int j = 0; j < hiddenSize; ++j) {
            hiddenToOutputWeights(i, j) = row[j].toDouble();
        }
    }
    layers[1].setWeights(hiddenToOutputWeights);

    // Load biases
    QJsonObject biases = json["biases"].toObject();
    QJsonArray hidden1Biases = biases["hidden1"].toArray();
    Eigen::VectorXf hidden1BiasesVector(hiddenSize);
    for (int i = 0; i < hiddenSize; ++i) {
        hidden1BiasesVector(i) = hidden1Biases[i].toDouble();
    }
    layers[0].setBiases(hidden1BiasesVector);

    QJsonArray outputBiases = biases["output"].toArray();
    Eigen::VectorXf outputBiasesVector(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        outputBiasesVector(i) = outputBiases[i].toDouble();
    }
    layers[1].setBiases(outputBiasesVector);

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

    // Write magic number and version
    stream << MAGIC_NUMBER << FORMAT_VERSION;

    // Create JSON metadata
    QJsonObject metadata;

    // Save architecture
    QJsonObject architecture;
    architecture["input_neurons"] = inputSize;

    QJsonArray hiddenLayers;
    QJsonObject hiddenLayer;
    hiddenLayer["neurons"] = hiddenSize;
    hiddenLayer["activation"] = QString::fromStdString(layers[0].getActivationFunction());
    hiddenLayers.append(hiddenLayer);
    architecture["hidden_layers"] = hiddenLayers;

    architecture["output_neurons"] = outputSize;
    architecture["output_activation"] = QString::fromStdString(layers[1].getActivationFunction());

    metadata["architecture"] = architecture;
    metadata["data_precision"] = "float";

    // Convert metadata to JSON string
    QJsonDocument doc(metadata);
    QByteArray jsonData = doc.toJson(QJsonDocument::Compact);

    // Write metadata length and metadata
    stream << static_cast<quint32>(jsonData.size());
    stream.writeRawData(jsonData.constData(), jsonData.size());

    // Write weights and biases as binary data

    // Layer 0 weights (input to hidden)
    const Eigen::MatrixXf& inputToHiddenWeights = layers[0].getWeights();
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            stream << static_cast<float>(inputToHiddenWeights(i, j));
        }
    }

    // Layer 0 biases (hidden)
    const Eigen::VectorXf& hiddenBiases = layers[0].getBiases();
    for (int i = 0; i < hiddenSize; ++i) {
        stream << static_cast<float>(hiddenBiases(i));
    }

    // Layer 1 weights (hidden to output)
    const Eigen::MatrixXf& hiddenToOutputWeights = layers[1].getWeights();
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            stream << static_cast<float>(hiddenToOutputWeights(i, j));
        }
    }

    // Layer 1 biases (output)
    const Eigen::VectorXf& outputBiases = layers[1].getBiases();
    for (int i = 0; i < outputSize; ++i) {
        stream << static_cast<float>(outputBiases(i));
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

    if (magicNumber != MAGIC_NUMBER || formatVersion != FORMAT_VERSION) {
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

    QJsonArray hiddenLayers = architecture["hidden_layers"].toArray();
    if (hiddenLayers.size() != 1 || hiddenLayers[0].toObject()["neurons"].toInt() != hiddenSize) {
        file.close();
        return false;
    }

    // Verify data precision
    QString dataPrecision = metadata["data_precision"].toString();
    if (dataPrecision != "float") {
        file.close();
        return false;
    }

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

    file.close();
    return true;
}
