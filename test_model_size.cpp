#include "mlp.h"
#include <QCoreApplication>
#include <QFile>
#include <QFileInfo>
#include <QDebug>
#include <QDir>

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    
    // Create a test MLP with 512x512 input, 128 hidden neurons, and 1 output neuron
    MLP mlp(512 * 512, 128, 1, "sigmoid", "sigmoid");
    
    // Save in JSON format
    QString jsonPath = "test_model.json";
    QJsonObject json = mlp.saveToJson();
    QJsonDocument doc(json);
    QByteArray jsonData = doc.toJson(QJsonDocument::Indented);
    
    QFile jsonFile(jsonPath);
    if (jsonFile.open(QIODevice::WriteOnly)) {
        jsonFile.write(jsonData);
        jsonFile.close();
        qDebug() << "Model saved in JSON format to" << QDir::currentPath() + "/" + jsonPath;
    } else {
        qDebug() << "Failed to save model in JSON format";
        return 1;
    }
    
    // Save in binary format
    QString binaryPath = "test_model.senm";
    if (mlp.saveToBinary(binaryPath)) {
        qDebug() << "Model saved in binary format to" << QDir::currentPath() + "/" + binaryPath;
    } else {
        qDebug() << "Failed to save model in binary format";
        return 1;
    }
    
    // Compare file sizes
    QFileInfo jsonInfo(jsonPath);
    QFileInfo binaryInfo(binaryPath);
    
    qint64 jsonSize = jsonInfo.size();
    qint64 binarySize = binaryInfo.size();
    
    qDebug() << "JSON file size:" << jsonSize << "bytes (" << jsonSize / (1024.0 * 1024.0) << "MB)";
    qDebug() << "Binary file size:" << binarySize << "bytes (" << binarySize / (1024.0 * 1024.0) << "MB)";
    qDebug() << "Size reduction:" << (1.0 - (double)binarySize / jsonSize) * 100.0 << "%";
    
    // Test loading the binary model
    MLP loadedMlp(512 * 512, 128, 1);
    if (loadedMlp.loadFromBinary(binaryPath)) {
        qDebug() << "Successfully loaded model from binary format";
    } else {
        qDebug() << "Failed to load model from binary format";
        return 1;
    }
    
    return 0;
}
