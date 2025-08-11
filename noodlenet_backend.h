#ifndef NOODLENET_BACKEND_H
#define NOODLENET_BACKEND_H

#include <QString>
#include <QImage>
#include <vector>
#include <string>

extern "C" {
#include <noodlenet.h>
#include "optimizertypes.h"
}

class NoodleNetBackend {
public:
    NoodleNetBackend();
    ~NoodleNetBackend();

    bool hasModel() const { return model != nullptr; }

    bool createModel(int inputNeurons,
                     const std::vector<int>& hiddenSizes,
                     const std::vector<std::string>& hiddenActivations,
                     ActivationFunction outputActivation);

    bool loadModel(const QString& path);
    bool saveModel(const QString& path) const;

    // Returns probability in [0,1], or negative on error
    float predict(const QImage& image) const;

    // Train for a number of steps; returns true on success and provides last loss
    bool trainFromDirs(const QString& posDir,
                       const QString& negDir,
                       const QString& valDir,
                       int steps,
                       int batchSize,
                       float learningRate,
                       float l1,
                       float l2,
                       float& lastLoss);

    // Evaluate on directories; returns true on success and sets metrics
    bool evaluateDirs(const QString& posDir,
                      const QString& negDir,
                      float& accuracy,
                      int& truePos,
                      int& trueNeg,
                      int& falsePos,
                      int& falseNeg) const;

    // Introspection
    int inputSize() const;
    int outputSize() const;
    int numHidden() const;
    int hiddenSize(int index) const;
    ActivationFunction hiddenActivation(int index) const;
    ActivationFunction outputActivation() const;

    // Optimizer
    bool setOptimizer(OptimizerType opt, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    // Extra introspection helpers
    int numWeightLayers() const; // includes output layer
    bool layerDims(int layerIndex, int& in, int& out) const;
    bool computeActivations(const QImage& image, int layerIndex, std::vector<float>& out) const; // layerIndex: 0=input, ... L=output
    bool computePreActivations(const QImage& image, int layerIndex, std::vector<float>& out) const; // layerIndex: 1..L

    // Export visualizations using libnoodlenet extended API
    bool exportVisualizations(const QString& outDir,
                              NN_VisMode mode,
                              NN_VisScale scale,
                              bool includeBias,
                              bool includeStats,
                              bool rawWeightsFull) const;

private:
    static ActivationFunction parseActivation(const std::string& name);
    static QString writeTempPNG(const QImage& image);

    NN_Model* model;
};

#endif // NOODLENET_BACKEND_H
