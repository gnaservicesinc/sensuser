#include "noodlenet_backend.h"
#include <QTemporaryFile>
#include <QDir>

NoodleNetBackend::NoodleNetBackend() : model(nullptr) {}

NoodleNetBackend::~NoodleNetBackend() {
    if (model) {
        nn_model_free(model);
        model = nullptr;
    }
}

bool NoodleNetBackend::createModel(int inputNeurons,
                                   const std::vector<int>& hiddenSizes,
                                   const std::vector<std::string>& hiddenActivations,
                                   ActivationFunction outputAct) {
    std::vector<ActivationFunction> acts(hiddenSizes.size(), NN_ACTIVATION_FUNCTION_SIGMOID);
    for (size_t i = 0; i < hiddenActivations.size() && i < acts.size(); ++i) {
        acts[i] = parseActivation(hiddenActivations[i]);
    }
    if (model) { nn_model_free(model); model = nullptr; }
    model = nn_model_create(inputNeurons,
                            hiddenSizes.empty() ? nullptr : hiddenSizes.data(),
                            static_cast<int>(hiddenSizes.size()),
                            1,
                            acts.empty() ? nullptr : acts.data(),
                            outputAct);
    return model != nullptr;
}

bool NoodleNetBackend::loadModel(const QString& path) {
    if (model) { nn_model_free(model); model = nullptr; }
    model = nn_model_load(path.toUtf8().constData());
    return model != nullptr;
}

bool NoodleNetBackend::saveModel(const QString& path) const {
    if (!model) return false;
    return nn_model_save(model, path.toUtf8().constData()) == 0;
}

float NoodleNetBackend::predict(const QImage& image) const {
    if (!model) return -1.0f;
    QString tmpPath = writeTempPNG(image);
    if (tmpPath.isEmpty()) return -1.0f;
    float p = 0.0f;
    int rc = nn_model_predict_image(model, tmpPath.toUtf8().constData(), &p);
    QFile::remove(tmpPath);
    if (rc != 0) return -1.0f;
    return p;
}

bool NoodleNetBackend::setOptimizer(OptimizerType opt, float beta1, float beta2, float epsilon) {
    if (!model) return false;
    NN_Optimizer o = NN_OPTIMIZER_SGD;
    switch (opt) {
        case OptimizerType::RMSprop: o = NN_OPTIMIZER_RMSPROP; break;
        case OptimizerType::Adam: o = NN_OPTIMIZER_ADAM; break;
        case OptimizerType::SGD:
        default: o = NN_OPTIMIZER_SGD; break;
    }
    return nn_model_set_optimizer(model, o, beta1, beta2, epsilon) == 0;
}

bool NoodleNetBackend::trainFromDirs(const QString& posDir,
                                     const QString& negDir,
                                     const QString& valDir,
                                     int steps,
                                     int batchSize,
                                     bool shuffle,
                                     float learningRate,
                                     float l1,
                                     float l2,
                                     float& lastLoss) {
    if (!model) return false;
    float outLoss = 0.0f; float outVal = 0.0f;
    int rc = nn_train_from_dirs(model,
                                posDir.isEmpty() ? nullptr : posDir.toUtf8().constData(),
                                negDir.isEmpty() ? nullptr : negDir.toUtf8().constData(),
                                valDir.isEmpty() ? nullptr : valDir.toUtf8().constData(),
                                steps,
                                batchSize,
                                shuffle ? 1 : 0,
                                learningRate,
                                l1,
                                l2,
                                &outLoss,
                                &outVal);
    if (rc != 0) return false;
    lastLoss = outLoss;
    return true;
}

bool NoodleNetBackend::evaluateDirs(const QString& posDir,
                                    const QString& negDir,
                                    float& accuracy,
                                    int& truePos,
                                    int& trueNeg,
                                    int& falsePos,
                                    int& falseNeg) const {
    if (!model) return false;
    float acc = 0.0f; int tp=0, tn=0, fp=0, fn=0;
    int rc = nn_evaluate_dirs(model,
                              posDir.isEmpty() ? nullptr : posDir.toUtf8().constData(),
                              negDir.isEmpty() ? nullptr : negDir.toUtf8().constData(),
                              &tp, &tn, &fp, &fn, &acc);
    if (rc != 0) return false;
    accuracy = acc; truePos = tp; trueNeg = tn; falsePos = fp; falseNeg = fn;
    return true;
}

int NoodleNetBackend::inputSize() const { return model ? nn_model_input_size(model) : -1; }
int NoodleNetBackend::outputSize() const { return model ? nn_model_output_size(model) : -1; }
int NoodleNetBackend::numHidden() const { return model ? nn_model_num_hidden(model) : -1; }
int NoodleNetBackend::hiddenSize(int index) const { return model ? nn_model_hidden_size(model, index) : -1; }
ActivationFunction NoodleNetBackend::hiddenActivation(int index) const { return model ? nn_model_hidden_activation(model, index) : NN_ACTIVATION_FUNCTION_SIGMOID; }
ActivationFunction NoodleNetBackend::outputActivation() const { return model ? nn_model_output_activation(model) : NN_ACTIVATION_FUNCTION_SIGMOID; }

ActivationFunction NoodleNetBackend::parseActivation(const std::string& name) {
    if (name == "tanh") return NN_ACTIVATION_FUNCTION_TANH;
    if (name == "relu") return NN_ACTIVATION_FUNCTION_RELU;
    if (name == "leaky_relu") return NN_ACTIVATION_FUNCTION_LEAKY_RELU;
    return NN_ACTIVATION_FUNCTION_SIGMOID;
}

QString NoodleNetBackend::writeTempPNG(const QImage& image) {
    // Use QTemporaryFile to create a temp PNG
    QTemporaryFile tmp(QDir::tempPath() + "/nn_img_XXXXXX.png");
    tmp.setAutoRemove(false);
    if (!tmp.open()) return QString();
    QString path = tmp.fileName();
    tmp.close();
    QImage toSave = image;
    if (toSave.isNull()) return QString();
    // Ensure it is saved as a PNG
    if (!toSave.save(path, "PNG")) {
        QFile::remove(path);
        return QString();
    }
    return path;
}

int NoodleNetBackend::numWeightLayers() const {
    if (!model) return -1;
    return nn_num_weight_layers(model);
}

bool NoodleNetBackend::layerDims(int layerIndex, int& in, int& out) const {
    if (!model) return false;
    return nn_layer_dims(model, layerIndex, &in, &out) == 0;
}

bool NoodleNetBackend::computeActivations(const QImage& image, int layerIndex, std::vector<float>& out) const {
    if (!model) return false;
    QString tmp = writeTempPNG(image);
    if (tmp.isEmpty()) return false;
    int in=0, outn=0;
    if (nn_layer_dims(model, layerIndex == 0 ? 0 : layerIndex, &in, &outn) != 0) { QFile::remove(tmp); return false; }
    out.resize(layerIndex == 0 ? inputSize() : outn);
    int rc = nn_compute_activations_from_image(model, tmp.toUtf8().constData(), layerIndex, out.data(), out.size());
    QFile::remove(tmp);
    return rc == 0;
}

bool NoodleNetBackend::computePreActivations(const QImage& image, int layerIndex, std::vector<float>& out) const {
    if (!model) return false;
    QString tmp = writeTempPNG(image);
    if (tmp.isEmpty()) return false;
    int in=0, outn=0;
    if (nn_layer_dims(model, layerIndex == 0 ? 0 : layerIndex-1, &in, &outn) != 0) { QFile::remove(tmp); return false; }
    out.resize(outn);
    int rc = nn_compute_pre_activations_from_image(model, tmp.toUtf8().constData(), layerIndex, out.data(), out.size());
    QFile::remove(tmp);
    return rc == 0;
}

bool NoodleNetBackend::exportVisualizations(const QString& outDir,
                                            NN_VisMode mode,
                                            NN_VisScale scale,
                                            bool includeBias,
                                            bool includeStats,
                                            bool rawWeightsFull,
                                            int onlyLayer) const {
    if (!model) return false;
    NN_VisOptions opts; opts.mode = mode; opts.scale = scale; opts.include_bias = includeBias ? 1 : 0; opts.include_stats = includeStats ? 1 : 0; opts.raw_weights_full = rawWeightsFull ? 1 : 0; opts.only_layer = onlyLayer;
    int rc = nn_export_layer_visualizations_ex(model, outDir.toUtf8().constData(), &opts);
    return rc == 0;
}

bool NoodleNetBackend::renderHiddenLayerVisualization(int layerIndex,
                                                      NN_VisMode mode,
                                                      NN_VisScale scale,
                                                      bool rawWeightsFull,
                                                      QImage& out) const {
    if (!model) return false;
    NN_VisOptions opts; opts.mode = mode; opts.scale = scale; opts.include_bias = 0; opts.include_stats = 0; opts.raw_weights_full = rawWeightsFull ? 1 : 0; opts.only_layer = layerIndex;
    unsigned char* pixels = nullptr; int w=0,h=0;
    int rc = nn_render_hidden_layer_visualization(model, layerIndex, &opts, &pixels, &w, &h);
    if (rc != 0 || !pixels || w<=0 || h<=0) { if (pixels) free(pixels); return false; }
    // QImage takes a copy when constructed from const uchar*, so we can free pixels after
    out = QImage((const uchar*)pixels, w, h, QImage::Format_Grayscale8).copy();
    free(pixels);
    return true;
}

bool NoodleNetBackend::setDataDirs(const QString& posDir, const QString& negDir, const QString& valDir) {
    if (!model) return false;
    return nn_model_set_data_dirs(model,
                                  posDir.isEmpty()?nullptr:posDir.toUtf8().constData(),
                                  negDir.isEmpty()?nullptr:negDir.toUtf8().constData(),
                                  valDir.isEmpty()?nullptr:valDir.toUtf8().constData()) == 0;
}

bool NoodleNetBackend::getDataDirs(QString& posDir, QString& negDir, QString& valDir) const {
    if (!model) return false;
    const char *p=nullptr, *n=nullptr, *v=nullptr;
    if (nn_model_get_data_dirs(model, &p, &n, &v) != 0) return false;
    posDir = p ? QString::fromUtf8(p) : QString();
    negDir = n ? QString::fromUtf8(n) : QString();
    valDir = v ? QString::fromUtf8(v) : QString();
    return true;
}

bool NoodleNetBackend::setLockedTrainingParams(int batchSize, float learningRate, bool shuffle, OptimizerType opt) {
    if (!model) return false;
    NN_Optimizer o = NN_OPTIMIZER_SGD;
    switch (opt) { case OptimizerType::RMSprop: o=NN_OPTIMIZER_RMSPROP; break; case OptimizerType::Adam: o=NN_OPTIMIZER_ADAM; break; default: o=NN_OPTIMIZER_SGD; break; }
    return nn_model_set_locked_training_params(model, batchSize, learningRate, shuffle?1:0, o) == 0;
}

bool NoodleNetBackend::getLockedTrainingParams(int& batchSize, float& learningRate, bool& shuffle, OptimizerType& opt) const {
    if (!model) return false;
    NN_Optimizer o = NN_OPTIMIZER_SGD; int sh=0; int bs=0; float lr=0.0f;
    if (nn_model_get_locked_training_params(model, &bs, &lr, &sh, &o) != 0) return false;
    batchSize = bs; learningRate = lr; shuffle = (sh!=0);
    switch (o) { case NN_OPTIMIZER_RMSPROP: opt = OptimizerType::RMSprop; break; case NN_OPTIMIZER_ADAM: opt = OptimizerType::Adam; break; default: opt = OptimizerType::SGD; break; }
    return true;
}

void NoodleNetBackend::resetModel() {
    if (model) {
        nn_model_free(model);
        model = nullptr;
    }
}
