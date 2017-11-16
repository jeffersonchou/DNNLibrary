//
// Created by daquexian on 2017/11/12.
//

#include <jni.h>
#include <string>
#include <vector>
#include <android/NeuralNetworks.h>

using std::string; using std::vector;

extern "C"
JNIEXPORT jfloatArray
JNICALL
Java_me_daquexian_dnnlibrary_ModelWrapper_init(
        JNIEnv *env,
        jobject /* this */) {

    ANeuralNetworksModel *model;

    ANeuralNetworksModel_create(&model);
    int ret;

    vector<uint32_t> inputDim{1, 2, 3, 4};
    ANeuralNetworksOperandType inputType;
    inputType.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    inputType.scale = 0.f;
    inputType.zeroPoint = 0;
    inputType.dimensionCount = static_cast<uint32_t>(inputDim.size());
    inputType.dimensions = &inputDim[0];
    if ((ret = ANeuralNetworksModel_addOperand(model, &inputType)) != ANEURALNETWORKS_NO_ERROR) {   // operand 0
        throw "Wrong return value";
    }
    if ((ret = ANeuralNetworksModel_addOperand(model, &inputType)) != ANEURALNETWORKS_NO_ERROR) {   // operand 1
        throw "Wrong return value";
    }

    ANeuralNetworksOperandType int32Type;
    int32Type.type = ANEURALNETWORKS_INT32;
    int32Type.scale = 0.f;
    int32Type.zeroPoint = 0;
    int32Type.dimensionCount = 0;
    int32Type.dimensions = nullptr;
    if ((ret = ANeuralNetworksModel_addOperand(model, &int32Type)) != ANEURALNETWORKS_NO_ERROR) {   // operand 2
        throw "Wrong return value";
    }
    int32_t axis = 3;
    if ((ret = ANeuralNetworksModel_setOperandValue(model, 2, &axis, sizeof(axis)) != ANEURALNETWORKS_NO_ERROR)) {
        throw "Wrong return value";
    }

    vector<uint32_t> outputDim{1, 2, 3, 8};
    ANeuralNetworksOperandType outputType;
    outputType.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    outputType.scale = 0.f;
    outputType.zeroPoint = 0;
    outputType.dimensionCount = static_cast<uint32_t>(outputDim.size());
    outputType.dimensions = &outputDim[0];

    if ((ret = ANeuralNetworksModel_addOperand(model, &outputType)) != ANEURALNETWORKS_NO_ERROR) {   // operand 3
        throw "Wrong return value";
    }

    uint32_t concatInputIndexes[3]{0, 1, 2};
    uint32_t outputIndexes[1]{3};

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONCATENATION,
                                      3, concatInputIndexes, 1, outputIndexes);

    uint32_t inputIndexes[2]{0, 1};
    if ((ret = ANeuralNetworksModel_identifyInputsAndOutputs(
            model,
            2, inputIndexes,
            1, outputIndexes)) != ANEURALNETWORKS_NO_ERROR) {

        throw "Wrong return value";
    }

    ret = ANeuralNetworksModel_finish(model);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw "Wrong return value";
    }

    ANeuralNetworksCompilation *compilation = nullptr;

    ret = ANeuralNetworksCompilation_create(model, &compilation);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw "Wrong return value";
    }

    ret = ANeuralNetworksCompilation_setPreference(compilation, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw "Wrong return value";
    }

    ret = ANeuralNetworksCompilation_finish(compilation);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw "Wrong return value";
    }

    ANeuralNetworksExecution *execution = nullptr;
    ANeuralNetworksExecution_create(compilation, &execution);

    const size_t len = 1 * 2 * 3 * 4;

    float data1[len];
    float data2[len];

    for (size_t i = 0; i < len; i++) {
        data1[i] = 100000.5f;
        data2[i] = -99999.5f;
    }

    ANeuralNetworksExecution_setInput(execution, 0, nullptr, data1, len);
    ANeuralNetworksExecution_setInput(execution, 1, nullptr, data2, len);

    float output[2 * len];
    ANeuralNetworksExecution_setOutput(execution, 0, nullptr, output, 2 * len);

    ANeuralNetworksEvent* event = nullptr;
    if ((ret = ANeuralNetworksExecution_startCompute(execution, &event)) != ANEURALNETWORKS_NO_ERROR) {
        throw "Wrong return value";
    }

    if ((ret = ANeuralNetworksEvent_wait(event)) != ANEURALNETWORKS_NO_ERROR) {
        throw "Wrong return value";
    }

    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);

    jfloatArray result = env->NewFloatArray(2 * len);
    env->SetFloatArrayRegion(result, 0, 2 * len, output);

    return result;
}

__attribute__((weak))
extern "C" int ANeuralNetworksModel_setInputsAndOutputs(
        ANeuralNetworksModel *model, uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
        const uint32_t* outputs);

extern "C" int ANeuralNetworksModel_identifyInputsAndOutputs(
        ANeuralNetworksModel *model, uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
        const uint32_t* outputs) {
    return ANeuralNetworksModel_setInputsAndOutputs(model,
            inputCount, inputs, outputCount, outputs);
}

