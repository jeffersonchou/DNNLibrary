//
// Created by daquexian on 5/21/18.
// For test.py in dnntools
//

#include <string>
#include <sstream>
#include <istream>
#include <fstream>
#include <iostream>
#include <chrono>

#include "android_log_helper.h"
#include "ModelBuilder.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using std::string; using std::cout; using std::endl;

// ./dnn_save_result daqName outputBlob [image]
int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        return -1;
    }
    string daqName = argv[1];
    string outputBlob = argv[2];
    bool useImage = argc == 4;

    ModelBuilder builder;
    builder.init();
    try {
        builder.readFromFile(daqName);
    } catch (string& str) {
        LOGE("Exception: %s", str.c_str());
    }
    builder.addIndexIntoOutput(builder.getBlobIndex(outputBlob));
    int ret = builder.compile(ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);

    Model model;
    auto inputDim = builder.getBlobDim(builder.getInputIndexes()[0]);
    auto inputLen = inputDim[1] * inputDim[2] * inputDim[3];
    float data[inputLen];
    if (useImage) {
        int width, height, channels;
        stbi_uc* rgb_img = stbi_load(argv[3], &width, &height, &channels, inputDim[3]);
        for (int i = 0; i < inputLen; i++) {
            data[i] = static_cast<float>(rgb_img[i]);
        }
    } else {
        for (int i = 0; i < inputLen; i++) {
            data[i] = i;
        }
    }

    const std::vector<uint32_t> &outputDim = builder.getBlobDim(builder.getOutputIndexes()[0]);
    uint32_t outputLen = product(outputDim);

    float output[outputLen];

    builder.prepareForExecution(model);
    builder.getInputIndexes()[0];
    builder.setInputBuffer(model, builder.getInputIndexes()[0], data, sizeof(data));
    builder.setOutputBuffer(model, builder.getOutputIndexes()[0], output, sizeof(output));
    model.predict();
    std::ofstream ofs("/data/local/tmp/result");
    for (int i = 0; i < outputLen; i++) {
        ofs << output[i] << endl;
    }

    builder.clear();
}
