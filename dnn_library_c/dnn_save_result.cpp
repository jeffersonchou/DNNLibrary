//
// Created by daquexian on 5/21/18.
//

#include <string>
#include <sstream>
#include <istream>
#include <fstream>
#include <iostream>
#include <chrono>

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
    builder.readFromFile(daqName);
    cout << builder.getBlobIndex("data_bak") << endl;
    // builder.addIndexIntoOutput(builder.getBlobIndex("data_bak"));
    builder.addIndexIntoOutput(builder.getBlobIndex("conv1"));
    int ret = builder.compile(ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    cout << ModelBuilder::getErrorProcedure(ret) << endl;

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
            data[i] = 1;
        }
    }
    uint32_t outputLen = product(builder.getBlobDim(builder.getOutputIndexes()[0]));

    cout << outputLen << endl;

    float output[outputLen];

    builder.prepareForExecution(model);
    cout << builder.getInputIndexes()[0] << endl;
    cout << builder.setInputBuffer(model, builder.getInputIndexes()[0], data, sizeof(data)) << endl;
    cout << builder.setOutputBuffer(model, builder.getOutputIndexes()[0], output, sizeof(output)) << endl;
    cout << model.predict() << endl;
    cout << "----" << endl;
    for (int i = 0; i < 10; i++) {
        cout << output[outputLen - 10 + i] << endl;
    }
}
