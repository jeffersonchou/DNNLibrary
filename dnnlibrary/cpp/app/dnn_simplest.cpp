//
// Created by daquexian on 5/21/18.
//

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <chrono>

#include "ModelBuilder.h"

using std::string; using std::cout; using std::endl; using std::vector;
typedef std::chrono::high_resolution_clock Clock;

int main(int argc, char** argv) {
    ModelBuilder builder;
    cout << builder.init() << endl;
    auto input = builder.addInput(5, 3, 4);
    vector<int32_t> starts{0, 0, 0, 0};
    vector<int32_t> ends{0, 0, 0, 2};
    vector<int32_t> strides{1, 1, 1, 1};
    uint32_t beginMask = 7;//14;
    uint32_t endMask = 7;//14;
    uint32_t shrinkMask = 0;
    auto index = builder.addStridedSlice(input, starts, ends, strides, beginMask, endMask, shrinkMask);
    builder.addIndexIntoOutput(index);
    int ret = builder.compile(ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    cout << ModelBuilder::getErrorProcedure(ret) << endl;
    cout << ModelBuilder::getErrorCause(ret) << endl;
    Model model;
    uint32_t outputLen = product(builder.getBlobDim(builder.getOutputIndexes()[0]));

    for (auto dim: builder.getBlobDim(builder.getOutputIndexes()[0])) {
        cout << dim << ", ";
    }
    cout << endl;
    cout << outputLen << endl;

    const uint32_t inputLen = product(builder.getBlobDim(builder.getInputIndexes()[0]));
    float data[inputLen];
    for (int i = 0; i < inputLen; i++) {
        data[i] = i;
    }
    float output[outputLen];

    builder.prepareForExecution(model);
    builder.getInputIndexes()[0];
    builder.setInputBuffer(model, builder.getInputIndexes()[0], data, sizeof(data));
    builder.setOutputBuffer(model, builder.getOutputIndexes()[0], output, sizeof(output));
    model.predict();
    for (int i = 0; i < outputLen; i++) {
        cout << output[i] << endl;
    }

    builder.clear();
    return 0;
}
