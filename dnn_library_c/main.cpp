//
// Created by daquexian on 5/10/18.
//

#include "ModelBuilder.h"

#include <string>
#include <sstream>
#include <istream>
#include <fstream>
#include <iostream>
#include <chrono>

using std::string; using std::cout; using std::endl;
typedef std::chrono::high_resolution_clock Clock;

int main(int argc, char** argv) {
    string daqName = "/data/local/tmp/daq/resnet18.daq";
    if (argc >= 3) {
        daqName = argv[2];
    }
    std::ifstream ifs(daqName);

    // read whole content of a file into a string
    string str(static_cast<std::stringstream const&>(std::stringstream() << ifs.rdbuf()).str());

    uint32_t preference = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER;
    if (argc >= 2) {
        preference = static_cast<uint32_t>(std::stoi(string(argv[1])));
        if (preference > 2) {
            preference = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER;
        }
    }
    cout << "preference is " << preference << endl;

    ModelBuilder builder;
    builder.init();
    builder.readFromBuffer(str.c_str());
    builder.addIndexIntoOutput(builder.getBlobIndex("prob"));
    builder.compile(preference);
    Model model;
    auto inputDim = builder.getBlobDim(builder.getInputIndexes()[0]);
    float data[inputDim[1] * inputDim[2] * inputDim[3]];
    uint32_t outputLen = product(builder.getBlobDim(builder.getOutputIndexes()[0]));

    float output[outputLen];

#define WARM_UP 5
    for (int i = 0; i < WARM_UP; i++) {
        builder.prepareForExecution(model);
        builder.setInputBuffer(model, builder.getInputIndexes()[0], data, sizeof(data));
        builder.setOutputBuffer(model, builder.getOutputIndexes()[0], output, sizeof(output));
        model.predict();
    }
#define RUNS 100
    auto t1 = Clock::now();
    for (int i = 0; i < RUNS; i++) {
        builder.prepareForExecution(model);
        builder.setInputBuffer(model, builder.getInputIndexes()[0], data, sizeof(data));
        builder.setOutputBuffer(model, builder.getOutputIndexes()[0], output, sizeof(output));
        model.predict();
    }
    auto t2 = Clock::now();

    cout << "time: " << (1. * std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / RUNS) << " microseconds." << endl;
}
