#include <jni.h>
#include <string>
#include <vector>
#include <numeric>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <android/log.h>
#include <string.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "ModelBuilder.h"
#include <sstream>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedMacroInspection"

using namespace std;

#define  LOG_TAG    "NNAPI Demo"

#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG,__VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)

#define LENGTH(x) sizeof((x)) / sizeof((x)[0])

jint throwException( JNIEnv *env, string message );

int getMaxIndex(float arr[], int length);

template <typename T>
std::string to_string(T value);

ModelBuilder builder;

extern "C"
JNIEXPORT void
JNICALL
Java_me_daquexian_nnapiexample_MainActivity_initModel(
        JNIEnv *env,
        jobject /* this */,
        jobject javaAssetManager) {

    AAssetManager *mgrr = AAssetManager_fromJava(env, javaAssetManager);
    if (builder.init(mgrr) != ANEURALNETWORKS_NO_ERROR) {
        throwException(env, "Create model error");

    }

    uint32_t data = builder.addInput(28, 28);
    uint32_t conv1 = builder.addConv("conv1", data, 1, 1, 0, 0, 5, 5, ModelBuilder::ACTIVATION_NONE, 20);
    uint32_t pool1 = builder.addPool(conv1, 2, 2, 0, 0, 2, 2, ModelBuilder::ACTIVATION_NONE,
                                     ModelBuilder::MAX_POOL);
    uint32_t conv2 = builder.addConv("conv2", pool1, 1, 1, 0, 0, 5, 5, ModelBuilder::ACTIVATION_NONE, 50);
    uint32_t pool2 = builder.addPool(conv2, 2, 2, 0, 0, 2, 2, ModelBuilder::ACTIVATION_NONE,
                                     ModelBuilder::MAX_POOL);
    uint32_t ip1 = builder.addFC("ip1", pool2, 500, ModelBuilder::ACTIVATION_RELU);
    uint32_t ip2 = builder.addFC("ip2", ip1, 10, ModelBuilder::ACTIVATION_NONE);

    uint32_t prob = builder.addSoftMax(ip2);

    builder.addIndexIntoOutput(prob);

    int ret;
    if ((ret = builder.compile(ModelBuilder::PREFERENCE_SUSTAINED_SPEED)) !=
            ANEURALNETWORKS_NO_ERROR) {
        throwException(env, "Create model error, code: " + to_string(ret));
    }
}


extern "C"
JNIEXPORT jint
JNICALL
Java_me_daquexian_nnapiexample_MainActivity_predict(
        JNIEnv *env,
        jobject /* this */,
        jfloatArray dataArrayObject) {
    jfloat *data = env->GetFloatArrayElements(dataArrayObject, nullptr);
    jsize len = env->GetArrayLength(dataArrayObject);

    Model model = builder.prepareForExecution();
    builder.setInputBuffer(model, builder.getInputIndexes()[0], data, static_cast<size_t>(len));

    float prob[10];
    builder.setOutputBuffer(model, builder.getOutputIndexes()[0], prob, sizeof(prob));

    model.predict();

    for (auto value : prob) {
        LOGD("prob: %f", value);
    }

    return getMaxIndex(prob, LENGTH(prob));
}

extern "C"
JNIEXPORT void
JNICALL
Java_me_daquexian_nnapiexample_MainActivity_clearModel(
        JNIEnv *env,
        jobject /* this */) {
    builder.clear();
}

int getMaxIndex(float arr[], int length) {
    int maxIndex = 0;
    auto max = arr[0];
    for (int i = 1; i < length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}


jint throwException(JNIEnv *env, std::string message) {
    jclass exClass;
    std::string className = "java/lang/RuntimeException" ;

    exClass = env->FindClass(className.c_str());

    return env->ThrowNew(exClass, message.c_str());
}

template<typename T>
string to_string(T value) {
    ostringstream os ;
    os << value ;
    return os.str() ;
}



#pragma clang diagnostic pop