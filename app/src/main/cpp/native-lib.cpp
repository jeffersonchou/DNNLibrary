#include <jni.h>
#include <string>
#include <vector>
#include <android/NeuralNetworks.h>

using namespace std;

extern "C"
jint throwException( JNIEnv *env, char *message );
ANeuralNetworksOperandType getFloat32OperandTypeWithDims(std::vector<uint32_t> &dims);


JNIEXPORT jstring

JNICALL
Java_me_daquexian_nnapiexample_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";


    ANeuralNetworksModel* model = nullptr;
    if (ANeuralNetworksModel_create(&model) != ANEURALNETWORKS_NO_ERROR) {
        throwException(env, "Create model error");
    }

    vector<uint32_t> dataDims{1, 28, 28, 1};
    vector<uint32_t> conv1BlobDims{1, 24, 24, 20};

    ANeuralNetworksOperandType dataType = getFloat32OperandTypeWithDims(dataDims);

    ANeuralNetworksOperandType conv1BlobType = getFloat32OperandTypeWithDims(conv1BlobDims);

    ANeuralNetworksOperandType strideOneType;
    strideOneType.type = ANEURALNETWORKS_INT32;
    strideOneType.scale = 0.f;
    strideOneType.zeroPoint = 0;
    strideOneType.dimensionCount = 0;
    strideOneType.dimensions = NULL;

    // Now we add the seven operands, in the same order defined in the diagram.
    ANeuralNetworksModel_addOperand(model, &dataType);  // operand 0
    ANeuralNetworksModel_addOperand(model, &conv1BlobType);  // operand 1

    return env->NewStringUTF(hello.c_str());

}

ANeuralNetworksOperandType getFloat32OperandTypeWithDims(std::vector<uint32_t> &dims) {
    ANeuralNetworksOperandType type;
    type.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    type.scale = 0.f;    // These fields are useful for quantized tensors.
    type.zeroPoint = 0;  // These fields are useful for quantized tensors.
    type.dimensionCount = static_cast<uint32_t>(dims.size());
    type.dimensions = &dims[0];

    return type;
}

jint throwException(JNIEnv *env, std::string message) {
    jclass exClass;
    std::string className = "java/lang/RuntimeException" ;

    exClass = env->FindClass(className.c_str());

    return env->ThrowNew(exClass, message.c_str());
}
