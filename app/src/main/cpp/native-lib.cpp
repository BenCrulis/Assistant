#include <jni.h>
#include <stdlib.h>

static void* buf = NULL;

extern "C"
JNIEXPORT void JNICALL
Java_com_example_androidassistant_utils_alloc_NativeLib_allocateMemory(JNIEnv* env, jobject obj, jint size) {
    buf = malloc(size);
    // Optionally initialize memory
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_androidassistant_utils_alloc_NativeLib_freeMemory(JNIEnv* env, jobject obj) {
    free(buf);
    buf = NULL;
}


extern "C"
JNIEXPORT jobject JNICALL Java_com_example_androidassistant_utils_alloc_NativeByteBuffer_createBuffer(JNIEnv *env, jclass cls, jint size) {
    void* buffer = malloc(size);
    if (buffer == NULL) {
        return NULL;
    }
    void* address = (void*) buffer;
    jobject byteBuffer = env->NewDirectByteBuffer(address, size);
    if (byteBuffer == NULL) {
        // Handle error
        return NULL;
    }
    return byteBuffer;
}

extern "C"
JNIEXPORT void JNICALL Java_com_example_androidassistant_utils_alloc_NativeByteBuffer_deallocateBuffer(JNIEnv *env, jclass cls, jobject buffer) {
    void* address = env->GetDirectBufferAddress(buffer);
    int size = env->GetDirectBufferCapacity(buffer);
    free(address);
}
