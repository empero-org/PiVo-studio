/* JNI bridge for pivo_runner — drop into app/src/main/cpp/pivo_jni.c and link
 * against libpivo_runner.so. See docs/MOBILE.md for the Gradle recipe. */
#include <jni.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "pivo_ffi.h"

static jstring to_jstring(JNIEnv *env, const char *s) {
    return s ? (*env)->NewStringUTF(env, s) : NULL;
}

JNIEXPORT jlong JNICALL Java_org_empero_pivo_PivoNative_loadModel(JNIEnv *env, jclass, jstring path) {
    const char *p = (*env)->GetStringUTFChars(env, path, NULL);
    jlong h = (jlong)(uintptr_t)pivo_load_model(p);
    (*env)->ReleaseStringUTFChars(env, path, p);
    return h;
}

JNIEXPORT jlong JNICALL Java_org_empero_pivo_PivoNative_loadBundle(JNIEnv *env, jclass, jbyteArray bytes) {
    jsize len = (*env)->GetArrayLength(env, bytes);
    jbyte *data = (*env)->GetByteArrayElements(env, bytes, NULL);
    jlong h = (jlong)(uintptr_t)pivo_load_bundle((const uint8_t *)data, (size_t)len);
    (*env)->ReleaseByteArrayElements(env, bytes, data, JNI_ABORT);
    return h;
}

JNIEXPORT void JNICALL Java_org_empero_pivo_PivoNative_freeEngine(JNIEnv *env, jclass, jlong engine) {
    pivo_free_engine((PivoEngine *)(uintptr_t)engine);
}

JNIEXPORT jint JNICALL Java_org_empero_pivo_PivoNative_sampleRate(JNIEnv *env, jclass, jlong engine) {
    return pivo_sample_rate((PivoEngine *)(uintptr_t)engine);
}

JNIEXPORT jstring JNICALL Java_org_empero_pivo_PivoNative_lastError(JNIEnv *env, jclass) {
    return to_jstring(env, pivo_last_error_message());
}

JNIEXPORT jbyteArray JNICALL Java_org_empero_pivo_PivoNative_synthesizeWav(
        JNIEnv *env, jclass, jlong engine, jstring text, jstring voice,
        jfloat speed, jboolean strict, jstring overridesJson) {
    const char *t = (*env)->GetStringUTFChars(env, text, NULL);
    const char *v = voice ? (*env)->GetStringUTFChars(env, voice, NULL) : NULL;
    const char *o = overridesJson ? (*env)->GetStringUTFChars(env, overridesJson, NULL) : NULL;
    PivoOptions opts = { v, (float)speed, (int)strict, o };
    size_t len = 0;
    unsigned char *wav = pivo_synthesize_wav(
        (PivoEngine *)(uintptr_t)engine, t, &opts, &len);
    if (t) (*env)->ReleaseStringUTFChars(env, text, t);
    if (v) (*env)->ReleaseStringUTFChars(env, voice, v);
    if (o) (*env)->ReleaseStringUTFChars(env, overridesJson, o);
    if (!wav) return NULL;
    jbyteArray out = (*env)->NewByteArray(env, (jsize)len);
    (*env)->SetByteArrayRegion(env, out, 0, (jsize)len, (const jbyte *)wav);
    pivo_free_buffer(wav, len);
    return out;
}

JNIEXPORT jlong JNICALL Java_org_empero_pivo_PivoNative_streamNew(
        JNIEnv *env, jclass, jlong engine, jstring text, jstring voice,
        jfloat speed, jboolean strict) {
    const char *t = (*env)->GetStringUTFChars(env, text, NULL);
    const char *v = voice ? (*env)->GetStringUTFChars(env, voice, NULL) : NULL;
    PivoOptions opts = { v, (float)speed, (int)strict, NULL };
    jlong h = (jlong)(uintptr_t)pivo_stream_new((PivoEngine *)(uintptr_t)engine, t, &opts);
    if (t) (*env)->ReleaseStringUTFChars(env, text, t);
    if (v) (*env)->ReleaseStringUTFChars(env, voice, v);
    return h;
}

JNIEXPORT jint JNICALL Java_org_empero_pivo_PivoNative_streamNextFrame(
        JNIEnv *env, jclass, jlong stream, jfloatArray out) {
    jsize cap = (*env)->GetArrayLength(env, out);
    float *buf = (float *)malloc(cap * sizeof(float));
    int n = pivo_stream_next_frame((PivoStream *)(uintptr_t)stream, buf, (size_t)cap);
    if (n > 0) {
        (*env)->SetFloatArrayRegion(env, out, 0, (jsize)n, buf);
    }
    free(buf);
    return n;
}

JNIEXPORT void JNICALL Java_org_empero_pivo_PivoNative_streamFree(JNIEnv *env, jclass, jlong stream) {
    pivo_stream_free((PivoStream *)(uintptr_t)stream);
}
