// Android JNI/Kotlin wrapper for pivo_runner (load `libpivo_runner.so` in your app).
//
// See `examples/android/PivoNative.kt` for the Kotlin side and
// `docs/MOBILE.md` for the Gradle + NDK build recipe.

package org.empero.pivo

/** Loads libpivo_runner.so and exposes the C ABI as Kotlin methods. */
object PivoNative {
    init {
        System.loadLibrary("pivo_runner")
    }

    /** Load an unpacked model directory (files: model_manifest.json + onnx/*.onnx). */
    external fun loadModel(path: String): Long

    /** Load a .pivomodel bundle from a byte array (e.g. from assets). */
    external fun loadBundle(bytes: ByteArray): Long

    external fun freeEngine(engine: Long)
    external fun sampleRate(engine: Long): Int
    external fun lastError(): String?

    /**
     * Synthesize whole text to WAV bytes (16 kHz mono PCM16 in a RIFF container).
     * Returns null on failure (see lastError()).
     */
    external fun synthesizeWav(
        engine: Long,
        text: String,
        voice: String,
        speed: Float,
        strict: Boolean,
        overridesJson: String?
    ): ByteArray?

    /** Begin streaming. Returns a stream handle, or 0 on failure. */
    external fun streamNew(engine: Long, text: String, voice: String, speed: Float, strict: Boolean): Long

    /**
     * Read the next 160 samples into `out` (FloatArray of length >= 160).
     * Returns 160, 0 (end), or -1 (error).
     */
    external fun streamNextFrame(stream: Long, out: FloatArray): Int

    external fun streamFree(stream: Long)
}

/**
 * JNI implementation (C side) — put this in `app/src/main/cpp/pivo_jni.c`
 * and register the methods below. The full file is in `examples/android/pivo_jni.c`.
 */
