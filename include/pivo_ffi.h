/* pivo_ffi.h — C ABI for the pivo_runner PiVo-1.4M TTS engine.
 *
 * Build products:
 *   - libpivo_runner.so / .dylib / .dll  (crate-type cdylib)
 *   - libpivo_runner.a                   (crate-type staticlib, ideal for iOS/Android)
 *
 * Thread-safety: engines are immutable after load and may be shared across
 * threads; each stream is single-threaded. Error strings are thread-local.
 */
#ifndef PIVO_FFI_H
#define PIVO_FFI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PivoEngine PivoEngine;
typedef struct PivoStream PivoStream;

/* Options for synthesis. All fields may be zero-initialized for defaults
 * (voice "Pi", speed 1.0, strict PiVo contract). */
typedef struct PivoOptions {
    const char *voice;          /* "Pi" or "Vo"; NULL → "Pi" */
    float speed;                /* 0.5..2.0; 0.0 → 1.0 */
    int strict;                 /* 1 = error on OOV words, 0 = spell them out */
    const char *overrides_json; /* NULL or {"word": ["AH0","..."]} */
} PivoOptions;

/* ------------------------------------------------------------------ *
 * Lifecycle                                                          *
 * ------------------------------------------------------------------ */

/* Load a model directory (with model_manifest.json + onnx/*.onnx).
 * Returns NULL on error (see pivo_last_error_message). */
PivoEngine *pivo_load_model(const char *path);

/* Load a single-file .pivomodel bundle from memory. */
PivoEngine *pivo_load_bundle(const uint8_t *bytes, size_t len);

/* Pack a verified model dir into a .pivomodel file. 0 on success. */
int pivo_pack_bundle(const char *model_dir, const char *out_path);

/* Pack into memory; free result with pivo_free_buffer. */
uint8_t *pivo_pack_bundle_mem(const char *model_dir, size_t *out_len);

void pivo_free_engine(PivoEngine *engine);
void pivo_free_buffer(uint8_t *buf, size_t len);
void pivo_free_string(char *s);

/* Sample rate of loaded engine (16000 for PiVo-1.4M — do not hardcode). */
int pivo_sample_rate(PivoEngine *engine);

/* Samples per output frame (hop_length; 160 in v1). Query this to size
 * the buffer for pivo_stream_next_frame — future models may differ. */
size_t pivo_frame_samples(PivoEngine *engine);

/* JSON model descriptor: {"name","format","sample_rate","frame_samples",
 * "voices":[...],"vocab_size":N}. Free with pivo_free_string. */
char *pivo_model_info_json(PivoEngine *engine);

/* ------------------------------------------------------------------ *
 * One-shot synthesis                                                 *
 * ------------------------------------------------------------------ */

/* Synthesize text → 16-bit PCM WAV bytes. Free with pivo_free_buffer. */
uint8_t *pivo_synthesize_wav(
    PivoEngine *engine,
    const char *text,
    const PivoOptions *opts,
    size_t *out_len);

/* Synthesize text → float32 PCM buffer (16 kHz mono).
 * Returns samples written, or -1 on error. If `capacity` is too small,
 * returns -(required+1) so you can retry with a bigger buffer. */
long long pivo_synthesize_pcm(
    PivoEngine *engine,
    const char *text,
    const PivoOptions *opts,
    float *out_pcm,
    size_t capacity);

/* ------------------------------------------------------------------ *
 * Pull-based streaming (sub-50 ms TTFA on arbitrarily long text)     *
 * ------------------------------------------------------------------ */

PivoStream *pivo_stream_new(
    PivoEngine *engine,
    const char *text,
    const PivoOptions *opts);

/* Read the next frame into `out` (allocate >= pivo_frame_samples() floats).
 * Returns samples written (>0), 0 at end of stream, -1 on error, or -2 if
 * `capacity` is smaller than the frame size. */
int pivo_stream_next_frame(PivoStream *stream, float *out, size_t capacity);

/* Read up to `max_samples` floats into `out` (whole frames only). */
long long pivo_stream_read(PivoStream *stream, float *out, size_t max_samples);

void pivo_stream_free(PivoStream *stream);

/* ------------------------------------------------------------------ *
 * Push-based callback streaming                                      *
 * ------------------------------------------------------------------ */

typedef int (*PivoFrameCallback)(void *user, const float *frame, size_t n);

/* Stream every frame to `cb` (n == pivo_frame_samples(engine) each call).
 * Return nonzero from cb to abort. Returns total samples, or -1 on error/abort. */
long long pivo_synthesize_stream_callback(
    PivoEngine *engine,
    const char *text,
    const PivoOptions *opts,
    PivoFrameCallback cb,
    void *user);

/* ------------------------------------------------------------------ *
 * Introspection / errors                                             *
 * ------------------------------------------------------------------ */

/* Phonemize text. Returns JSON string (free with pivo_free_string). */
char *pivo_phonemize(PivoEngine *engine, const char *text, int strict);

/* Last error message for this thread (NULL if none). Do not free. */
const char *pivo_last_error_message(void);

#ifdef __cplusplus
}
#endif

#endif /* PIVO_FFI_H */
