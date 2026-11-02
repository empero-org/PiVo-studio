/* Minimal C consumer of libpivo_runner.
 *
 * Build (Linux):
 *   gcc -O2 -o pivo_c examples/c_main.c -I include -L target/release -lpivo_runner -lm -lpthread -ldl
 * Run:
 *   LD_LIBRARY_PATH=target/release ./pivo_c /path/to/model_dir "NASA is ready for the next mission." out.wav
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pivo_ffi.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model_dir_or_bundle> <text> [out.wav]\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *text = argv[2];
    const char *out_path = argc > 3 ? argv[3] : "pivo.wav";

    /* If path ends with .pivomodel, load bundle; else directory. */
    PivoEngine *engine;
    if (strstr(model_path, ".pivomodel")) {
        FILE *f = fopen(model_path, "rb");
        if (!f) { perror("fopen"); return 1; }
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        unsigned char *buf = malloc(n);
        fread(buf, 1, n, f);
        fclose(f);
        engine = pivo_load_bundle(buf, (size_t)n);
        free(buf);
    } else {
        engine = pivo_load_model(model_path);
    }
    if (!engine) {
        fprintf(stderr, "load error: %s\n", pivo_last_error_message());
        return 1;
    }
    fprintf(stderr, "engine ready @ %d Hz\n", pivo_sample_rate(engine));

    /* One-shot WAV */
    PivoOptions opts = {0};
    opts.voice = "Pi";
    opts.speed = 1.0f;
    opts.strict = 0; /* spell out OOV words */
    size_t wav_len = 0;
    unsigned char *wav = pivo_synthesize_wav(engine, text, &opts, &wav_len);
    if (!wav) {
        fprintf(stderr, "synth error: %s\n", pivo_last_error_message());
        pivo_free_engine(engine);
        return 1;
    }
    FILE *f = fopen(out_path, "wb");
    fwrite(wav, 1, wav_len, f);
    fclose(f);
    pivo_free_buffer(wav, wav_len);
    fprintf(stderr, "wrote %s (%zu bytes)\n", out_path, wav_len);

    /* Streaming: print first frame stats (frame size is model-defined) */
    size_t frame_n = pivo_frame_samples(engine);
    PivoStream *s = pivo_stream_new(engine, "Streaming hello, world.", &opts);
    float *frame = calloc(frame_n, sizeof(float));
    int got = pivo_stream_next_frame(s, frame, frame_n);
    fprintf(stderr, "stream: first frame = %d samples (hop %zu)\n", got, frame_n);
    while (pivo_stream_next_frame(s, frame, frame_n) > 0) {}
    pivo_stream_free(s);
    free(frame);

    pivo_free_engine(engine);
    return 0;
}
