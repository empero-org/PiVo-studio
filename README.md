# PiVo Studio

**Local, streaming text-to-speech — by Empero.**

A **pure-Rust, zero-native-dependency** inference engine, CLI, HTTP server and
Web Studio GUI for the **PiVo** family of small TTS models (ships with PiVo-1.4M —
1.4M-parameter, two-voice English TTS). No Python, no ONNX Runtime, no PyTorch:
the ONNX weight graphs are parsed directly and executed by a hand-written,
cache-friendly f32 kernel (dot-product + matvec primitives that LLVM
auto-vectorizes for AVX2/NEON).

## Quick start

You need a [Rust toolchain](https://rustup.rs) and the PiVo model weights
([`empero-ai/PiVo-1.4M`](https://huggingface.co/empero-ai/PiVo-1.4M) on
Hugging Face).

```bash
# 1. Build (≈1 minute, one single binary)
cargo build --release

# 2. Get the model (any of these works)
git clone https://huggingface.co/empero-ai/PiVo-1.4M models/PiVo-1.4M
#   …or: pivo download --repo empero-ai/PiVo-1.4M --output models/PiVo-1.4M

# 3. Open the Studio in your browser
./target/release/pivo-studio studio
```

That's it: the Studio opens at `http://127.0.0.1:8000`. Type some text, pick
a voice and press **Speak** (or <kbd>Ctrl</kbd>+<kbd>Enter</kbd>). Audio starts
playing within a few milliseconds while the rest is still being generated.

`--model` is optional. pivo-studio looks for a model in `$PIVO_MODEL`, `./`,
`./models/`, `./PiVo-1.4M/`, `../huggingface/` and next to the executable
(either a model directory or a `.pivomodel` bundle). If the Studio's default
port is taken, it picks a free one.

### PiVo Studio

- **Streaming playback** with a live waveform, time-to-first-audio and
  real-time-factor readouts.
- **Replay & download** every clip as WAV, with a session **history**.
- **Voice cards** with one-click previews, speed presets and sample texts.
- **Pronunciation inspector**: normalized text, ARPAbet phonemes per word,
  clause split and warnings for out-of-dictionary words.
- **API tab** with copy-paste `curl` / Python (OpenAI SDK) / JavaScript
  snippets for your current text and settings.
- Light & dark themes, keyboard shortcuts, responsive layout, and support for
  servers protected by an API key. Fully offline: no CDN, no tracking.

## Future-proofed for PiVo-v2

Every dimension is resolved at load time — nothing is hard-coded to v1:

- `metadata.json` `config` supplies widths, layers, heads, KV grouping, attention
  window, hop length, sample rate, preemphasis, and pipeline flags
  (`phoneme_phase`, `rope`).
- ONNX **initializer shapes** supply vocab size, voice count, feature dim,
  duration width, and all vocoder dimensions.
- ONNX **node structure** supplies block types (conv vs attention), per-block
  RoPE (detected via `Sin`/`Cos` ops), conv dilations/pads, and stable
  name-based weight lookup (`/text_blocks.0/attention/q/MatMul`) — ordinal
  `onnx::MatMul_NNN` names are never used.
- Manifest components resolve by basename, so a v2 export may rename or relocate
  the five assets; `pivo-model-manifest-vN` and `pivo-onnx-vN` accept any N.

A v2 model with different dims/voices/vocab drops in with **zero code changes**.

## Why

- **Faster than real time**: RTF ≈ **0.4–0.6** on a single CPU thread (≈1.6–2.5×
  real-time), vs ≈1.0 for the Python+ONNX Runtime reference.
- **Streams like a phone app**: pull-based 10 ms frame generation with clause
  splitting — first audio in **≈30–50 ms** even for long documents, so playback
  starts while the rest is still rendering.
- **Mobile-ready C ABI**: `include/pivo_ffi.h` + `cdylib`/`staticlib` builds for
  iOS (`aarch64-apple-ios`), Android (`aarch64-linux-android`), and WASM hosts,
  plus a single-file `.pivomodel` bundle format.
- **Drop-in API**: OpenAI-compatible `/v1/audio/speech`, chunked PCM + SSE
  streaming endpoints, `/v1/voices`, `/v1/phonemize`, `/health`, and a
  self-contained interactive Web Studio GUI.

## Build

```bash
cargo build --release        # → target/release/pivo-studio + libpivo_runner.{a,so}
cargo install --path .       # optional: put pivo-studio on your PATH
```

The CMU pronouncing dictionary is compiled into the binary (`include_str!`), so
the only runtime data needed is the model weights.

## Model files

Point `--model` at either:

- a **PiVo model directory** containing `model_manifest.json` and
  `onnx/{metadata.json,encoder,decoder,conditioning,vocoder_step}.onnx`
  (hashes are verified on load), or
- a single **`.pivomodel` bundle** created by `pivo-studio pack`:

```bash
pivo-studio pack --model models/PiVo-1.4M --output pivo.pivomodel
```

## CLI

```bash
# Full WAV render (writes pivo.wav unless -o is given)
pivo-studio synthesize --voice Pi --text "NASA is ready for the next mission." -o pi.wav

# Read text from stdin
cat chapter.txt | pivo-studio synthesize --relaxed -t - -o chapter.wav

# Streaming benchmark (prints time-to-first-audio + RTF)
pivo-studio stream --voice Vo --text "Streaming on a phone, completely locally." --dump-pcm out.f32

# Inspect
pivo-studio voices
pivo-studio phonemize --text "On March 4th, 2026."
```

All commands accept `--model <dir|bundle>` (or `PIVO_MODEL`); it is
auto-detected when omitted. `--relaxed` spells out-of-vocabulary words
letter-by-letter instead of failing (strict PiVo v1 contract is the default).

## HTTP server + Web Studio

```bash
pivo-studio serve                    # API + Studio on http://127.0.0.1:8000
pivo-studio studio                   # same, and opens your browser
pivo-studio serve --host 0.0.0.0 --port 9000 --max-concurrent 8
```

| Option | Env | Default | Notes |
| --- | --- | --- | --- |
| `--model`, `-m` | `PIVO_MODEL` | auto-detect | model directory or `.pivomodel` bundle |
| `--host` | `PIVO_HOST` | `127.0.0.1` | use `0.0.0.0` to allow other devices |
| `--port`, `-p` | `PIVO_PORT` | `8000` | `studio` falls back to a free port if busy |
| `--max-concurrent` | | `4` | synthesis jobs before `429 Too Many Requests` (0 = unlimited) |
| | `PIVO_API_KEY` | unset | require `Authorization: Bearer <key>` on `/v1/*` |
| `--no-browser` | | | `studio` only |

### Endpoints

| Route | Notes |
| --- | --- |
| `GET /health` | `{"status":"ready","version","auth_required","max_concurrent","busy","model":{name,format,sample_rate,frame_samples,voices,vocab_size}}` (never requires auth) |
| `GET /v1/voices` | model-derived voice list |
| `POST /v1/audio/speech` | OpenAI-compatible. Body: `input`, `voice`, `speed` (0.5–2.0), `response_format`, `strict`, `overrides`. |
| `GET /v1/audio/speech?...` | Same as POST but via query params (handy for `<audio>`/EventSource). |
| `POST /v1/phonemize` | `{ "input": "…", "strict": true }` → normalized text, ARPAbet symbols, token ids (mapped through the model's symbol table), OOV list, and per-clause breakdown. |
| `GET /`, `/studio` | The embedded Web Studio HTML page. |

`response_format` values:

- `"wav"` — buffered PCM16 WAV (OpenAI-compatible default)
- `"pcm"` — buffered raw PCM16
- `"stream-pcm"` — `Transfer-Encoding: chunked` 16-bit PCM16 at the model's
  sample rate, one chunk per frame (`frame_samples` × 2 bytes) as it is
  generated (TTFA ~30–50 ms)
- `"sse"` / `"stream"` — `text/event-stream`, `event: pcm` + base64 PCM16 data

Authentication: set `PIVO_API_KEY` to require `Authorization: Bearer <key>` on
all `/v1/*` routes. The Studio page itself stays public and asks for the key.
Errors are always JSON: `{"error": "…"}`. CORS is enabled for all origins.

The OpenAI Python SDK works out of the box:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
client.audio.speech.create(model="pivo", voice="Pi", input="Hello!").write_to_file("hello.wav")
```

```bash
# OpenAI-style buffered WAV
curl -o out.wav -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"NASA is ready for the next mission.","voice":"Pi","response_format":"wav"}'

# Chunked PCM16 stream (first bytes arrive in ~30 ms)
curl -N -o live.pcm -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"A long document…","response_format":"stream-pcm","strict":false}'

# SSE stream
curl -N -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello.","response_format":"sse"}'
```

## C ABI (mobile / embeddable)

```c
#include "pivo_ffi.h"

PivoEngine *e = pivo_load_bundle(bundle_bytes, bundle_len);
size_t frame_n = pivo_frame_samples(e);      // query; 160 in v1
float *frame = malloc(frame_n * sizeof(float));
PivoStream *s = pivo_stream_new(e, "Hello, world.", NULL);
while (pivo_stream_next_frame(s, frame, frame_n) > 0) { /* play */ }
pivo_stream_free(s);
pivo_free_engine(e);
```

`pivo_model_info_json(e)` returns the model name, format, sample rate, frame
size and voice list — use it to populate UI and validate voices.

See `include/pivo_ffi.h`, `examples/c_main.c`,
`examples/android/pivo_jni.c` + `PivoJni.kt`, `examples/ios/PivoSwift.swift`,
and **`docs/MOBILE.md`** for iOS/Android/WASM build recipes.

## Architecture (how it maps to the ONNX graphs)

| ONNX file | Rust implementation |
| --- | --- |
| `encoder.onnx` | 6×96-width blocks alternating GQA attention (RoPE, 4 Q/2 KV heads) and dilated convs, duration predictor (SiLU → depthwise/pointwise convs → RMSNorm → out proj) |
| `decoder.onnx` | 2 causal GQA blocks with a sliding 64-frame window; KV cache keeps the last 63 post-RoPE keys/values; feature head predicts 18 cepstra + pitch + pitch-correlation |
| `conditioning.onnx` | pitch embedding (224×12) + dense→conv→dense→320 |
| `vocoder_step.onnx` | FARGAN: 3 GRUs (160/128/128) + GLU skips + 256-sample pitch memory + frame-state convolution → 40 samples per subframe; de-emphasis 0.85 applied on the host |

All weight tensors are read straight out of the ONNX graphs by a minimal
protobuf decoder (`onnx_loader.rs`) that understands `TensorProto` initializers
and `NodeProto` names/inputs/attributes — no `onnx` crate, no protobuf codegen,
no external runtime.

## Numerical accuracy

Checked against the Python ONNX Runtime reference (`tests/gen_reference.py` +
`tests/numerics.rs`, max |Δ|):

| Stage | max \|Δ\| |
| --- | --- |
| encoder | 1.8e-6 |
| log_durations | 3.2e-6 |
| decoder features (187 frames) | 1.7e-5 |
| end-to-end PCM | 5.4e-5 |

## Benchmarks (single thread, x86-64, release)

- RTF ≈ **0.44–0.64** (1.6–2.3× real-time) for whole-utterance synthesis
- TTFA ≈ **30–35 ms** for streaming (model load ≈100 ms once, then amortized)
- Peak RSS ≈ **~50 MB** (weights ~6 MB + lexicon hashmap + scratch)

## Tests

```bash
# Generate Python reference tensors (requires onnxruntime + numpy)
python3 tests/gen_reference.py --model-dir ../huggingface --out /tmp/pivo_ref

# Run everything (frontend contract + per-stage numerics + end-to-end)
PIVO_REF_DIR=/tmp/pivo_ref PIVO_MODEL_DIR=../huggingface cargo test --release
```

## Layout

```
pivo-studio/
├── Cargo.toml            # rlib + cdylib + staticlib + pivo-studio binary
├── include/pivo_ffi.h    # stable C ABI for mobile/embedded
├── assets/cmudict.dict   # CMU pronouncing dictionary (compiled in)
├── src/
│   ├── main.rs           # CLI: synthesize/stream/serve/pack/studio/phonemize
│   ├── lib.rs            # Engine, SynthOptions, SpeechStream, wav_bytes
│   ├── text.rs           # English normalizer + CMUdict phonemizer + clause splitter
│   ├── onnx_loader.rs    # manifest verify, ONNX protobuf parser, .pivomodel bundle
│   ├── model.rs          # encoder / decoder / conditioning / vocoder kernels
│   ├── ffi.rs            # #[no_mangle] extern "C" functions
│   ├── server.rs         # zero-dep threaded HTTP server
│   ├── studio.rs         # embeds the Web Studio into the binary
│   └── studio.html       # the Web Studio (single dependency-free page)
├── examples/
│   ├── c_main.c          # plain C consumer
│   ├── android/          # JNI + Kotlin
│   └── ios/              # Swift wrapper
├── docs/MOBILE.md        # iOS/Android/WASM build recipes
└── tests/
    ├── frontend.rs       # text contract tests
    ├── numerics.rs       # per-stage + end-to-end equivalence vs Python
    └── gen_reference.py  # generates reference tensors
```

## License

Engine code: **Empero PiVo Attribution License 1.0** (`LICENSE`).

- **All uses and derivatives** must carry prominent, user-visible attribution:
  "Made with Empero PiVo" or "Powered by Empero PiVo".
- **Commercial use** must additionally credit Empero directly and link
  to https://empero.org.
- **Commercial use by entities with annual revenue above US$20M** requires a
  separate commercial license — contact research@empero.org.

The PiVo model weights are Apache-2.0 and distributed separately.
FARGAN-derived vocoder code/weights: Xiph/Opus BSD.
CMUdict: CMU license. See `NOTICE`, `THIRD_PARTY.md`, `LICENSES/`.
