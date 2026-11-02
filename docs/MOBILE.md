# PiVo Studio — Mobile & Embedded Build Guide

`pivo_runner` is a **pure-Rust, zero-native-dependency** inference engine for
PiVo-1.4M TTS (~6 MB of weights). The core compiles to a `cdylib`/`staticlib`
with a stable C ABI (`include/pivo_ffi.h`) — the same code that powers the
desktop CLI and HTTP server runs on phones, tablets, WASM runtimes and embedded
targets.

## What's inside

| Piece | Notes |
| --- | --- |
| `pivo_load_model` / `pivo_load_bundle` | Load a model directory *or* a single `.pivomodel` file (ideal for shipping in an app bundle — one file instead of five). |
| `pivo_stream_new` / `pivo_stream_next_frame` | Pull-based streaming: each call returns one frame of `pivo_frame_samples()` floats (hop_length — 160 = 10 ms in v1; **always query it**, future models may differ). First frame typically arrives in < 50 ms. |
| `pivo_synthesize_wav` / `pivo_synthesize_pcm` | One-shot buffered synthesis for short phrases. |
| `pivo_model_info_json` | Model name, format, sample rate, frame size, voice list — populate your UI from this instead of hardcoding v1's two voices. |
| `pivo_synthesize_stream_callback` | Push-based streaming for hosts that prefer callbacks. |
| `pivo_phonemize` / `pivo_last_error_message` | Introspection + thread-local error strings. |

## Build products

```bash
cargo build --release --target <triple>
```

- `target/<triple>/release/libpivo_runner.so|.dylib|.dll` — shared library
- `target/<triple>/release/libpivo_runner.a` — static library (recommended for iOS/Android)

`pivo_ffi.h` needs no generated bindings — it is written by hand and stays stable.

## Model packaging for mobile

Pack the verified model directory once on desktop:

```bash
pivo-studio pack --model models/PiVo-1.4M --output pivo.pivomodel
```

`pivo.pivomodel` (~6 MB) embeds the manifest + all 5 ONNX weight assets and is
hash-verified on load. Ship it in your app bundle and call
`pivo_load_bundle(bytes, len)` — no filesystem access required.

## iOS (Swift / Objective-C)

1. Build the static library for device + simulator:

   ```bash
   rustup target add aarch64-apple-ios x86_64-apple-ios aarch64-apple-ios-sim
   cargo build --release --target aarch64-apple-ios
   cargo build --release --target x86_64-apple-ios
   # (optionally lipo them into a universal .a / XCFramework)
   ```

2. Add `libpivo_runner.a` to your Xcode target (Link Binary With Libraries).

3. Create a **Bridging Header** containing:
   ```c
   #import "pivo_ffi.h"
   ```

4. Wrap it with `examples/ios/PivoSwift.swift` — it exposes
   `PivoEngine.synthesizeWav(...)` and an `AsyncThrowingStream` for frame-by-frame
   playback through `AVAudioEngine` / `AudioQueue`.

Audio playback recipe: create an `AVAudioEngine` + `AVAudioPlayerNode` at the
engine's `sample_rate` (query `pivo_sample_rate`/`pivo_model_info_json` rather
than hardcoding 16 kHz), and push each `frame_samples`-long f32 frame as an
`AVAudioPCMBuffer` — playback starts after ~1–2 frames (~32 ms buffer in v1),
so TTFA stays sub-50 ms.

## Android (Kotlin / JNI)

1. Cross-compile with cargo-ndk:

   ```bash
   cargo install cargo-ndk
   rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
   cargo ndk -t arm64-v8a -o app/src/main/jniLibs build --release
   cargo ndk -t armeabi-v7a -o app/src/main/jniLibs build --release
   cargo ndk -t x86_64 -o app/src/main/jniLibs build --release
   ```

   This produces `libpivo_runner.so` per ABI under `jniLibs/`.

2. Put `examples/android/pivo_jni.c` in `app/src/main/cpp/` and register the
   JNI methods (see `examples/android/PivoJni.kt` for the `PivoNative` object).
   Link `libpivo_runner.so` in CMakeLists.txt:

   ```cmake
   add_library(pivo_runner SHARED IMPORTED)
   set_target_properties(pivo_runner PROPERTIES IMPORTED_LOCATION
       ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libpivo_runner.so)
   add_library(pivo_jni SHARED pivo_jni.c)
   target_include_directories(pivo_jni PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include)
   target_link_libraries(pivo_jni pivo_runner log)
   ```

3. Ship `pivo.pivomodel` in `assets/` and read it into a `ByteArray` for
   `PivoNative.loadBundle(bytes)`.

4. Playback: use `AudioTrack` in `MODE_STREAM` at the model's sample rate
   mono PCM16; convert each f32 frame to `short[]` (`v * 32767`) and write —
   TTFA < 50 ms on-device. Query `pivo_frame_samples()` / `pivo_sample_rate()`
   for sizes and rate rather than assuming 160/16000.

## Desktop

```bash
cargo build --release
# Shared:   target/release/libpivo_runner.so / .dylib / .dll
# Static:   target/release/libpivo_runner.a
# CLI+HTTP: target/release/pivo-studio
```

## Threading model

- The engine is immutable after load; share `PivoEngine*` across threads safely.
- Each `PivoStream` is single-threaded — create one per concurrent utterance.
- `pivo_last_error_message()` is thread-local.

## WASM notes

The engine compiles to `wasm32-unknown-unknown` (no WASI needed for the numeric
core; file-loading APIs are feature-gated by the `load_bundle(bytes)` path). For
in-browser TTS, feed a `.pivomodel` `ArrayBuffer` into `pivo_load_bundle` via
`wasm-bindgen`/`emscripten` bindings and pull frames through
`pivo_stream_next_frame` into an `AudioWorkletProcessor`.
