//! Stable C ABI for `pivo_runner` — the mobile/embeddable surface.
//!
//! All functions are `#[no_mangle] extern "C"` and safe to call from C, JNI (Android),
//! Swift (iOS), CPython ctypes, WebAssembly host shims, etc. See `include/pivo_ffi.h`
//! for the matching header. Errors are reported via `pivo_last_error()` /
//! `pivo_last_error_message()`.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_longlong, c_uchar, c_void, CStr, CString};

use crate::{Engine, SpeechStream, SynthOptions};

// ---------------------------------------------------------------------------
// Error plumbing (thread-local so concurrent streams can't clobber each other)
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = const { std::cell::RefCell::new(None) };
}

fn set_last_error(msg: &str) {
    let c = CString::new(msg).unwrap_or_else(|_| CString::new("pivo error").unwrap());
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

fn clear_last_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Returns a thread-local pointer to the last error message (never freed by caller).
/// Returns NULL when no error occurred.
#[no_mangle]
pub extern "C" fn pivo_last_error_message() -> *const c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(std::ptr::null())
    })
}

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------

/// Opaque handle to a loaded PiVo engine.
pub struct PivoEngine {
    engine: Engine,
}

/// Opaque handle to an in-progress streaming utterance.
pub struct PivoStream {
    stream: SpeechStream,
}

/// Load from a model directory (contains model_manifest.json + onnx/*.onnx).
/// Returns NULL on failure; call pivo_last_error_message() for details.
#[no_mangle]
pub extern "C" fn pivo_load_model(path: *const c_char) -> *mut PivoEngine {
    clear_last_error();
    if path.is_null() {
        set_last_error("model path is NULL");
        return std::ptr::null_mut();
    }
    let cstr = unsafe { CStr::from_ptr(path) };
    let path_str = match cstr.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("model path is not valid UTF-8");
            return std::ptr::null_mut();
        }
    };
    match Engine::load(path_str) {
        Ok(engine) => Box::into_raw(Box::new(PivoEngine { engine })),
        Err(e) => {
            set_last_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Load from an in-memory `.pivomodel` bundle.
#[no_mangle]
pub extern "C" fn pivo_load_bundle(bytes: *const c_uchar, len: usize) -> *mut PivoEngine {
    clear_last_error();
    if bytes.is_null() || len == 0 {
        set_last_error("bundle bytes are NULL/empty");
        return std::ptr::null_mut();
    }
    let data = unsafe { std::slice::from_raw_parts(bytes, len) };
    match Engine::load_bundle(data) {
        Ok(engine) => Box::into_raw(Box::new(PivoEngine { engine })),
        Err(e) => {
            set_last_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Pack a verified model directory into a `.pivomodel` file on disk.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn pivo_pack_bundle(model_dir: *const c_char, out_path: *const c_char) -> c_int {
    clear_last_error();
    let dir = match unsafe { cstr_to_string(model_dir) } {
        Some(s) => s,
        None => {
            set_last_error("model_dir is NULL/invalid");
            return -1;
        }
    };
    let out = match unsafe { cstr_to_string(out_path) } {
        Some(s) => s,
        None => {
            set_last_error("out_path is NULL/invalid");
            return -1;
        }
    };
    match crate::pack_model_bundle(&dir, &out) {
        Ok(_) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Pack a verified model directory into memory. `*out_len` receives the byte count.
/// Call `pivo_free_buffer` on the result.
#[no_mangle]
pub extern "C" fn pivo_pack_bundle_mem(
    model_dir: *const c_char,
    out_len: *mut usize,
) -> *mut c_uchar {
    clear_last_error();
    if out_len.is_null() {
        set_last_error("out_len is NULL");
        return std::ptr::null_mut();
    }
    let dir = match unsafe { cstr_to_string(model_dir) } {
        Some(s) => s,
        None => {
            set_last_error("model_dir is NULL/invalid");
            return std::ptr::null_mut();
        }
    };
    // Write to a temp file then read back to keep the C ABI simple and reliable.
    let tmp = std::env::temp_dir().join(format!("pivo_bundle_{}.pivomodel", std::process::id()));
    match crate::pack_model_bundle(&dir, &tmp).and_then(|_| {
        std::fs::read(&tmp).map_err(crate::onnx_loader::ModelLoadError::Io)
    }) {
        Ok(bytes) => {
            let _ = std::fs::remove_file(&tmp);
            let mut b = bytes.into_boxed_slice();
            let ptr = b.as_mut_ptr();
            unsafe { *out_len = b.len() };
            std::mem::forget(b);
            ptr
        }
        Err(e) => {
            set_last_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn pivo_free_buffer(buf: *mut c_uchar, len: usize) {
    if !buf.is_null() && len > 0 {
        unsafe {
            let _ = Vec::from_raw_parts(buf, len, len);
        }
    }
}

#[no_mangle]
pub extern "C" fn pivo_free_engine(engine: *mut PivoEngine) {
    if !engine.is_null() {
        unsafe {
            drop(Box::from_raw(engine));
        }
    }
}

#[no_mangle]
pub extern "C" fn pivo_sample_rate(engine: *mut PivoEngine) -> c_int {
    if engine.is_null() {
        return 0;
    }
    unsafe { (*engine).engine.sample_rate() as c_int }
}

/// Samples per output frame (hop_length; 160 in v1 — query this, don't hardcode).
#[no_mangle]
pub extern "C" fn pivo_frame_samples(engine: *mut PivoEngine) -> usize {
    if engine.is_null() {
        return 0;
    }
    unsafe { (*engine).engine.frame_len() }
}

/// JSON model descriptor: {"name","format","sample_rate","frame_samples","voices",...}.
/// Free with `pivo_free_string`.
#[no_mangle]
pub extern "C" fn pivo_model_info_json(engine: *mut PivoEngine) -> *mut c_char {
    if engine.is_null() {
        return std::ptr::null_mut();
    }
    let json = unsafe { (*engine).engine.model_info() }.to_string();
    match CString::new(json) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

// ---------------------------------------------------------------------------
// One-shot synthesis
// ---------------------------------------------------------------------------

/// Options struct passed to `pivo_stream_new` / `pivo_synthesize_wav`.
/// Keep this layout stable: append-only for future fields.
#[repr(C)]
pub struct PivoOptions {
    /// "Pi" or "Vo" (NULL → "Pi").
    pub voice: *const c_char,
    /// 0.5..=2.0 (0.0 → 1.0).
    pub speed: f32,
    /// 0 = relaxed OOV spelling, 1 = strict PiVo v1 contract (error on OOV).
    pub strict: c_int,
    /// Optional JSON object: {"word": ["AH0","..."]} pronunciation overrides.
    pub overrides_json: *const c_char,
}

unsafe fn cstr_to_string(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    CStr::from_ptr(p).to_str().ok().map(|s| s.to_string())
}

fn parse_overrides(json: &str) -> Result<HashMap<String, Vec<String>>, String> {
    if json.trim().is_empty() {
        return Ok(HashMap::new());
    }
    serde_json::from_str::<HashMap<String, Vec<String>>>(json).map_err(|e| e.to_string())
}

fn build_options(opts: *const PivoOptions) -> Result<SynthOptions, String> {
    let mut o = SynthOptions::default();
    if !opts.is_null() {
        let r = unsafe { &*opts };
        if let Some(v) = unsafe { cstr_to_string(r.voice) } {
            o.voice = v;
        }
        if r.speed != 0.0 {
            o.speed = r.speed;
        }
        o.strict = r.strict != 0;
        if let Some(j) = unsafe { cstr_to_string(r.overrides_json) } {
            o.overrides = parse_overrides(&j)?;
        }
    }
    Ok(o)
}

/// Synthesize `text` fully and return 16-bit PCM WAV bytes.
/// `*out_len` receives the byte count. Free with `pivo_free_buffer`.
#[no_mangle]
pub extern "C" fn pivo_synthesize_wav(
    engine: *mut PivoEngine,
    text: *const c_char,
    opts: *const PivoOptions,
    out_len: *mut usize,
) -> *mut c_uchar {
    clear_last_error();
    if engine.is_null() || out_len.is_null() {
        set_last_error("engine or out_len is NULL");
        return std::ptr::null_mut();
    }
    let text = match unsafe { cstr_to_string(text) } {
        Some(t) => t,
        None => {
            set_last_error("text is NULL/invalid");
            return std::ptr::null_mut();
        }
    };
    let options = match build_options(opts) {
        Ok(o) => o,
        Err(e) => {
            set_last_error(&e);
            return std::ptr::null_mut();
        }
    };
    let eng = unsafe { &(*engine).engine };
    match eng.synthesize(&text, &options) {
        Ok(audio) => {
            let wav = crate::wav_bytes(&audio, eng.sample_rate());
            let mut b = wav.into_boxed_slice();
            let ptr = b.as_mut_ptr();
            unsafe { *out_len = b.len() };
            std::mem::forget(b);
            ptr
        }
        Err(e) => {
            set_last_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Synthesize `text` into a caller-provided f32 buffer.
/// Returns the number of samples written, or -1 on error.
/// If `capacity` is too small, the required sample count is returned negated minus 1
/// (i.e. `-required-1`), allowing a retry with a larger buffer.
#[no_mangle]
pub extern "C" fn pivo_synthesize_pcm(
    engine: *mut PivoEngine,
    text: *const c_char,
    opts: *const PivoOptions,
    out_pcm: *mut f32,
    capacity: usize,
) -> c_longlong {
    clear_last_error();
    if engine.is_null() {
        set_last_error("engine is NULL");
        return -1;
    }
    let text = match unsafe { cstr_to_string(text) } {
        Some(t) => t,
        None => {
            set_last_error("text is NULL/invalid");
            return -1;
        }
    };
    let options = match build_options(opts) {
        Ok(o) => o,
        Err(e) => {
            set_last_error(&e);
            return -1;
        }
    };
    let eng = unsafe { &(*engine).engine };
    match eng.synthesize(&text, &options) {
        Ok(audio) => {
            if audio.len() > capacity {
                set_last_error(&format!(
                    "PCM buffer too small: need {} samples, have {}",
                    audio.len(),
                    capacity
                ));
                return -(audio.len() as c_longlong) - 1;
            }
            if !out_pcm.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(audio.as_ptr(), out_pcm, audio.len());
                }
            }
            audio.len() as c_longlong
        }
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// Pull-based streaming API (sub-50 ms TTFA on arbitrarily long text)
// ---------------------------------------------------------------------------

/// Begin a streaming utterance. Returns NULL on error.
#[no_mangle]
pub extern "C" fn pivo_stream_new(
    engine: *mut PivoEngine,
    text: *const c_char,
    opts: *const PivoOptions,
) -> *mut PivoStream {
    clear_last_error();
    if engine.is_null() {
        set_last_error("engine is NULL");
        return std::ptr::null_mut();
    }
    let text = match unsafe { cstr_to_string(text) } {
        Some(t) => t,
        None => {
            set_last_error("text is NULL/invalid");
            return std::ptr::null_mut();
        }
    };
    let options = match build_options(opts) {
        Ok(o) => o,
        Err(e) => {
            set_last_error(&e);
            return std::ptr::null_mut();
        }
    };
    let eng = unsafe { &(*engine).engine };
    match eng.synthesize_stream(&text, &options) {
        Ok(stream) => Box::into_raw(Box::new(PivoStream { stream })),
        Err(e) => {
            set_last_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Read the next frame into `out` (allocate ≥ pivo_frame_samples() floats).
/// Returns samples written (>0), 0 at end of stream, -1 on error, or -2 if
/// `capacity` is smaller than the frame size.
#[no_mangle]
pub extern "C" fn pivo_stream_next_frame(
    stream: *mut PivoStream,
    out: *mut f32,
    capacity: usize,
) -> c_int {
    clear_last_error();
    if stream.is_null() || out.is_null() {
        set_last_error("stream or out is NULL");
        return -1;
    }
    let s = unsafe { &mut (*stream).stream };
    match s.next_frame() {
        Ok(Some(frame)) => {
            if capacity < frame.len() {
                set_last_error(&format!(
                    "output buffer too small: need {} floats, have {}",
                    frame.len(),
                    capacity
                ));
                return -2;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(frame.as_ptr(), out, frame.len());
            }
            frame.len() as c_int
        }
        Ok(None) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Read up to `max_samples` float samples into `out` (whole frames only).
/// Returns samples written, 0 at end, -1 on error.
#[no_mangle]
pub extern "C" fn pivo_stream_read(
    stream: *mut PivoStream,
    out: *mut f32,
    max_samples: usize,
) -> c_longlong {
    clear_last_error();
    if stream.is_null() || out.is_null() || max_samples == 0 {
        set_last_error("stream or out is NULL");
        return -1;
    }
    let s = unsafe { &mut (*stream).stream };
    let frame_len = s.frame_len();
    let mut written = 0usize;
    while written + frame_len <= max_samples {
        match s.next_frame() {
            Ok(Some(frame)) => unsafe {
                std::ptr::copy_nonoverlapping(frame.as_ptr(), out.add(written), frame.len());
                written += frame.len();
            },
            Ok(None) => break,
            Err(e) => {
                set_last_error(&e.to_string());
                return if written > 0 {
                    written as c_longlong
                } else {
                    -1
                };
            }
        }
    }
    written as c_longlong
}

#[no_mangle]
pub extern "C" fn pivo_stream_free(stream: *mut PivoStream) {
    if !stream.is_null() {
        unsafe {
            drop(Box::from_raw(stream));
        }
    }
}

// ---------------------------------------------------------------------------
// Phonemize / introspection
// ---------------------------------------------------------------------------

/// Phonemize text. Returns a JSON string (caller frees with `pivo_free_string`).
/// `strict` semantics match `PivoOptions.strict`.
#[no_mangle]
pub extern "C" fn pivo_phonemize(
    engine: *mut PivoEngine,
    text: *const c_char,
    strict: c_int,
) -> *mut c_char {
    clear_last_error();
    if engine.is_null() {
        set_last_error("engine is NULL");
        return std::ptr::null_mut();
    }
    let text = match unsafe { cstr_to_string(text) } {
        Some(t) => t,
        None => {
            set_last_error("text is NULL/invalid");
            return std::ptr::null_mut();
        }
    };
    let eng = unsafe { &(*engine).engine };
    match eng.phonemize(&text, strict != 0, None) {
        Ok(front) => {
            let json = serde_json::to_string(&front).unwrap_or_default();
            match CString::new(json) {
                Ok(c) => c.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }
        Err(e) => {
            set_last_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn pivo_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}

// ---------------------------------------------------------------------------
// Push-based streaming convenience (avoids per-frame FFI churn on some hosts)
// ---------------------------------------------------------------------------

/// Called once per frame: `cb(user, frame_ptr, frame_samples)`.
/// Return non-zero from `cb` to abort streaming early.
pub type PivoFrameCallback = Option<unsafe extern "C" fn(*mut c_void, *const f32, usize) -> c_int>;

/// Stream all frames of `text` to `cb` with optional `user` context.
/// Returns total samples produced, or -1 on error/abort.
#[no_mangle]
pub extern "C" fn pivo_synthesize_stream_callback(
    engine: *mut PivoEngine,
    text: *const c_char,
    opts: *const PivoOptions,
    cb: PivoFrameCallback,
    user: *mut c_void,
) -> c_longlong {
    clear_last_error();
    if engine.is_null() {
        set_last_error("engine is NULL");
        return -1;
    }
    let Some(cb) = cb else {
        set_last_error("callback is NULL");
        return -1;
    };
    let text = match unsafe { cstr_to_string(text) } {
        Some(t) => t,
        None => {
            set_last_error("text is NULL/invalid");
            return -1;
        }
    };
    let options = match build_options(opts) {
        Ok(o) => o,
        Err(e) => {
            set_last_error(&e);
            return -1;
        }
    };
    let eng = unsafe { &(*engine).engine };
    let mut stream = match eng.synthesize_stream(&text, &options) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&e.to_string());
            return -1;
        }
    };
    let mut total: c_longlong = 0;
    loop {
        match stream.next_frame() {
            Ok(Some(frame)) => {
                total += frame.len() as c_longlong;
                let r = unsafe { cb(user, frame.as_ptr(), frame.len()) };
                if r != 0 {
                    return -1;
                }
            }
            Ok(None) => return total,
            Err(e) => {
                set_last_error(&e.to_string());
                return -1;
            }
        }
    }
}

