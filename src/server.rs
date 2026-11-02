//! Zero-dependency threaded HTTP server: OpenAI-compatible speech endpoint,
//! chunked-PCM + SSE streaming, a phonemize route, and the self-contained
//! PiVo Web Studio GUI.
//!
//! Only `std::net` + threads — no tokio/axum — so the same code runs on iOS,
//! Android (via JNI/host), WASM-wasi and plain desktops.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{wav_bytes, Engine, SynthOptions};

const MAX_BODY: usize = 256 * 1024;

const CORS_HEADERS: &str = "Access-Control-Allow-Origin: *\r\n\
Access-Control-Expose-Headers: X-PiVo-Sample-Rate, X-PiVo-Samples, X-PiVo-Model\r\n";

/// OpenAI-compatible TTS serving config.
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Optional bearer token required on /v1/* routes (PIVO_API_KEY).
    pub api_key: Option<String>,
    /// Max concurrent synthesis jobs (429 beyond this; 0 = unlimited).
    pub max_concurrent: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8000,
            api_key: std::env::var("PIVO_API_KEY").ok().filter(|k| !k.is_empty()),
            max_concurrent: 4,
        }
    }
}

/// Shared state handed to every connection thread.
struct Ctx {
    engine: Engine,
    config: ServerConfig,
    busy: AtomicUsize,
}

/// One running synthesis job; the slot is released on drop.
struct Slot<'a>(&'a AtomicUsize);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Ctx {
    fn try_slot(&self) -> Option<Slot<'_>> {
        let prev = self.busy.fetch_add(1, Ordering::SeqCst);
        let max = self.config.max_concurrent;
        if max != 0 && prev >= max {
            self.busy.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(Slot(&self.busy))
    }
}

/// Bind the listening socket for `config`. Use with [`serve_listener`] when the
/// bound address is needed before serving (e.g. port 0, or to open a browser).
pub fn bind(config: &ServerConfig) -> std::io::Result<TcpListener> {
    TcpListener::bind((config.host.as_str(), config.port))
}

/// Start the HTTP server (blocks the calling thread). Spawns one thread per connection.
pub fn serve(engine: Engine, config: ServerConfig) -> std::io::Result<()> {
    let listener = bind(&config)?;
    serve_listener(engine, listener, config)
}

/// Serve on an already-bound listener (blocks the calling thread).
pub fn serve_listener(
    engine: Engine,
    listener: TcpListener,
    config: ServerConfig,
) -> std::io::Result<()> {
    let ctx = Arc::new(Ctx {
        engine,
        config,
        busy: AtomicUsize::new(0),
    });
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let ctx = ctx.clone();
                std::thread::spawn(move || handle_connection(stream, &ctx));
            }
            Err(e) => eprintln!("[pivo-studio] accept error: {}", e),
        }
    }
    Ok(())
}

/// Browser-friendly URL for a bound address (`0.0.0.0` / `::` shown as loopback).
pub fn display_url(addr: SocketAddr) -> String {
    let ip = addr.ip();
    let host = if ip.is_unspecified() {
        if ip.is_ipv4() {
            "127.0.0.1".to_string()
        } else {
            "[::1]".to_string()
        }
    } else if ip.is_ipv6() {
        format!("[{}]", ip)
    } else {
        ip.to_string()
    };
    format!("http://{}:{}", host, addr.port())
}

// ---------------------------------------------------------------------------
// Minimal HTTP/1.1 handling
// ---------------------------------------------------------------------------

struct Request {
    method: String,
    path: String,
    query: HashMap<String, String>,
    headers: HashMap<String, String>,
    body: Vec<u8>,
    too_large: bool,
}

fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<Request>> {
    stream.set_read_timeout(Some(Duration::from_secs(120)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(None);
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let uri = parts.next().unwrap_or("/").to_string();
    if method.is_empty() {
        return Ok(None);
    }
    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let (path, query) = match uri.split_once('?') {
        Some((p, q)) => (p.to_string(), parse_query(q)),
        None => (uri, HashMap::new()),
    };
    let content_len: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if content_len > MAX_BODY {
        return Ok(Some(Request {
            method,
            path,
            query,
            headers,
            body: Vec::new(),
            too_large: true,
        }));
    }
    let mut body = vec![0u8; content_len];
    reader.read_exact(&mut body)?;

    Ok(Some(Request {
        method,
        path,
        query,
        headers,
        body,
        too_large: false,
    }))
}

fn parse_query(q: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for pair in q.split('&') {
        let mut it = pair.splitn(2, '=');
        let k = url_decode(it.next().unwrap_or(""));
        let v = url_decode(it.next().unwrap_or(""));
        if !k.is_empty() {
            m.insert(k, v);
        }
    }
    m
}

fn url_decode(s: &str) -> String {
    let mut out = Vec::with_capacity(s.len());
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len()
                && b[i + 1].is_ascii_hexdigit()
                && b[i + 2].is_ascii_hexdigit() =>
            {
                let h = (b[i + 1] as char).to_digit(16).unwrap_or(0) as u8;
                let l = (b[i + 2] as char).to_digit(16).unwrap_or(0) as u8;
                out.push(h * 16 + l);
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
    extra_headers: &[(&str, &str)],
) -> std::io::Result<()> {
    let mut resp = format!(
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n{}",
        status,
        content_type,
        body.len(),
        CORS_HEADERS
    );
    for (k, v) in extra_headers {
        resp.push_str(&format!("{}: {}\r\n", k, v));
    }
    resp.push_str("\r\n");
    stream.write_all(resp.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn write_chunked_start(
    stream: &mut TcpStream,
    content_type: &str,
    extra: &[(&str, &str)],
) -> std::io::Result<()> {
    let mut resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\n{}",
        content_type, CORS_HEADERS
    );
    for (k, v) in extra {
        resp.push_str(&format!("{}: {}\r\n", k, v));
    }
    resp.push_str("\r\n");
    stream.write_all(resp.as_bytes())?;
    stream.flush()
}

fn write_chunk(stream: &mut TcpStream, data: &[u8]) -> std::io::Result<()> {
    stream.write_all(format!("{:x}\r\n", data.len()).as_bytes())?;
    stream.write_all(data)?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

fn write_chunked_end(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()
}

fn json_ok(stream: &mut TcpStream, body: serde_json::Value) {
    let _ = write_response(stream, "200 OK", "application/json", body.to_string().as_bytes(), &[]);
}

fn json_error(stream: &mut TcpStream, status: &str, msg: &str) {
    let body = serde_json::json!({ "error": msg }).to_string();
    let _ = write_response(stream, status, "application/json", body.as_bytes(), &[]);
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
struct SpeechRequest {
    input: Option<String>,
    text: Option<String>,
    voice: Option<String>,
    response_format: Option<String>,
    format: Option<String>,
    speed: Option<f32>,
    /// reserved for multi-model deployments (accepted and ignored)
    #[allow(dead_code)]
    model: Option<String>,
    overrides: Option<HashMap<String, Vec<String>>>,
    /// relaxed OOV (0/1); when omitted we honor strict PiVo contract (error on OOV)
    strict: Option<bool>,
}

fn authed(req: &Request, cfg: &ServerConfig) -> bool {
    let Some(key) = &cfg.api_key else {
        return true;
    };
    let want = format!("Bearer {}", key);
    req.headers
        .get("authorization")
        .map(|v| {
            // constant-time comparison over equal-length inputs
            v.len() == want.len()
                && v
                    .as_bytes()
                    .iter()
                    .zip(want.as_bytes())
                    .fold(0u8, |acc, (a, b)| acc | (a ^ b))
                    == 0
        })
        .unwrap_or(false)
}

const FAVICON_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#5b36d6"/><path d="M8 18v-4M12 22V10M16 25V7M20 21v-10M24 17v-2" stroke="#fff" stroke-width="2.6" stroke-linecap="round"/></svg>"##;

fn handle_connection(mut stream: TcpStream, ctx: &Ctx) {
    let req = match read_request(&mut stream) {
        Ok(Some(r)) => r,
        _ => return,
    };

    // CORS preflight
    if req.method == "OPTIONS" {
        let _ = write_response(
            &mut stream,
            "204 No Content",
            "text/plain",
            b"",
            &[
                ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
                ("Access-Control-Allow-Headers", "Content-Type, Authorization"),
                ("Access-Control-Max-Age", "86400"),
            ],
        );
        return;
    }
    if req.too_large {
        json_error(
            &mut stream,
            "413 Payload Too Large",
            &format!("request body exceeds {} KiB", MAX_BODY / 1024),
        );
        return;
    }

    let engine = &ctx.engine;
    let config = &ctx.config;
    let is_api = req.path.starts_with("/v1/")
        || matches!(req.path.as_str(), "/voices" | "/phonemize" | "/speech");
    if is_api && !authed(&req, config) {
        json_error(
            &mut stream,
            "401 Unauthorized",
            "missing or invalid API key (send 'Authorization: Bearer <PIVO_API_KEY>')",
        );
        return;
    }

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") | ("GET", "/studio") | ("GET", "/index.html") => {
            let _ = write_response(
                &mut stream,
                "200 OK",
                "text/html; charset=utf-8",
                crate::studio::INDEX_HTML.as_bytes(),
                &[("Cache-Control", "no-cache")],
            );
        }
        ("GET", "/favicon.ico") | ("GET", "/favicon.svg") => {
            let _ = write_response(
                &mut stream,
                "200 OK",
                "image/svg+xml",
                FAVICON_SVG.as_bytes(),
                &[("Cache-Control", "max-age=86400")],
            );
        }
        ("GET", p) if p.starts_with("/fonts/") => match crate::studio::font(&p[7..]) {
            Some(bytes) => {
                let _ = write_response(
                    &mut stream,
                    "200 OK",
                    "font/ttf",
                    bytes,
                    &[("Cache-Control", "public, max-age=31536000, immutable")],
                );
            }
            None => json_error(&mut stream, "404 Not Found", "unknown font"),
        },
        ("GET", "/health") => json_ok(
            &mut stream,
            serde_json::json!({
                "status": "ready",
                "version": env!("CARGO_PKG_VERSION"),
                "auth_required": config.api_key.is_some(),
                "max_concurrent": config.max_concurrent,
                "busy": ctx.busy.load(Ordering::SeqCst),
                "model": engine.model_info(),
            }),
        ),
        ("GET", "/v1/voices") | ("GET", "/voices") => {
            let voices: Vec<serde_json::Value> = engine
                .voices()
                .iter()
                .map(|v| serde_json::json!({"id": v, "name": v, "language": "en"}))
                .collect();
            json_ok(
                &mut stream,
                serde_json::json!({ "voices": voices, "model": engine.model_name() }),
            );
        }
        ("POST", "/v1/phonemize") | ("POST", "/phonemize") => route_phonemize(&mut stream, &req, engine),
        ("POST", "/v1/audio/speech") | ("POST", "/speech") | ("GET", "/v1/audio/speech") => {
            route_speech(&mut stream, &req, ctx)
        }
        _ => json_error(
            &mut stream,
            "404 Not Found",
            &format!("unknown route {} {}", req.method, req.path),
        ),
    }
}

fn route_phonemize(stream: &mut TcpStream, req: &Request, engine: &Engine) {
    let sr = match serde_json::from_slice::<SpeechRequest>(&req.body) {
        Ok(sr) => sr,
        Err(e) => return json_error(stream, "400 Bad Request", &format!("invalid JSON body: {}", e)),
    };
    let text = sr.input.or(sr.text).unwrap_or_default();
    let strict = sr.strict.unwrap_or(true);
    let front = match engine.phonemize(&text, strict, sr.overrides.as_ref()) {
        Ok(f) => f,
        Err(e) => return json_error(stream, "422 Unprocessable Entity", &e.to_string()),
    };
    // Clause breakdown + per-clause symbols (used by the Studio for clause chips).
    let clause_json: Vec<serde_json::Value> = engine
        .clauses_for(&text)
        .iter()
        .map(|c| match engine.phonemize(c, strict, sr.overrides.as_ref()) {
            Ok(cf) => serde_json::json!({
                "text": c,
                "symbols": cf.symbols,
                "unknown_words": cf.unknown_words,
            }),
            Err(e) => serde_json::json!({
                "text": c,
                "symbols": [],
                "unknown_words": [],
                "error": e.to_string(),
            }),
        })
        .collect();
    json_ok(
        stream,
        serde_json::json!({
            "normalized_text": front.normalized_text,
            "symbols": front.symbols,
            "token_ids": front.token_ids,
            "unknown_words": front.unknown_words,
            "clauses": clause_json,
        }),
    );
}

fn route_speech(stream: &mut TcpStream, req: &Request, ctx: &Ctx) {
    let engine = &ctx.engine;
    let sr: SpeechRequest = if req.method == "GET" {
        SpeechRequest {
            input: req.query.get("input").or_else(|| req.query.get("text")).cloned(),
            text: None,
            voice: req.query.get("voice").cloned(),
            response_format: req.query.get("response_format").cloned(),
            format: None,
            speed: req.query.get("speed").and_then(|s| s.parse().ok()),
            model: None,
            overrides: None,
            strict: req.query.get("strict").map(|v| v == "1" || v == "true"),
        }
    } else {
        match serde_json::from_slice(&req.body) {
            Ok(v) => v,
            Err(e) => {
                return json_error(stream, "400 Bad Request", &format!("invalid JSON body: {}", e))
            }
        }
    };
    let text = sr.input.or(sr.text).unwrap_or_default();
    if text.trim().is_empty() {
        return json_error(stream, "422 Unprocessable Entity", "input must contain spoken text");
    }
    let format = sr
        .response_format
        .or(sr.format)
        .unwrap_or_else(|| "wav".to_string())
        .to_ascii_lowercase();
    let default_voice = engine
        .voices()
        .first()
        .cloned()
        .unwrap_or_else(|| "Pi".to_string());
    let opts = SynthOptions {
        voice: sr.voice.unwrap_or(default_voice),
        speed: sr.speed.unwrap_or(1.0),
        strict: sr.strict.unwrap_or(true),
        overrides: sr.overrides.unwrap_or_default(),
    };
    if !opts.speed.is_finite() || !(0.5..=2.0).contains(&opts.speed) {
        return json_error(stream, "422 Unprocessable Entity", "speed must be between 0.5 and 2.0");
    }
    if !matches!(
        format.as_str(),
        "wav" | "pcm" | "mp3" | "stream-pcm" | "pcm-stream" | "sse" | "stream"
    ) {
        return json_error(
            stream,
            "422 Unprocessable Entity",
            &format!(
                "unsupported response_format '{}' (use wav, pcm, stream-pcm or sse)",
                format
            ),
        );
    }

    let Some(_slot) = ctx.try_slot() else {
        let _ = write_response(
            stream,
            "429 Too Many Requests",
            "application/json",
            serde_json::json!({
                "error": format!(
                    "server busy: {} synthesis jobs already running, retry shortly",
                    ctx.config.max_concurrent
                )
            })
            .to_string()
            .as_bytes(),
            &[("Retry-After", "1")],
        );
        return;
    };

    let sr_str = engine.sample_rate().to_string();
    let model_name = engine.model_name();
    let t0 = Instant::now();
    let log_done = |samples: usize, how: &str| {
        let secs = samples as f32 / engine.sample_rate() as f32;
        let el = t0.elapsed().as_secs_f32();
        eprintln!(
            "[pivo-studio] {} voice={} chars={} → {:.2}s audio in {:.2}s (RTF {:.2})",
            how,
            opts.voice,
            text.chars().count(),
            secs,
            el,
            el / secs.max(0.001)
        );
    };

    match format.as_str() {
        // Buffered audio — compatible with the Python reference + OpenAI clients.
        // mp3 is aliased to wav (no codec dependencies).
        "wav" | "pcm" | "mp3" => match engine.synthesize(&text, &opts) {
            Ok(audio) => {
                let (body, ct) = if format == "pcm" {
                    (pcm16_bytes(&audio), "audio/pcm")
                } else {
                    (wav_bytes(&audio, engine.sample_rate()), "audio/wav")
                };
                log_done(audio.len(), &format);
                let _ = write_response(
                    stream,
                    "200 OK",
                    ct,
                    &body,
                    &[
                        ("X-PiVo-Sample-Rate", &sr_str),
                        ("X-PiVo-Samples", &audio.len().to_string()),
                        ("X-PiVo-Model", &model_name),
                    ],
                );
            }
            Err(e) => json_error(stream, "422 Unprocessable Entity", &e.to_string()),
        },
        // Chunked streaming: PCM16 little-endian frames as they are generated.
        "stream-pcm" | "pcm-stream" => {
            let mut s = match engine.synthesize_stream(&text, &opts) {
                Ok(s) => s,
                Err(e) => return json_error(stream, "422 Unprocessable Entity", &e.to_string()),
            };
            let ct = format!("audio/pcm; rate={}; channels=1; bit=16", engine.sample_rate());
            if write_chunked_start(stream, &ct, &[("X-PiVo-Sample-Rate", &sr_str), ("X-PiVo-Model", &model_name)])
                .is_err()
            {
                return;
            }
            let mut samples = 0usize;
            loop {
                match s.next_frame() {
                    Ok(Some(frame)) => {
                        samples += frame.len();
                        if write_chunk(stream, &pcm16_bytes(&frame)).is_err() {
                            return; // client went away
                        }
                    }
                    Ok(None) => {
                        log_done(samples, "stream-pcm");
                        let _ = write_chunked_end(stream);
                        return;
                    }
                    Err(e) => {
                        // Headers are already sent: drop the connection without the
                        // terminating chunk so clients see a truncated (failed) body.
                        eprintln!("[pivo-studio] stream aborted: {}", e);
                        return;
                    }
                }
            }
        }
        // SSE streaming: base64 PCM16 frames as `event: pcm` messages.
        _ => {
            let mut s = match engine.synthesize_stream(&text, &opts) {
                Ok(s) => s,
                Err(e) => return json_error(stream, "422 Unprocessable Entity", &e.to_string()),
            };
            if write_chunked_start(stream, "text/event-stream", &[("X-Accel-Buffering", "no")]).is_err() {
                return;
            }
            let mut idx = 0usize;
            let mut samples = 0usize;
            loop {
                match s.next_frame() {
                    Ok(Some(frame)) => {
                        samples += frame.len();
                        let ev = format!(
                            "id: {}\nevent: pcm\ndata: {}\n\n",
                            idx,
                            base64_encode(&pcm16_bytes(&frame))
                        );
                        if write_chunk(stream, ev.as_bytes()).is_err() {
                            return;
                        }
                        idx += 1;
                    }
                    Ok(None) => {
                        log_done(samples, "sse");
                        let _ = write_chunk(stream, b"event: done\ndata: {}\n\n");
                        let _ = write_chunked_end(stream);
                        return;
                    }
                    Err(e) => {
                        let ev = format!(
                            "event: error\ndata: {}\n\n",
                            serde_json::json!({ "error": e.to_string() })
                        );
                        let _ = write_chunk(stream, ev.as_bytes());
                        let _ = write_chunked_end(stream);
                        return;
                    }
                }
            }
        }
    }
}

fn pcm16_bytes(audio: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(audio.len() * 2);
    for &s in audio {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    out
}

fn base64_encode(data: &[u8]) -> String {
    const TBL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len() * 4 / 3 + 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TBL[(n >> 18) as usize & 63] as char);
        out.push(TBL[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { TBL[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TBL[n as usize & 63] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_decode_handles_escapes_and_truncation() {
        assert_eq!(url_decode("a+b%20c%2C"), "a b c,");
        assert_eq!(url_decode("100%"), "100%");
        assert_eq!(url_decode("x%4"), "x%4");
    }

    #[test]
    fn display_url_maps_unspecified_to_loopback() {
        assert_eq!(display_url("0.0.0.0:8000".parse().unwrap()), "http://127.0.0.1:8000");
        assert_eq!(display_url("127.0.0.1:9".parse().unwrap()), "http://127.0.0.1:9");
    }
}
