//! Numerical equivalence tests: compares every stage (encoder, durations,
//! decoder, conditioning, vocoder, end-to-end audio) against reference outputs
//! produced by the Python ONNX Runtime implementation.
//!
//! Run the reference generator first:
//!   python3 tests/gen_reference.py --model-dir ../huggingface --out /tmp/pivo_ref
//! Then:
//!   PIVO_REF_DIR=/tmp/pivo_ref cargo test --test numerics -- --nocapture

use std::path::PathBuf;

fn ref_dir() -> Option<PathBuf> {
    let d = std::env::var("PIVO_REF_DIR").ok()?;
    let p = PathBuf::from(d);
    if p.join("tokens.bin").exists() {
        Some(p)
    } else {
        None
    }
}

fn model_dir() -> PathBuf {
    std::env::var("PIVO_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../huggingface"))
}

/// Read a little-endian f32 binary file (as dumped by np.tofile).
fn read_f32_bin(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).expect(&format!("missing {}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i64_bin(path: &PathBuf) -> Vec<i64> {
    let bytes = std::fs::read(path).expect(&format!("missing {}", path.display()));
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([
            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
        ]))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn encoder_matches_reference() {
    let Some(rd) = ref_dir() else {
        eprintln!("skipping: PIVO_REF_DIR not set");
        return;
    };
    let tokens = read_i64_bin(&rd.join("tokens.bin"));
    let ref_encoded = read_f32_bin(&rd.join("encoded.bin"));
    let ref_logdur = read_f32_bin(&rd.join("log_durations.bin"));

    let model = pivo_runner::model::PivoModel::load_dir(model_dir())
        .expect("failed to load model");
    let (enc, log_dur) = model.encoder.run(&tokens, 0, tokens.len());

    let d_enc = max_abs_diff(&enc, &ref_encoded);
    let d_dur = max_abs_diff(&log_dur, &ref_logdur);
    eprintln!("encoder  max|Δ| = {:.3e}", d_enc);
    eprintln!("log_dur  max|Δ| = {:.3e}", d_dur);
    assert!(d_enc < 1e-4, "encoder diverged: max|Δ|={:.3e}", d_enc);
    assert!(d_dur < 1e-4, "duration diverged: max|Δ|={:.3e}", d_dur);
}

#[test]
fn decoder_matches_reference() {
    let Some(rd) = ref_dir() else {
        eprintln!("skipping: PIVO_REF_DIR not set");
        return;
    };
    let expanded = read_f32_bin(&rd.join("expanded.bin"));
    let ref_feats = read_f32_bin(&rd.join("features.bin"));

    let model = std::sync::Arc::new(
        pivo_runner::model::PivoModel::load_dir(model_dir()).expect("failed to load model"),
    );
    // derive dims from the loaded model rather than hardcoding v1 values
    let fd = model.decoder.feature_dim;
    let expanded_dim = model.expanded_dim;
    let n_frames = expanded.len() / expanded_dim;
    assert_eq!(expanded.len() % expanded_dim, 0);
    assert_eq!(ref_feats.len(), n_frames * fd);

    let feats_v = model.decode_all_features(&expanded, n_frames);
    let feats = feats_v;
    let d = max_abs_diff(&feats, &ref_feats);
    eprintln!("decoder  max|Δ| = {:.3e} ({} frames)", d, n_frames);
    assert!(d < 1e-3, "decoder diverged: max|Δ|={:.3e}", d);
}

#[test]
fn end_to_end_audio() {
    let Some(rd) = ref_dir() else {
        eprintln!("skipping: PIVO_REF_DIR not set");
        return;
    };
    let ref_audio = read_f32_bin(&rd.join("audio.bin"));
    let engine = pivo_runner::Engine::load_dir(model_dir()).expect("load failed");
    let opts = pivo_runner::SynthOptions {
        voice: "Pi".into(),
        speed: 1.0,
        strict: true,
        overrides: Default::default(),
    };
    let audio = engine
        .synthesize("NASA is ready for the next mission.", &opts)
        .expect("synthesis failed");
    eprintln!("ref {} samples, rust {} samples", ref_audio.len(), audio.len());
    assert_eq!(audio.len(), ref_audio.len(), "sample count mismatch");
    let d = max_abs_diff(&audio, &ref_audio);
    let rms_ref = (ref_audio.iter().map(|v| v * v).sum::<f32>() / ref_audio.len() as f32).sqrt();
    eprintln!("end-to-end audio max|Δ| = {:.3e} (ref RMS {:.4})", d, rms_ref);
    assert!(d < 0.02, "audio diverged: max|Δ|={:.3e}", d);
}
