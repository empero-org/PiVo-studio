//! Minimal zero-dependency ONNX protobuf initializer parser and `.pivomodel` bundle loader.
//!
//! PiVo's four ONNX runtime graphs (`encoder.onnx`, `decoder.onnx`, `conditioning.onnx`,
//! `vocoder_step.onnx`) store all learned parameters as `float32` raw-data `TensorProto`
//! initializers inside `ModelProto.graph.initializer`. By parsing the wire format directly,
//! `pivo_runner` needs neither `onnxruntime` nor a heavy protobuf dependency, making it
//! trivial to compile for iOS, Android, WebAssembly, and desktop targets.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Logical model components resolved from a manifest. Paths are matched by
/// basename pattern rather than a fixed list, so a future export may rename
/// or relocate files (e.g. `onnx/encoder_v2.onnx` or `nets/enc.onnx`).
#[derive(Debug, Clone)]
pub struct ModelComponents {
    pub metadata: String,
    pub encoder: String,
    pub decoder: String,
    pub conditioning: String,
    pub vocoder: String,
    /// Extra manifest-pinned assets we don't recognize (preserved when packing).
    pub extra: Vec<String>,
}

/// v1 export layout (kept for documentation; resolution is manifest-driven).
pub const MODEL_FILES: &[&str] = &[
    "onnx/metadata.json",
    "onnx/encoder.onnx",
    "onnx/decoder.onnx",
    "onnx/conditioning.onnx",
    "onnx/vocoder_step.onnx",
];

pub const BUNDLE_MAGIC: &[u8; 8] = b"PIVOMDL1";

#[derive(Debug, Error)]
pub enum ModelLoadError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("validation error: {0}")]
    Validation(String),
    #[error("ONNX protobuf parse error: {0}")]
    Protobuf(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub format: String,
    #[serde(default)]
    pub model_name: String,
    #[serde(default)]
    pub total_inference_parameters: usize,
    #[serde(default)]
    pub sample_rate: u32,
    /// Optional display name used by the API/GUI (falls back to model_name).
    #[serde(default)]
    pub name: String,
    pub files: Vec<ManifestFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub width: usize,
    pub ff_width: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub text_layers: usize,
    pub decoder_layers: usize,
    pub kernel: usize,
    pub duration_width: usize,
    pub feature_dim: usize,
    pub attention_window: usize,
    #[serde(default)]
    pub dropout: f32,
    pub rope: bool,
    #[serde(default)]
    pub residual_bridges: bool,
    pub phoneme_phase: bool,
    pub sample_rate: u32,
    pub hop_length: usize,
    pub max_token_frames: usize,
    pub max_utterance_frames: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    #[serde(default)]
    pub provisional_name: String,
    #[serde(default)]
    pub total_parameters: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub format: String,
    pub trained: bool,
    pub config: ModelConfig,
    pub symbols: Vec<String>,
    pub voices: HashMap<String, i64>,
    #[serde(default = "default_preemphasis")]
    pub preemphasis: f32,
    /// Feature vector layout tag (e.g. "opus-fargan-20-v1"). Unknown layouts are
    /// tolerated — the pitch feature index is derived from the vector size.
    #[serde(default)]
    pub feature_schema: String,
    /// Parameter accounting (provisional_name used as display name).
    #[serde(default)]
    pub parameters: Option<ModelParameters>,
    #[serde(default)]
    pub cache_names: Vec<String>,
    #[serde(default)]
    pub step_state_names: Vec<String>,
}

fn default_preemphasis() -> f32 {
    0.85
}

/// Accept any `pivo-<thing>-vN` format tag and return N; reject anything else.
pub fn parse_format_version(format: &str, expected_prefix: &str) -> Result<u32, ModelLoadError> {
    let rest = format
        .strip_prefix(expected_prefix)
        .and_then(|s| s.strip_prefix('v'))
        .ok_or_else(|| {
            ModelLoadError::Validation(format!(
                "unsupported format tag '{}' (expected {}vN)",
                format, expected_prefix
            ))
        })?;
    rest.parse::<u32>().map_err(|_| {
        ModelLoadError::Validation(format!(
            "unsupported format tag '{}' (expected {}vN)",
            format, expected_prefix
        ))
    })
}

/// Resolve the five logical model components from manifest-pinned paths by
/// basename pattern. Unknown extra assets are preserved in `extra`.
pub fn resolve_components(
    paths: impl Iterator<Item = String>,
) -> Result<ModelComponents, ModelLoadError> {
    let mut metadata = None;
    let mut encoder = None;
    let mut decoder = None;
    let mut conditioning = None;
    let mut vocoder = None;
    let mut extra = Vec::new();
    for p in paths {
        let base = p
            .rsplit('/')
            .next()
            .unwrap_or(&p)
            .to_ascii_lowercase();
        if base == "model_manifest.json" {
            continue;
        } else if base.ends_with(".json") {
            if metadata.is_none() {
                metadata = Some(p);
            } else {
                extra.push(p);
            }
        } else if base.contains("encoder") {
            encoder = Some(p);
        } else if base.contains("decoder") {
            decoder = Some(p);
        } else if base.contains("conditioning") || base.contains("cond") {
            conditioning = Some(p);
        } else if base.contains("vocoder") {
            vocoder = Some(p);
        } else {
            extra.push(p);
        }
    }
    let missing = |what: &str| {
        ModelLoadError::Validation(format!("model bundle missing {} asset", what))
    };
    Ok(ModelComponents {
        metadata: metadata.ok_or_else(|| missing("metadata.json"))?,
        encoder: encoder.ok_or_else(|| missing("encoder"))?,
        decoder: decoder.ok_or_else(|| missing("decoder"))?,
        conditioning: conditioning.ok_or_else(|| missing("conditioning"))?,
        vocoder: vocoder.ok_or_else(|| missing("vocoder"))?,
        extra,
    })
}

#[derive(Debug, Clone)]
pub struct TensorData {
    pub name: String,
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

impl TensorData {
    pub fn assert_shape(&self, expected: &[usize]) -> Result<(), ModelLoadError> {
        if self.dims != expected {
            return Err(ModelLoadError::Validation(format!(
                "tensor '{}' shape mismatch: expected {:?}, got {:?}",
                self.name, expected, self.dims
            )));
        }
        Ok(())
    }
}

/// Verify a PiVo model directory against its `model_manifest.json` (hashes, sizes, and path safety).
/// Accepts `pivo-model-manifest-vN` for any N and resolves component files by basename,
/// so future exports may rename or relocate the five assets.
pub fn verify_model_dir(directory: impl AsRef<Path>) -> Result<(PathBuf, ModelManifest), ModelLoadError> {
    let root = directory.as_ref().canonicalize().map_err(|e| {
        ModelLoadError::Validation(format!(
            "cannot resolve model directory '{}': {}",
            directory.as_ref().display(),
            e
        ))
    })?;
    let manifest_path = root.join("model_manifest.json");
    if !manifest_path.is_file() {
        return Err(ModelLoadError::Validation(format!(
            "missing model_manifest.json in '{}'",
            root.display()
        )));
    }
    let manifest_text = fs::read_to_string(&manifest_path)?;
    let manifest: ModelManifest = serde_json::from_str(&manifest_text)?;
    let ver = parse_format_version(&manifest.format, "pivo-model-manifest-")?;
    if ver > 1 {
        eprintln!(
            "[pivo-studio] note: model manifest format v{} (newer than v1) — resolving components by name",
            ver
        );
    }
    // Ensure all five logical components are pinned.
    resolve_components(manifest.files.iter().map(|r| r.path.clone()))?;

    for row in &manifest.files {
        let joined = root.join(&row.path);
        let resolved = joined.canonicalize().map_err(|_| {
            ModelLoadError::Validation(format!("missing or unsafe model asset: {}", row.path))
        })?;
        if !resolved.starts_with(&root) || !resolved.is_file() {
            return Err(ModelLoadError::Validation(format!(
                "missing or unsafe model asset: {}",
                row.path
            )));
        }
        let meta = fs::metadata(&resolved)?;
        if meta.len() != row.bytes {
            return Err(ModelLoadError::Validation(format!(
                "model asset size mismatch: {}",
                row.path
            )));
        }
        let bytes = fs::read(&resolved)?;
        let digest = hex_sha256(&bytes);
        if digest != row.sha256 {
            return Err(ModelLoadError::Validation(format!(
                "model asset hash mismatch: {}",
                row.path
            )));
        }
    }

    Ok((root, manifest))
}

pub fn hex_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    let mut out = String::with_capacity(64);
    for b in result {
        use std::fmt::Write;
        let _ = write!(&mut out, "{:02x}", b);
    }
    out
}

fn read_varint(buf: &[u8], mut pos: usize) -> Result<(u64, usize), ModelLoadError> {
    let mut res = 0u64;
    let mut shift = 0u32;
    loop {
        if pos >= buf.len() {
            return Err(ModelLoadError::Protobuf("unexpected EOF in varint".into()));
        }
        let b = buf[pos];
        pos += 1;
        res |= ((b & 0x7f) as u64) << shift;
        if (b & 0x80) == 0 {
            return Ok((res, pos));
        }
        shift += 7;
        if shift >= 64 {
            return Err(ModelLoadError::Protobuf("varint overflow".into()));
        }
    }
}

fn skip_field(buf: &[u8], pos: usize, wire_type: u64) -> Result<usize, ModelLoadError> {
    match wire_type {
        0 => {
            let (_, next) = read_varint(buf, pos)?;
            Ok(next)
        }
        1 => {
            if pos + 8 > buf.len() {
                return Err(ModelLoadError::Protobuf("unexpected EOF in 64-bit field".into()));
            }
            Ok(pos + 8)
        }
        2 => {
            let (len, next) = read_varint(buf, pos)?;
            let end = next.checked_add(len as usize).ok_or_else(|| {
                ModelLoadError::Protobuf("length-delimited field overflow".into())
            })?;
            if end > buf.len() {
                return Err(ModelLoadError::Protobuf(
                    "unexpected EOF in length-delimited field".into(),
                ));
            }
            Ok(end)
        }
        5 => {
            if pos + 4 > buf.len() {
                return Err(ModelLoadError::Protobuf("unexpected EOF in 32-bit field".into()));
            }
            Ok(pos + 4)
        }
        other => Err(ModelLoadError::Protobuf(format!(
            "unsupported protobuf wire type {}",
            other
        ))),
    }
}

fn parse_tensor_proto(buf: &[u8]) -> Result<TensorData, ModelLoadError> {
    let mut pos = 0;
    let mut dims = Vec::new();
    let mut data_type = 0u64;
    let mut name = String::new();
    let mut raw_data: Option<&[u8]> = None;
    let mut float_data: Vec<f32> = Vec::new();

    while pos < buf.len() {
        let (tag, next) = read_varint(buf, pos)?;
        pos = next;
        let field_num = tag >> 3;
        let wire_type = tag & 7;
        match (field_num, wire_type) {
            (1, 0) => {
                let (d, n) = read_varint(buf, pos)?;
                pos = n;
                dims.push(d as usize);
            }
            (1, 2) => {
                let (len, n) = read_varint(buf, pos)?;
                pos = n;
                let end = pos + len as usize;
                while pos < end {
                    let (d, n2) = read_varint(buf, pos)?;
                    pos = n2;
                    dims.push(d as usize);
                }
            }
            (2, 0) => {
                let (dt, n) = read_varint(buf, pos)?;
                pos = n;
                data_type = dt;
            }
            (4, 2) => {
                let (len, n) = read_varint(buf, pos)?;
                pos = n;
                let end = pos + len as usize;
                if end > buf.len() || (len as usize) % 4 != 0 {
                    return Err(ModelLoadError::Protobuf("invalid packed float_data".into()));
                }
                for chunk in buf[pos..end].chunks_exact(4) {
                    float_data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                pos = end;
            }
            (4, 5) => {
                if pos + 4 > buf.len() {
                    return Err(ModelLoadError::Protobuf("unexpected EOF in float_data".into()));
                }
                float_data.push(f32::from_le_bytes([
                    buf[pos],
                    buf[pos + 1],
                    buf[pos + 2],
                    buf[pos + 3],
                ]));
                pos += 4;
            }
            (8, 2) => {
                let (len, n) = read_varint(buf, pos)?;
                pos = n;
                let end = pos + len as usize;
                if end > buf.len() {
                    return Err(ModelLoadError::Protobuf("unexpected EOF in tensor name".into()));
                }
                name = std::str::from_utf8(&buf[pos..end])
                    .map_err(|e| ModelLoadError::Protobuf(e.to_string()))?
                    .to_string();
                pos = end;
            }
            (9, 2) => {
                let (len, n) = read_varint(buf, pos)?;
                pos = n;
                let end = pos + len as usize;
                if end > buf.len() {
                    return Err(ModelLoadError::Protobuf("unexpected EOF in raw_data".into()));
                }
                raw_data = Some(&buf[pos..end]);
                pos = end;
            }
            _ => {
                pos = skip_field(buf, pos, wire_type)?;
            }
        }
    }

    if data_type != 1 {
        return Err(ModelLoadError::Protobuf(format!(
            "tensor '{}' has unsupported ONNX data_type {} (expected FLOAT=1)",
            name, data_type
        )));
    }

    let expected_elems: usize = dims.iter().copied().product();
    let data = if let Some(raw) = raw_data {
        if raw.len() != expected_elems * 4 {
            return Err(ModelLoadError::Protobuf(format!(
                "tensor '{}' raw_data byte length {} does not match shape {:?} ({} floats)",
                name,
                raw.len(),
                dims,
                expected_elems
            )));
        }
        let mut v = Vec::with_capacity(expected_elems);
        for chunk in raw.chunks_exact(4) {
            v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        v
    } else if float_data.len() == expected_elems {
        float_data
    } else {
        return Err(ModelLoadError::Protobuf(format!(
            "tensor '{}' missing float data",
            name
        )));
    };

    Ok(TensorData { name, dims, data })
}

// ---------------------------------------------------------------------------
// NodeProto parsing (names, inputs, integer attributes) for stable weight lookup
// ---------------------------------------------------------------------------

/// A parsed ONNX graph node — just enough to map stable node names to the
/// weight initializer they consume, independent of ordinal `onnx::MatMul_NNN`
/// initializer names that change on every re-export.
#[derive(Debug, Clone, Default)]
pub struct OnnxNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    /// list integer attributes (dilations, pads, kernel_shape, axes, …)
    pub ints: HashMap<String, Vec<i64>>,
    /// scalar int attributes (group, axis, …)
    pub int: HashMap<String, i64>,
}

/// Parsed ONNX graph: initializers plus node info for name-based weight lookup.
pub struct OnnxGraph {
    pub tensors: HashMap<String, TensorData>,
    pub nodes: Vec<OnnxNode>,
}

impl OnnxGraph {
    /// First node with this exact name (`/`-prefix optional), falling back to a
    /// path-suffix match (`/text_blocks.0/attention/q/MatMul` also matches
    /// `encoder/text_blocks.0/attention/q/MatMul`).
    pub fn node(&self, name: &str) -> Option<&OnnxNode> {
        let trimmed = name.trim_start_matches('/');
        self.nodes
            .iter()
            .find(|n| n.name == trimmed || n.name.ends_with(&format!("/{}", trimmed)))
    }

    /// Does any node named `<prefix>*` exist with one of `ops` as op_type?
    /// Used to detect optional sub-graphs (e.g. RoPE trig nodes) per block.
    pub fn has_op_under(&self, prefix: &str, ops: &[&str]) -> bool {
        self.nodes
            .iter()
            .any(|n| n.name.starts_with(prefix) && ops.iter().any(|o| *o == n.op_type))
    }

    /// The input of `node` that is an initializer (for MatMul: the weight).
    fn weight_input_name<'a>(&'a self, node: &'a OnnxNode) -> Option<&'a str> {
        node.inputs
            .iter()
            .find(|i| self.tensors.contains_key(i.as_str()))
            .map(|s| s.as_str())
    }

    /// Take the weight initializer consumed by node `name`, as raw [in,out].
    /// Returns (data, dims).
    pub fn take_node_weight(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>), ModelLoadError> {
        let node = self.node(name).cloned().ok_or_else(|| {
            ModelLoadError::Validation(format!("missing ONNX node '{}'", name))
        })?;
        let wname = self
            .weight_input_name(&node)
            .ok_or_else(|| {
                ModelLoadError::Validation(format!("node '{}' has no weight initializer", name))
            })?
            .to_string();
        let t = self.tensors.remove(&wname).ok_or_else(|| {
            ModelLoadError::Validation(format!("missing initializer '{}'", wname))
        })?;
        Ok((t.data, t.dims))
    }

    /// Take a 2-D node weight and transpose [in,out] → [out,in] for matvec use.
    /// Returns (transposed_data, out_dim, in_dim).
    pub fn take_node_weight_t(
        &mut self,
        name: &str,
    ) -> Result<(Vec<f32>, usize, usize), ModelLoadError> {
        let (data, dims) = self.take_node_weight(name)?;
        if dims.len() != 2 {
            return Err(ModelLoadError::Validation(format!(
                "node '{}' weight must be 2-D, got {:?}",
                name, dims
            )));
        }
        let (in_dim, out_dim) = (dims[0], dims[1]);
        let mut tpose = vec![0f32; in_dim * out_dim];
        for i in 0..in_dim {
            for o in 0..out_dim {
                tpose[o * in_dim + i] = data[i * out_dim + o];
            }
        }
        Ok((tpose, out_dim, in_dim))
    }

    /// Named-initializer lookup (for weights not referenced as node inputs we
    /// can rely on, e.g. norm scale vectors and biases consumed directly).
    pub fn take(&mut self, name: &str) -> Result<TensorData, ModelLoadError> {
        self.tensors.remove(name).ok_or_else(|| {
            ModelLoadError::Validation(format!("missing initializer '{}'", name))
        })
    }

    /// Conv attribute helpers.
    pub fn conv_attr_ints(&self, node_name: &str, attr: &str) -> Option<&Vec<i64>> {
        self.node(node_name)?.ints.get(attr)
    }
    pub fn conv_attr_int(&self, node_name: &str, attr: &str) -> Option<i64> {
        self.node(node_name)?.int.get(attr).copied()
    }
}

fn parse_node_proto(buf: &[u8]) -> Result<OnnxNode, ModelLoadError> {
    let mut node = OnnxNode::default();
    let mut pos = 0;
    while pos < buf.len() {
        let (tag, next) = read_varint(buf, pos)?;
        pos = next;
        let field_num = tag >> 3;
        let wire_type = tag & 7;
        match (field_num, wire_type) {
            (1, 2) | (3, 2) | (4, 2) => {
                let (len, n) = read_varint(buf, pos)?;
                pos = n;
                let end = pos + len as usize;
                if end > buf.len() {
                    return Err(ModelLoadError::Protobuf("truncated node field".into()));
                }
                let s = std::str::from_utf8(&buf[pos..end])
                    .map_err(|e| ModelLoadError::Protobuf(e.to_string()))?;
                match field_num {
                    1 => node.inputs.push(s.to_string()),
                    3 => node.name = s.to_string(),
                    _ => node.op_type = s.to_string(),
                }
                pos = end;
            }
            (2, 2) => {
                // outputs — not needed for weight lookup
                let (len, n) = read_varint(buf, pos)?;
                pos = n + len as usize;
                if pos > buf.len() {
                    return Err(ModelLoadError::Protobuf("truncated node field".into()));
                }
            }
            (5, 2) => {
                let (len, n) = read_varint(buf, pos)?;
                pos = n;
                let end = pos + len as usize;
                if end > buf.len() {
                    return Err(ModelLoadError::Protobuf("truncated attribute".into()));
                }
                parse_attribute_proto(&buf[pos..end], &mut node)?;
                pos = end;
            }
            _ => {
                pos = skip_field(buf, pos, wire_type)?;
            }
        }
    }
    Ok(node)
}

fn parse_attribute_proto(buf: &[u8], node: &mut OnnxNode) -> Result<(), ModelLoadError> {
    let mut pos = 0;
    let mut name = String::new();
    let mut i_val: Option<i64> = None;
    let mut ints: Vec<i64> = Vec::new();
    while pos < buf.len() {
        let (tag, next) = read_varint(buf, pos)?;
        pos = next;
        let field_num = tag >> 3;
        let wire_type = tag & 7;
        match (field_num, wire_type) {
            (1, 2) => {
                let (len, n) = read_varint(buf, pos)?;
                pos = n;
                let end = pos + len as usize;
                if end > buf.len() {
                    return Err(ModelLoadError::Protobuf("truncated attribute name".into()));
                }
                name = std::str::from_utf8(&buf[pos..end])
                    .map_err(|e| ModelLoadError::Protobuf(e.to_string()))?
                    .to_string();
                pos = end;
            }
            (3, 0) => {
                let (v, n) = read_varint(buf, pos)?;
                pos = n;
                i_val = Some(v as i64);
            }
            (8, 0) => {
                let (v, n) = read_varint(buf, pos)?;
                pos = n;
                ints.push(v as i64);
            }
            (8, 2) => {
                let (len, n) = read_varint(buf, pos)?;
                pos = n;
                let end = pos + len as usize;
                if end > buf.len() {
                    return Err(ModelLoadError::Protobuf("truncated ints attribute".into()));
                }
                while pos < end {
                    let (v, n2) = read_varint(buf, pos)?;
                    pos = n2;
                    ints.push(v as i64);
                }
            }
            _ => {
                pos = skip_field(buf, pos, wire_type)?;
            }
        }
    }
    if !name.is_empty() {
        if let Some(v) = i_val {
            node.int.insert(name.clone(), v);
        }
        if !ints.is_empty() {
            node.ints.insert(name, ints);
        }
    }
    Ok(())
}

/// Parse all `TensorProto` initializers + `NodeProto` metadata from an ONNX model byte buffer.
pub fn parse_onnx_graph(buf: &[u8]) -> Result<OnnxGraph, ModelLoadError> {
    let mut pos = 0;
    let mut tensors = HashMap::new();
    let mut nodes = Vec::new();
    while pos < buf.len() {
        let (tag, next) = read_varint(buf, pos)?;
        pos = next;
        let field_num = tag >> 3;
        let wire_type = tag & 7;
        if field_num == 7 && wire_type == 2 {
            // ModelProto.graph (GraphProto)
            let (glen, gnext) = read_varint(buf, pos)?;
            pos = gnext;
            let gend = pos + glen as usize;
            if gend > buf.len() {
                return Err(ModelLoadError::Protobuf("graph length exceeds file".into()));
            }
            let mut gpos = pos;
            while gpos < gend {
                let (gtag, gn) = read_varint(buf, gpos)?;
                gpos = gn;
                let gfield = gtag >> 3;
                let gwire = gtag & 7;
                if gfield == 1 && gwire == 2 {
                    // GraphProto.node (NodeProto)
                    let (nlen, nn) = read_varint(buf, gpos)?;
                    gpos = nn;
                    let nend = gpos + nlen as usize;
                    if nend > gend {
                        return Err(ModelLoadError::Protobuf(
                            "node length exceeds graph".into(),
                        ));
                    }
                    nodes.push(parse_node_proto(&buf[gpos..nend])?);
                    gpos = nend;
                } else if gfield == 5 && gwire == 2 {
                    // GraphProto.initializer (TensorProto)
                    let (tlen, tn) = read_varint(buf, gpos)?;
                    gpos = tn;
                    let tend = gpos + tlen as usize;
                    if tend > gend {
                        return Err(ModelLoadError::Protobuf(
                            "initializer length exceeds graph".into(),
                        ));
                    }
                    let tensor = parse_tensor_proto(&buf[gpos..tend])?;
                    tensors.insert(tensor.name.clone(), tensor);
                    gpos = tend;
                } else {
                    gpos = skip_field(buf, gpos, gwire)?;
                }
            }
            pos = gend;
        } else {
            pos = skip_field(buf, pos, wire_type)?;
        }
    }
    Ok(OnnxGraph { tensors, nodes })
}

/// Parse all `TensorProto` initializers from an ONNX model byte buffer
/// (kept for simple consumers that only need weights by name).
pub fn parse_onnx_initializers(buf: &[u8]) -> Result<HashMap<String, TensorData>, ModelLoadError> {
    Ok(parse_onnx_graph(buf)?.tensors)
}

/// Pack a verified model directory into a single portable `.pivomodel` binary file for mobile/embedded deployment.
///
/// Format:
/// - 8-byte magic: `PIVOMDL1`
/// - u32 LE: number of sections (6: `model_manifest.json` + the 5 `MODEL_FILES`)
/// - For each section:
///   - u32 LE: path length, followed by UTF-8 path bytes
///   - u64 LE: content byte length, followed by raw content bytes
pub fn pack_model_bundle(
    model_dir: impl AsRef<Path>,
    output_file: impl AsRef<Path>,
) -> Result<PathBuf, ModelLoadError> {
    let (root, manifest) = verify_model_dir(&model_dir)?;
    let mut bytes = Vec::with_capacity(6 * 1024 * 1024);
    bytes.extend_from_slice(BUNDLE_MAGIC);

    let mut paths: Vec<&str> = Vec::with_capacity(1 + manifest.files.len());
    paths.push("model_manifest.json");
    for f in &manifest.files {
        paths.push(f.path.as_str());
    }

    bytes.extend_from_slice(&(paths.len() as u32).to_le_bytes());
    for rel_path in paths {
        let data = fs::read(root.join(rel_path))?;
        let pbytes = rel_path.as_bytes();
        bytes.extend_from_slice(&(pbytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(pbytes);
        bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&data);
    }

    let out_path = output_file.as_ref().to_path_buf();
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&out_path, &bytes)?;
    Ok(out_path)
}

/// Unpack and verify a `.pivomodel` bundle from memory.
pub fn unpack_model_bundle(bundle: &[u8]) -> Result<HashMap<String, Vec<u8>>, ModelLoadError> {
    if bundle.len() < 12 || &bundle[0..8] != BUNDLE_MAGIC {
        return Err(ModelLoadError::Validation(
            "invalid .pivomodel bundle magic".to_string(),
        ));
    }
    let count = u32::from_le_bytes([bundle[8], bundle[9], bundle[10], bundle[11]]) as usize;
    let mut pos = 12;
    let mut files = HashMap::with_capacity(count);

    for _ in 0..count {
        if pos + 4 > bundle.len() {
            return Err(ModelLoadError::Validation("truncated bundle path len".into()));
        }
        let plen = u32::from_le_bytes([
            bundle[pos],
            bundle[pos + 1],
            bundle[pos + 2],
            bundle[pos + 3],
        ]) as usize;
        pos += 4;
        if pos + plen + 8 > bundle.len() {
            return Err(ModelLoadError::Validation("truncated bundle entry".into()));
        }
        let path = std::str::from_utf8(&bundle[pos..pos + plen])
            .map_err(|e| ModelLoadError::Validation(e.to_string()))?
            .to_string();
        pos += plen;
        let dlen = u64::from_le_bytes([
            bundle[pos],
            bundle[pos + 1],
            bundle[pos + 2],
            bundle[pos + 3],
            bundle[pos + 4],
            bundle[pos + 5],
            bundle[pos + 6],
            bundle[pos + 7],
        ]) as usize;
        pos += 8;
        if pos + dlen > bundle.len() {
            return Err(ModelLoadError::Validation("truncated bundle payload".into()));
        }
        files.insert(path, bundle[pos..pos + dlen].to_vec());
        pos += dlen;
    }

    // Verify hashes against the bundled model_manifest.json
    let manifest_bytes = files.get("model_manifest.json").ok_or_else(|| {
        ModelLoadError::Validation("bundle missing model_manifest.json".to_string())
    })?;
    let manifest: ModelManifest = serde_json::from_slice(manifest_bytes)?;
    parse_format_version(&manifest.format, "pivo-model-manifest-")?;
    if manifest.files.is_empty() {
        return Err(ModelLoadError::Validation(
            "invalid manifest inside .pivomodel bundle".to_string(),
        ));
    }
    for row in &manifest.files {
        let data = files.get(&row.path).ok_or_else(|| {
            ModelLoadError::Validation(format!("bundle missing asset: {}", row.path))
        })?;
        if data.len() as u64 != row.bytes {
            return Err(ModelLoadError::Validation(format!(
                "bundle asset size mismatch: {}",
                row.path
            )));
        }
        if hex_sha256(data) != row.sha256 {
            return Err(ModelLoadError::Validation(format!(
                "bundle asset hash mismatch: {}",
                row.path
            )));
        }
    }

    Ok(files)
}
