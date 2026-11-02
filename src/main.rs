//! `pivo-studio` CLI: synthesize, stream, serve, pack bundles, open the Web Studio.

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "pivo-studio",
    version,
    about = "PiVo Studio by Empero: local, streaming text-to-speech",
    long_about = "PiVo Studio by Empero: local, streaming text-to-speech.\n\nRun without a command to open the Web Studio in your browser.",
    args_conflicts_with_subcommands = true,
    after_help = "Quick start:\n  pivo-studio                                      # open the Web Studio\n  pivo-studio --model path/to/PiVo-1.4M            # …with an explicit model\n  pivo-studio synthesize --text \"Hello there.\" -o hello.wav\n\n\
--model may be omitted when PIVO_MODEL is set, or when a model directory or\n\
.pivomodel bundle is found in ./, ./models/, ./PiVo-1.4M/ or ../huggingface/."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
    /// Options for the default action (opening the Web Studio)
    #[command(flatten)]
    studio: StudioArgs,
}

#[derive(clap::Args)]
struct StudioArgs {
    #[command(flatten)]
    serve: ServeArgs,
    /// Don't open a browser window
    #[arg(long)]
    no_browser: bool,
}

#[derive(clap::Args)]
struct ModelArg {
    /// Model directory (with model_manifest.json + onnx/) or a .pivomodel bundle.
    /// Auto-detected when omitted.
    #[arg(long, short = 'm', env = "PIVO_MODEL")]
    model: Option<PathBuf>,
}

#[derive(clap::Args)]
struct ServeArgs {
    #[command(flatten)]
    model: ModelArg,
    /// Address to bind (use 0.0.0.0 to allow other devices on your network)
    #[arg(long, default_value = "127.0.0.1", env = "PIVO_HOST")]
    host: String,
    /// Port to listen on
    #[arg(long, short = 'p', default_value_t = 8000, env = "PIVO_PORT")]
    port: u16,
    /// Max simultaneous synthesis jobs before answering 429 (0 = unlimited)
    #[arg(long, default_value_t = 4)]
    max_concurrent: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// Synthesize text to a WAV file (whole-text render)
    Synthesize {
        #[command(flatten)]
        model: ModelArg,
        /// Text to speak (use "-" to read from stdin)
        #[arg(long, short)]
        text: String,
        /// Voice name (see `pivo-studio voices`)
        #[arg(long, short, default_value = "Pi")]
        voice: String,
        /// Speed 0.5..=2.0
        #[arg(long, default_value_t = 1.0)]
        speed: f32,
        /// Spell out OOV words instead of erroring
        #[arg(long)]
        relaxed: bool,
        /// Output WAV path
        #[arg(long, short, default_value = "pivo.wav")]
        output: PathBuf,
    },
    /// Stream synthesis frame-by-frame and report time-to-first-audio + RTF
    Stream {
        #[command(flatten)]
        model: ModelArg,
        /// Text to speak (use "-" to read from stdin)
        #[arg(long, short)]
        text: String,
        #[arg(long, short, default_value = "Pi")]
        voice: String,
        #[arg(long, default_value_t = 1.0)]
        speed: f32,
        #[arg(long)]
        relaxed: bool,
        /// Also write all generated samples to this raw little-endian f32 file
        #[arg(long)]
        dump_pcm: Option<PathBuf>,
    },
    /// Start the HTTP API server + PiVo Web Studio GUI
    Serve(ServeArgs),
    /// Start the server and open the PiVo Web Studio in your browser (default)
    Studio(StudioArgs),
    /// List the voices of a model
    Voices {
        #[command(flatten)]
        model: ModelArg,
    },
    /// Phonemize text and print symbols/token ids
    Phonemize {
        #[command(flatten)]
        model: ModelArg,
        #[arg(long, short)]
        text: String,
        #[arg(long)]
        relaxed: bool,
    },
    /// Pack a verified model directory into a single .pivomodel bundle
    Pack {
        #[command(flatten)]
        model: ModelArg,
        #[arg(long, short)]
        output: PathBuf,
    },
}

type CmdResult = Result<(), String>;

fn main() {
    let cli = Cli::parse();
    let command = cli.command.unwrap_or(Commands::Studio(cli.studio));
    let res = match command {
        Commands::Synthesize { model, text, voice, speed, relaxed, output } => {
            cmd_synthesize(model, text, voice, speed, relaxed, output)
        }
        Commands::Stream { model, text, voice, speed, relaxed, dump_pcm } => {
            cmd_stream(model, text, voice, speed, relaxed, dump_pcm)
        }
        Commands::Serve(args) => cmd_serve(args, false),
        Commands::Studio(a) => cmd_serve(a.serve, !a.no_browser),
        Commands::Voices { model } => cmd_voices(model),
        Commands::Phonemize { model, text, relaxed } => cmd_phonemize(model, text, relaxed),
        Commands::Pack { model, output } => cmd_pack(model, output),
    };
    if let Err(e) = res {
        eprintln!("error: {}", e);
        std::process::exit(2);
    }
}

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------

fn is_model_dir(p: &Path) -> bool {
    p.join("model_manifest.json").is_file()
}

fn is_bundle(p: &Path) -> bool {
    p.is_file() && p.extension().map_or(false, |e| e == "pivomodel")
}

/// Look for a model in the usual places when `--model` / `PIVO_MODEL` is absent.
fn discover_model() -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = vec![
        ".".into(),
        "models".into(),
        "model".into(),
        "PiVo-1.4M".into(),
        "models/PiVo-1.4M".into(),
        "models/pivo".into(),
        "../huggingface".into(),
    ];
    // Also next to the executable (handy for unzipped release archives).
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        roots.push(dir.clone());
        roots.push(dir.join("models"));
    }
    for root in &roots {
        if is_model_dir(root) {
            return Some(root.clone());
        }
    }
    // Fall back to any *.pivomodel bundle or model subdirectory one level down.
    for root in &roots {
        let Ok(rd) = std::fs::read_dir(root) else { continue };
        let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
        entries.sort();
        if let Some(p) = entries.iter().find(|p| is_bundle(p)) {
            return Some(p.clone());
        }
        if let Some(p) = entries.iter().find(|p| p.is_dir() && is_model_dir(p)) {
            return Some(p.clone());
        }
    }
    None
}

fn resolve_model(arg: ModelArg) -> Result<PathBuf, String> {
    if let Some(p) = arg.model {
        if !p.exists() {
            return Err(format!(
                "model path '{}' does not exist.\n\
                 Pass a PiVo model directory (containing model_manifest.json) or a .pivomodel bundle.",
                p.display()
            ));
        }
        return Ok(p);
    }
    discover_model().ok_or_else(|| {
        "no PiVo model found.\n\n\
         Tell pivo-studio where the model is, for example:\n  \
           pivo-studio studio --model path/to/PiVo-1.4M\n  \
           export PIVO_MODEL=path/to/PiVo-1.4M\n\n\
         Or put the model directory (or a .pivomodel bundle) in ./models/.\n\
         Model weights: https://huggingface.co/empero-org/PiVo-1.4M"
            .to_string()
    })
}

fn load_engine(arg: ModelArg) -> Result<pivo_runner::Engine, String> {
    let path = resolve_model(arg)?;
    let t0 = Instant::now();
    let engine = pivo_runner::Engine::load(&path)
        .map_err(|e| format!("failed to load model from '{}': {}", path.display(), e))?;
    eprintln!(
        "[pivo-studio] loaded {} from {} in {} ms",
        engine.model_name(),
        path.display(),
        t0.elapsed().as_millis()
    );
    Ok(engine)
}

fn read_text(text: String) -> Result<String, String> {
    if text == "-" {
        let mut s = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut s)
            .map_err(|e| format!("cannot read stdin: {}", e))?;
        Ok(s)
    } else {
        Ok(text)
    }
}

fn synth_opts(voice: String, speed: f32, relaxed: bool) -> pivo_runner::SynthOptions {
    pivo_runner::SynthOptions {
        voice,
        speed,
        strict: !relaxed,
        overrides: Default::default(),
    }
}

fn oov_hint(e: impl ToString) -> String {
    let s = e.to_string();
    if s.contains("out-of-vocabulary") {
        format!("{}\nhint: pass --relaxed to spell unknown words letter-by-letter", s)
    } else {
        s
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn cmd_synthesize(
    model: ModelArg,
    text: String,
    voice: String,
    speed: f32,
    relaxed: bool,
    output: PathBuf,
) -> CmdResult {
    let text = read_text(text)?;
    let engine = load_engine(model)?;
    let t0 = Instant::now();
    let audio = engine
        .synthesize(&text, &synth_opts(voice, speed, relaxed))
        .map_err(oov_hint)?;
    let secs = audio.len() as f32 / engine.sample_rate() as f32;
    let elapsed = t0.elapsed().as_secs_f32();
    let wav = pivo_runner::wav_bytes(&audio, engine.sample_rate());
    std::fs::write(&output, &wav).map_err(|e| format!("cannot write {}: {}", output.display(), e))?;
    eprintln!(
        "[pivo-studio] {:.2}s of audio in {:.2}s (RTF {:.2})",
        secs,
        elapsed,
        elapsed / secs.max(0.001)
    );
    println!("{}", output.display());
    Ok(())
}

fn cmd_stream(
    model: ModelArg,
    text: String,
    voice: String,
    speed: f32,
    relaxed: bool,
    dump_pcm: Option<PathBuf>,
) -> CmdResult {
    let text = read_text(text)?;
    let engine = load_engine(model)?;
    let t0 = Instant::now();
    let mut stream = engine
        .synthesize_stream(&text, &synth_opts(voice, speed, relaxed))
        .map_err(oov_hint)?;
    let mut dump = Vec::new();
    let mut samples = 0usize;
    let mut frames = 0usize;
    while let Some(f) = stream.next_frame().map_err(oov_hint)? {
        if frames == 0 {
            eprintln!("[pivo-studio] first audio after {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        samples += f.len();
        frames += 1;
        if dump_pcm.is_some() {
            dump.extend_from_slice(&f);
        }
    }
    let secs = samples as f32 / engine.sample_rate() as f32;
    let el = t0.elapsed().as_secs_f32();
    eprintln!(
        "[pivo-studio] {} frames ({:.2}s audio) in {:.2}s — RTF {:.2}",
        frames,
        secs,
        el,
        el / secs.max(0.001)
    );
    if let Some(path) = dump_pcm {
        let bytes: Vec<u8> = dump.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).map_err(|e| format!("cannot write {}: {}", path.display(), e))?;
        println!("{}", path.display());
    }
    Ok(())
}

fn cmd_serve(args: ServeArgs, open_browser: bool) -> CmdResult {
    let engine = load_engine(args.model)?;
    let mut cfg = pivo_runner::server::ServerConfig {
        host: args.host,
        port: args.port,
        api_key: std::env::var("PIVO_API_KEY").ok().filter(|k| !k.is_empty()),
        max_concurrent: args.max_concurrent,
    };
    let listener = match pivo_runner::server::bind(&cfg) {
        Ok(l) => l,
        // For the Studio, a busy port shouldn't be a dead end: take any free one.
        Err(e) if open_browser && e.kind() == std::io::ErrorKind::AddrInUse => {
            eprintln!("[pivo-studio] port {} is busy, picking a free one", cfg.port);
            cfg.port = 0;
            pivo_runner::server::bind(&cfg).map_err(|e| e.to_string())?
        }
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            return Err(format!(
                "port {} is already in use — try --port {} (or another free port)",
                cfg.port,
                cfg.port.wrapping_add(1)
            ))
        }
        Err(e) => return Err(format!("cannot listen on {}:{}: {}", cfg.host, cfg.port, e)),
    };
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let url = pivo_runner::server::display_url(addr);

    let voices = engine.voices().join(", ");
    eprintln!();
    eprintln!("  PiVo Studio is running");
    eprintln!();
    eprintln!("  ➜  Studio:  {}", url);
    eprintln!("  ➜  API:     {}/v1/audio/speech  (OpenAI-compatible)", url);
    eprintln!("     Model:   {} · voices: {} · {} Hz", engine.model_name(), voices, engine.sample_rate());
    eprintln!(
        "     Auth:    {}",
        if cfg.api_key.is_some() { "API key required (PIVO_API_KEY)" } else { "none (local use)" }
    );
    if addr.ip().is_unspecified() {
        eprintln!("     Network: listening on all interfaces (port {})", addr.port());
    }
    eprintln!();
    eprintln!("  Press Ctrl+C to stop.");
    eprintln!();

    if open_browser {
        // The socket is already bound, so the page is reachable immediately.
        if let Err(e) = open::that(&url) {
            eprintln!("[pivo-studio] could not open a browser ({}); visit {} manually", e, url);
        }
    }
    pivo_runner::server::serve_listener(engine, listener, cfg).map_err(|e| e.to_string())
}

mod open {
    pub fn that(url: &str) -> std::io::Result<()> {
        #[cfg(target_os = "macos")]
        std::process::Command::new("open").arg(url).spawn()?;
        #[cfg(target_os = "windows")]
        std::process::Command::new("cmd").args(["/c", "start", "", url]).spawn()?;
        #[cfg(all(unix, not(target_os = "macos")))]
        std::process::Command::new("xdg-open")
            .arg(url)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
        Ok(())
    }
}

fn cmd_voices(model: ModelArg) -> CmdResult {
    let engine = load_engine(model)?;
    for v in engine.voices() {
        println!("{}", v);
    }
    Ok(())
}

fn cmd_pack(model: ModelArg, output: PathBuf) -> CmdResult {
    let model = resolve_model(model)?;
    let p = pivo_runner::pack_model_bundle(&model, &output).map_err(|e| e.to_string())?;
    println!("{}", p.display());
    Ok(())
}

fn cmd_phonemize(model: ModelArg, text: String, relaxed: bool) -> CmdResult {
    let text = read_text(text)?;
    let engine = load_engine(model)?;
    let front = engine.phonemize(&text, !relaxed, None).map_err(oov_hint)?;
    println!("normalized: {}", front.normalized_text);
    println!("symbols:    {}", front.symbols.join(" "));
    println!("token_ids:  {:?}", front.token_ids);
    if !front.unknown_words.is_empty() {
        println!("unknown:    {}", front.unknown_words.join(", "));
    }
    Ok(())
}
