//! Self-contained PiVo Web Studio GUI — served at `/` by the built-in HTTP server.
//!
//! A single dependency-free HTML page (no CDN, no build step) compiled into the
//! binary: streaming playback via the Web Audio API with a live waveform,
//! replay + WAV download of every clip, session history, voice previews, a
//! pronunciation inspector, light/dark themes and copy-paste API snippets.
//! Edit `studio.html` directly; it is embedded with `include_str!`.

pub const INDEX_HTML: &str = include_str!("studio.html");

/// Empero brand fonts (SIL OFL 1.1, see `LICENSES/OFL-*.txt`), embedded so the
/// Studio works fully offline. Served at `/fonts/<name>`.
pub const FONTS: &[(&str, &[u8])] = &[
    ("HankenGrotesk-Regular.ttf", include_bytes!("../assets/fonts/HankenGrotesk-Regular.ttf")),
    ("HankenGrotesk-Medium.ttf", include_bytes!("../assets/fonts/HankenGrotesk-Medium.ttf")),
    ("HankenGrotesk-Semibold.ttf", include_bytes!("../assets/fonts/HankenGrotesk-Semibold.ttf")),
    ("NewsreaderDisplay-Regular.ttf", include_bytes!("../assets/fonts/NewsreaderDisplay-Regular.ttf")),
    ("JetBrainsMono-Regular.ttf", include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf")),
];

/// Look up an embedded font by file name.
pub fn font(name: &str) -> Option<&'static [u8]> {
    FONTS.iter().find(|(n, _)| *n == name).map(|(_, b)| *b)
}
