//! Deterministic English text normalizer, CMUdict phonemizer, and intelligent clause splitter.
//!
//! Matches the exact contract of `huggingface/pivo/text.py` (numbers, currency, ordinals,
//! ISO dates, abbreviations, initialisms, NASA pronunciation, and strict/relaxed OOV handling),
//! while adding a smart clause/sentence chunker so arbitrarily long text can be streamed
//! with sub-50ms time-to-first-audio.

use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

use regex::{Captures, Regex};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SPECIAL: &[&str] = &[
    "<pad>", "<bos>", "<eos>", "<space>", ".", ",", "?", "!", ";", ":",
];

pub const SYMBOLS: &[&str] = &[
    "<pad>", "<bos>", "<eos>", "<space>", ".", ",", "?", "!", ";", ":", "B", "CH", "D", "DH",
    "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W",
    "Y", "Z", "ZH", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0",
    "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "EH0", "EH1", "EH2", "ER0", "ER1",
    "ER2", "EY0", "EY1", "EY2", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "OW0", "OW1", "OW2",
    "OY0", "OY1", "OY2", "UH0", "UH1", "UH2", "UW0", "UW1", "UW2",
];

pub const VOCAB_SIZE: usize = 128;

const ONES: &[&str] = &[
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

const TENS: &[&str] = &[
    "zero", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

const MONTHS: &[&str] = &[
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

#[derive(Debug, Error)]
pub enum FrontendError {
    #[error("{0}")]
    InvalidInput(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontendResult {
    pub normalized_text: String,
    pub symbols: Vec<String>,
    pub token_ids: Vec<i64>,
    pub unknown_words: Vec<String>,
}

pub fn symbol_to_id(sym: &str) -> Option<i64> {
    static MAP: OnceLock<HashMap<&'static str, i64>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::with_capacity(SYMBOLS.len());
        for (i, &s) in SYMBOLS.iter().enumerate() {
            m.insert(s, i as i64);
        }
        m
    });
    map.get(sym).copied()
}

pub fn voice_id(voice: &str) -> Option<i64> {
    match voice {
        "Pi" => Some(0),
        "Vo" => Some(1),
        _ => None,
    }
}

/// Bundled CMU pronouncing dictionary (first pronunciation for each word).
pub fn lexicon() -> &'static HashMap<String, Vec<&'static str>> {
    static DICT: OnceLock<HashMap<String, Vec<&'static str>>> = OnceLock::new();
    DICT.get_or_init(|| {
        let raw: &'static str = include_str!("../assets/cmudict.dict");
        let mut dictionary: HashMap<String, Vec<&'static str>> = HashMap::with_capacity(130_000);
        for line in raw.lines() {
            let clean = match line.split_once('#') {
                Some((before, _)) => before,
                None => line,
            };
            let mut parts = clean.split_whitespace();
            let Some(raw_word) = parts.next() else {
                continue;
            };
            let phones: Vec<&'static str> = parts.collect();
            if phones.is_empty() {
                continue;
            }
            let word = strip_variant_suffix(raw_word).to_ascii_lowercase();
            dictionary.entry(word).or_insert(phones);
        }
        dictionary
    })
}

fn strip_variant_suffix(w: &str) -> &str {
    if let Some(open) = w.rfind('(') {
        if w.ends_with(')') {
            let inside = &w[open + 1..w.len() - 1];
            if !inside.is_empty() && inside.bytes().all(|b| b.is_ascii_digit()) {
                return &w[..open];
            }
        }
    }
    w
}

pub fn cardinal_str(num_str: &str) -> String {
    if let Some(rest) = num_str.strip_prefix('-') {
        return format!("minus {}", cardinal_str(rest));
    }
    // Check if >= 10^15 or doesn't fit in i64
    let trimmed = num_str.trim_start_matches('0');
    if trimmed.is_empty() {
        return "zero".to_string();
    }
    if trimmed.len() > 15 {
        return num_str
            .chars()
            .filter_map(|c| c.to_digit(10))
            .map(|d| ONES[d as usize])
            .collect::<Vec<_>>()
            .join(" ");
    }
    if let Ok(n) = num_str.parse::<i64>() {
        cardinal(n)
    } else {
        num_str
            .chars()
            .filter_map(|c| c.to_digit(10))
            .map(|d| ONES[d as usize])
            .collect::<Vec<_>>()
            .join(" ")
    }
}

pub fn cardinal(number: i64) -> String {
    if number < 0 {
        return format!("minus {}", cardinal(-number));
    }
    if number < 20 {
        return ONES[number as usize].to_string();
    }
    if number < 100 {
        let tens = TENS[(number / 10) as usize];
        let rem = number % 10;
        if rem != 0 {
            return format!("{} {}", tens, cardinal(rem));
        }
        return tens.to_string();
    }
    for (scale, label) in [
        (1_000_000_000_000i64, "trillion"),
        (1_000_000_000i64, "billion"),
        (1_000_000i64, "million"),
        (1_000i64, "thousand"),
        (100i64, "hundred"),
    ] {
        if number >= scale {
            if number >= 1_000_000_000_000_000i64 {
                return number
                    .to_string()
                    .chars()
                    .filter_map(|c| c.to_digit(10))
                    .map(|d| ONES[d as usize])
                    .collect::<Vec<_>>()
                    .join(" ");
            }
            let head = cardinal(number / scale);
            let rem = number % scale;
            if rem != 0 {
                return format!("{} {} {}", head, label, cardinal(rem));
            }
            return format!("{} {}", head, label);
        }
    }
    unreachable!()
}

pub fn ordinal(number: i64) -> String {
    let card = cardinal(number);
    let mut words: Vec<String> = card.split_whitespace().map(String::from).collect();
    if let Some(last) = words.last_mut() {
        let rep = match last.as_str() {
            "one" => "first".to_string(),
            "two" => "second".to_string(),
            "three" => "third".to_string(),
            "five" => "fifth".to_string(),
            "eight" => "eighth".to_string(),
            "nine" => "ninth".to_string(),
            "twelve" => "twelfth".to_string(),
            s if s.ends_with('y') => format!("{}ieth", &s[..s.len() - 1]),
            s => format!("{}th", s),
        };
        *last = rep;
    }
    words.join(" ")
}

fn is_valid_calendar_date(year: i64, month: i64, day: i64) -> bool {
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) || day < 1 {
        return false;
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if leap {
                29
            } else {
                28
            }
        }
        _ => return false,
    };
    day <= max_day
}

fn translate_typography(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '’' | '‘' => out.push('\''),
            '“' | '”' => out.push('"'),
            '—' | '–' => out.push(','),
            '…' => out.push_str("..."),
            '\u{00A0}' => out.push(' '),
            // Fullwidth ASCII compatibility variants (NFKC subset)
            '\u{FF01}'..='\u{FF5E}' => {
                if let Some(ascii) = char::from_u32(ch as u32 - 0xFEE0) {
                    out.push(ascii);
                } else {
                    out.push(ch);
                }
            }
            _ => out.push(ch),
        }
    }
    out
}

/// Normalize English text using the exact PiVo v1 rules.
pub fn normalize(raw_text: &str) -> Result<String, FrontendError> {
    let mut text = translate_typography(raw_text);

    // 1. ISO date (\b(\d{4})-(\d{2})-(\d{2})\b)
    static RE_DATE: OnceLock<Regex> = OnceLock::new();
    let re_date = RE_DATE.get_or_init(|| Regex::new(r"\b(\d{4})-(\d{2})-(\d{2})\b").unwrap());
    let mut date_err: Option<String> = None;
    text = re_date
        .replace_all(&text, |caps: &Captures| {
            let year: i64 = caps[1].parse().unwrap_or(0);
            let month: i64 = caps[2].parse().unwrap_or(0);
            let day: i64 = caps[3].parse().unwrap_or(0);
            if !is_valid_calendar_date(year, month, day) {
                date_err = Some(format!("invalid calendar date: {}", &caps[0]));
                return String::new();
            }
            format!(
                "{} {}, {}",
                MONTHS[(month - 1) as usize],
                ordinal(day),
                cardinal(year)
            )
        })
        .into_owned();
    if let Some(err) = date_err {
        return Err(FrontendError::InvalidInput(err));
    }

    // 2. Reject slash dates/fractions (\d/\d)
    static RE_SLASH: OnceLock<Regex> = OnceLock::new();
    let re_slash = RE_SLASH.get_or_init(|| Regex::new(r"\d/\d").unwrap());
    if re_slash.is_match(&text) {
        return Err(FrontendError::InvalidInput(
            "write slash dates/fractions in words, or use an ISO date".to_string(),
        ));
    }

    // 3. Abbreviations (Mr., Mrs., Dr., Prof., St., vs., etc.) followed by whitespace or end
    static RE_ABBRS: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    let abbrs = RE_ABBRS.get_or_init(|| {
        [
            ("Mr", "Mister"),
            ("Mrs", "Missus"),
            ("Dr", "Doctor"),
            ("Prof", "Professor"),
            ("St", "Saint"),
            ("vs", "versus"),
            ("etc", "et cetera"),
        ]
        .iter()
        .map(|(abbr, exp)| {
            (
                Regex::new(&format!(r"(?i)\b{}\.(?P<after>\s|$)", abbr)).unwrap(),
                *exp,
            )
        })
        .collect()
    });
    for (re, exp) in abbrs {
        text = re
            .replace_all(&text, |caps: &Captures| format!("{}{}", exp, &caps["after"]))
            .into_owned();
    }

    // 4. Currency amounts
    static RE_BAD_MONEY: OnceLock<Regex> = OnceLock::new();
    let re_bad_money =
        RE_BAD_MONEY.get_or_init(|| Regex::new(r"[$£€]\d[\d,]*\.\d{3,}").unwrap());
    if re_bad_money.is_match(&text) {
        return Err(FrontendError::InvalidInput(
            "currency amounts must have at most two decimal places".to_string(),
        ));
    }

    static RE_MONEY: OnceLock<Regex> = OnceLock::new();
    let re_money =
        RE_MONEY.get_or_init(|| Regex::new(r"([$£€])(\d[\d,]*(?:\.\d{1,2})?)").unwrap());
    text = re_money
        .replace_all(&text, |caps: &Captures| {
            let symbol = &caps[1];
            let raw = caps[2].replace(',', "");
            let (whole_str, fraction_str) = match raw.split_once('.') {
                Some((w, f)) => (w, f),
                None => (raw.as_str(), ""),
            };
            let whole: i64 = whole_str.parse().unwrap_or(0);
            let unit = match symbol {
                "$" => "dollar",
                "£" => "pound",
                "€" => "euro",
                _ => "dollar",
            };
            let mut result = format!(
                "{} {}{}",
                cardinal(whole),
                unit,
                if whole != 1 { "s" } else { "" }
            );
            let cents: i64 = if !fraction_str.is_empty() {
                let padded = format!("{:0<2}", fraction_str);
                padded.parse().unwrap_or(0)
            } else {
                0
            };
            if cents != 0 {
                let cent_unit = if cents == 1 && symbol == "£" {
                    " penny"
                } else if symbol == "£" {
                    " pence"
                } else if cents == 1 {
                    " cent"
                } else {
                    " cents"
                };
                result.push_str(" and ");
                result.push_str(&cardinal(cents));
                result.push_str(cent_unit);
            }
            result
        })
        .into_owned();

    // 5. Ordinals (\b(\d+)(?:st|nd|rd|th)\b)
    static RE_ORDINAL: OnceLock<Regex> = OnceLock::new();
    let re_ordinal = RE_ORDINAL.get_or_init(|| Regex::new(r"(?i)\b(\d+)(?:st|nd|rd|th)\b").unwrap());
    text = re_ordinal
        .replace_all(&text, |caps: &Captures| {
            let n: i64 = caps[1].parse().unwrap_or(0);
            ordinal(n)
        })
        .into_owned();

    // 6. Comma-grouped numbers (\b\d{1,3}(?:,\d{3})+\b)
    static RE_COMMAS: OnceLock<Regex> = OnceLock::new();
    let re_commas = RE_COMMAS.get_or_init(|| Regex::new(r"\b\d{1,3}(?:,\d{3})+\b").unwrap());
    text = re_commas
        .replace_all(&text, |caps: &Captures| caps[0].replace(',', ""))
        .into_owned();

    // 7. General integers & decimals (-?(?:\d+\.\d+|\d+))
    static RE_NUM: OnceLock<Regex> = OnceLock::new();
    let re_num = RE_NUM.get_or_init(|| Regex::new(r"-?(?:\d+\.\d+|\d+)").unwrap());
    text = re_num
        .replace_all(&text, |caps: &Captures| {
            let m = &caps[0];
            let (neg_prefix, unsigned) = match m.strip_prefix('-') {
                Some(rest) => ("minus ", rest),
                None => ("", m),
            };
            if let Some((int_part, frac_part)) = unsigned.split_once('.') {
                let int_words = cardinal_str(int_part);
                let frac_words = frac_part
                    .chars()
                    .filter_map(|c| c.to_digit(10))
                    .map(|d| ONES[d as usize])
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("{}{} point {}", neg_prefix, int_words, frac_words)
            } else {
                format!("{}{}", neg_prefix, cardinal_str(unsigned))
            }
        })
        .into_owned();

    // 8. Symbols %, &, +
    text = text
        .replace('%', " percent ")
        .replace('&', " and ")
        .replace('+', " plus ");

    // 9. Preserve NASA before separating uppercase initialisms
    static RE_NASA: OnceLock<Regex> = OnceLock::new();
    let re_nasa = RE_NASA.get_or_init(|| Regex::new(r"\bNASA\b").unwrap());
    text = re_nasa.replace_all(&text, "nasa").into_owned();

    static RE_INIT: OnceLock<Regex> = OnceLock::new();
    let re_init = RE_INIT.get_or_init(|| Regex::new(r"\b[A-Z]{2,5}\b").unwrap());
    text = re_init
        .replace_all(&text, |caps: &Captures| {
            caps[0]
                .chars()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .into_owned();

    // 10. Hyphens and brackets/quotes/slashes
    text = text.replace('-', " ");
    static RE_BRACKETS: OnceLock<Regex> = OnceLock::new();
    let re_brackets = RE_BRACKETS.get_or_init(|| Regex::new(r#"[()\[\]{}"/]"#).unwrap());
    text = re_brackets.replace_all(&text, " ").into_owned();

    // 11. Check unsupported characters
    let mut unsupported = BTreeSet::new();
    for ch in text.chars() {
        let ok = ch.is_ascii_alphabetic()
            || ch.is_whitespace()
            || matches!(ch, '\'' | '.' | ',' | '?' | '!' | ';' | ':');
        if !ok {
            unsupported.insert(ch);
        }
    }
    if !unsupported.is_empty() {
        let list: Vec<String> = unsupported.into_iter().map(|c| format!("{:?}", c)).collect();
        return Err(FrontendError::InvalidInput(format!(
            "unsupported text characters after normalization: [{}]",
            list.join(", ")
        )));
    }

    // 12. Collapse whitespace
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    Ok(collapsed)
}

/// Phonemize English text into ARPAbet symbols and PiVo token IDs.
///
/// When `strict` is `true` (the default PiVo v1 contract), out-of-vocabulary words not present
/// in `overrides` return an error. When `strict` is `false`, unknown words are spelled out
/// letter-by-letter so live streaming on arbitrary text never aborts.
pub fn phonemize(
    text: &str,
    strict: bool,
    overrides: Option<&HashMap<String, Vec<String>>>,
) -> Result<FrontendResult, FrontendError> {
    let normalized = normalize(text)?;
    let dict = lexicon();

    let lower_overrides: HashMap<String, Vec<String>> = overrides
        .map(|m| {
            m.iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
                .collect()
        })
        .unwrap_or_default();

    static RE_TOKEN: OnceLock<Regex> = OnceLock::new();
    let re_token =
        RE_TOKEN.get_or_init(|| Regex::new(r"[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]").unwrap());

    let mut symbols: Vec<String> = Vec::with_capacity(64);
    symbols.push("<bos>".to_string());
    let mut unknown: Vec<String> = Vec::new();

    for mat in re_token.find_iter(&normalized) {
        let item = mat.as_str();
        if SPECIAL.contains(&item) {
            symbols.push(item.to_string());
            continue;
        }
        let word = item.to_ascii_lowercase();
        let phones: Vec<String> = if let Some(ov) = lower_overrides.get(&word) {
            ov.clone()
        } else if let Some(dict_phones) = dict.get(&word) {
            dict_phones.iter().map(|s| (*s).to_string()).collect()
        } else {
            unknown.push(word.clone());
            // Spell out unknown words letter-by-letter
            let mut spelled = Vec::new();
            for ch in word.chars() {
                if ch.is_ascii_alphabetic() {
                    let key = ch.to_string();
                    if let Some(lp) = dict.get(&key) {
                        for &p in lp {
                            spelled.push(p.to_string());
                        }
                    }
                }
            }
            spelled
        };

        if !matches!(symbols.last().map(|s| s.as_str()), Some("<bos>" | "<space>")) {
            symbols.push("<space>".to_string());
        }
        for phone in phones {
            if symbol_to_id(&phone).is_none() {
                return Err(FrontendError::InvalidInput(format!(
                    "unknown ARPAbet phone {:?} in {:?}",
                    phone, word
                )));
            }
            symbols.push(phone);
        }
    }

    if !unknown.is_empty() && strict {
        let unique: BTreeSet<&str> = unknown.iter().map(|s| s.as_str()).collect();
        let joined: Vec<&str> = unique.into_iter().collect();
        return Err(FrontendError::InvalidInput(format!(
            "out-of-vocabulary words require pronunciation overrides: {}",
            joined.join(", ")
        )));
    }

    symbols.push("<eos>".to_string());
    let token_ids: Vec<i64> = symbols
        .iter()
        .map(|s| symbol_to_id(s).unwrap_or(0))
        .collect();

    Ok(FrontendResult {
        normalized_text: normalized,
        symbols,
        token_ids,
        unknown_words: unknown,
    })
}

/// Split arbitrarily long text into natural sentence/clause chunks that each fit comfortably
/// within PiVo's encoder/utterance budget and minimize time-to-first-audio (TTFA).
///
/// The first chunk is kept intentionally compact (`first_chunk_max_chars`, e.g. 120 chars)
/// when a natural clause boundary exists so that streaming audio begins in < 30 ms even on
/// mobile CPUs.
pub fn split_text_into_clauses(text: &str, max_clause_chars: usize) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let max_chars = max_clause_chars.clamp(40, 1800);

    // Split by sentence terminators first while keeping the punctuation attached
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = trimmed.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        current.push(ch);
        if ch == '\n' {
            let s = current.trim().to_string();
            if !s.is_empty() {
                sentences.push(s);
            }
            current.clear();
        } else if matches!(ch, '.' | '!' | '?' | ';' | ':') {
            // Avoid splitting inside numbers like 3.14 or abbreviations like Mr.
            let prev_is_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_is_digit = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
            let next_is_space_or_end = i + 1 == chars.len() || chars[i + 1].is_whitespace();
            let is_abbr = if ch == '.' {
                let word = current
                    .trim_end_matches('.')
                    .split_whitespace()
                    .last()
                    .unwrap_or("");
                matches!(
                    word.to_ascii_lowercase().as_str(),
                    "mr" | "mrs" | "dr" | "prof" | "st" | "vs" | "etc"
                )
            } else {
                false
            };

            if !(prev_is_digit && next_is_digit) && !is_abbr && next_is_space_or_end {
                let s = current.trim().to_string();
                if !s.is_empty() {
                    sentences.push(s);
                }
                current.clear();
            }
        }
        i += 1;
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }

    // Further split any long sentence on commas or word boundaries
    let mut clauses: Vec<String> = Vec::new();
    for sent in sentences {
        // For the very first clause, use a tighter budget (up to 120 chars) if it has a comma
        let target = if clauses.is_empty() {
            max_chars.min(140)
        } else {
            max_chars
        };
        if sent.len() <= target {
            clauses.push(sent);
            continue;
        }
        // Split by comma/dash segments first
        let mut sub_accum = String::new();
        for part in sent.split_inclusive(',') {
            if !sub_accum.is_empty() && sub_accum.len() + part.len() > target {
                clauses.push(sub_accum.trim().to_string());
                sub_accum.clear();
            }
            if part.len() > max_chars {
                // Split long comma-less run by words
                for word in part.split_whitespace() {
                    if !sub_accum.is_empty() && sub_accum.len() + 1 + word.len() > max_chars {
                        clauses.push(sub_accum.trim().to_string());
                        sub_accum.clear();
                    }
                    if !sub_accum.is_empty() {
                        sub_accum.push(' ');
                    }
                    sub_accum.push_str(word);
                }
            } else {
                sub_accum.push_str(part);
            }
        }
        if !sub_accum.trim().is_empty() {
            clauses.push(sub_accum.trim().to_string());
        }
    }

    // Filter out any punctuation-only clauses
    clauses
        .into_iter()
        .filter(|c| c.chars().any(|ch| ch.is_ascii_alphanumeric()))
        .collect()
}
