//! Frontend contract tests — must match huggingface/pivo/text.py semantics exactly.

use pivo_runner::text::{normalize, phonemize, split_text_into_clauses};

#[test]
fn nasa_and_initialisms() {
    assert_eq!(normalize("NASA and CPU").unwrap(), "nasa and C P U");
}

#[test]
fn oov_strict_and_override() {
    assert!(phonemize("Hello, Zzvo.", true, None).is_err());
    let mut ov = std::collections::HashMap::new();
    ov.insert("zzvo".to_string(), vec!["Z".into(), "IY1".into(), "V".into(), "OW0".into()]);
    let front = phonemize("Hello, Zzvo.", true, Some(&ov)).unwrap();
    assert!(front.unknown_words.is_empty());
    let spoken: Vec<_> = front
        .symbols
        .iter()
        .filter(|s| !pivo_runner::text::SPECIAL.contains(&s.as_str()))
        .cloned()
        .collect();
    assert_eq!(spoken, vec!["HH", "AH0", "L", "OW1", "Z", "IY1", "V", "OW0"]);
}

#[test]
fn nasa_phonemes() {
    let front = phonemize("NASA", true, None).unwrap();
    let spoken: Vec<_> = front
        .symbols
        .iter()
        .filter(|s| !pivo_runner::text::SPECIAL.contains(&s.as_str()))
        .cloned()
        .collect();
    assert_eq!(spoken, vec!["N", "AE1", "S", "AH0"]);
}

#[test]
fn numbers_currency_ordinals_dates() {
    assert_eq!(
        normalize("On March 4th, 2026, it cost $1,250.75.").unwrap(),
        "On March fourth, two thousand twenty six, it cost one thousand two hundred fifty dollars and seventy five cents."
    );
    assert_eq!(
        normalize("He was born on 2001-11-03 and paid $0.99.").unwrap(),
        "He was born on November third, two thousand one and paid zero dollars and ninety nine cents."
    );
    assert!(normalize("the date is 2026-02-30").is_err(), "invalid calendar date must error");
    assert!(normalize("1/2 of it").is_err(), "slash fractions must error");
    assert!(normalize("$5.999 too precise").is_err());
    assert_eq!(normalize("£2").unwrap(), "two pounds");
    assert_eq!(normalize("£1").unwrap(), "one pound");
}

#[test]
fn relaxed_oov_spells_out() {
    let strict_err = phonemize("the word xqz", true, None);
    assert!(strict_err.is_err());
    let front = phonemize("the word xqz", false, None).unwrap();
    assert!(!front.symbols.is_empty());
    assert_eq!(front.unknown_words, vec!["xqz"]);
}

#[test]
fn clause_splitter_bounds() {
    let long = "First sentence is short. ".to_string()
        + &"A very long clause with many words and no punctuation at all to see how the splitter handles extremely lengthy text input ".repeat(12)
        + "Done.";
    let clauses = split_text_into_clauses(&long, 900);
    assert!(clauses.len() >= 3);
    assert!(clauses.iter().all(|c| c.len() <= 900));
    assert_eq!(clauses[0], "First sentence is short.");
    assert!(clauses.last().unwrap().ends_with("Done."));
    assert!(split_text_into_clauses("  \n  ", 900).is_empty());
    // punctuation-only input filters out
    assert!(split_text_into_clauses("...", 900).is_empty());
}
