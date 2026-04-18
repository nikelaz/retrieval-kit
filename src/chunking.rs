#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkingConfig {
    pub chunk_size: usize,
    pub overlap_size: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1500,
            overlap_size: 200,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChunkingError {
    InvalidChunkSize,
    InvalidOverlapSize {
        chunk_size: usize,
        overlap_size: usize,
    },
}

impl std::fmt::Display for ChunkingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidChunkSize => write!(f, "chunk_size must be greater than zero"),
            Self::InvalidOverlapSize {
                chunk_size,
                overlap_size,
            } => write!(
                f,
                "overlap_size ({overlap_size}) must be smaller than chunk_size ({chunk_size})"
            ),
        }
    }
}

impl std::error::Error for ChunkingError {}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum BoundaryStrength {
    Whitespace,
    Weak,
    Medium,
    Strong,
}

const LOOKBACK_WINDOW: usize = 400;
const LOOKAHEAD_WINDOW: usize = 200;
const OVERLAP_WINDOW: usize = 120;

pub fn chunk_text(text: &str, config: ChunkingConfig) -> Result<Vec<String>, ChunkingError> {
    validate_config(config)?;

    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let chars: Vec<char> = text.chars().collect();
    let total_chars = chars.len();
    let mut boundaries = collect_boundaries(text, &chars);
    boundaries.push(text.len());

    let mut chunks = Vec::new();
    let mut start_char = first_non_whitespace_char(&chars, 0);

    while start_char < total_chars {
        let end_char = choose_chunk_end(&chars, &boundaries, start_char, config.chunk_size);
        let start_byte = char_to_byte_idx(text, start_char);
        let end_byte = char_to_byte_idx(text, end_char);

        let chunk = text[start_byte..end_byte].trim();
        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }

        if end_char >= total_chars {
            break;
        }

        let next_start = choose_overlap_start(
            &chars,
            &boundaries,
            start_char,
            end_char,
            config.overlap_size,
        );

        if next_start <= start_char {
            start_char = first_non_whitespace_char(&chars, end_char);
        } else {
            start_char = first_non_whitespace_char(&chars, next_start);
        }
    }

    Ok(chunks)
}

fn validate_config(config: ChunkingConfig) -> Result<(), ChunkingError> {
    if config.chunk_size == 0 {
        return Err(ChunkingError::InvalidChunkSize);
    }

    if config.overlap_size >= config.chunk_size {
        return Err(ChunkingError::InvalidOverlapSize {
            chunk_size: config.chunk_size,
            overlap_size: config.overlap_size,
        });
    }

    Ok(())
}

fn collect_boundaries(text: &str, chars: &[char]) -> Vec<usize> {
    let mut boundaries = Vec::new();

    for boundary in 1..chars.len() {
        if classify_boundary(chars, boundary).is_some() {
            boundaries.push(char_to_byte_idx(text, boundary));
        }
    }

    boundaries
}

fn choose_chunk_end(
    chars: &[char],
    boundaries: &[usize],
    start_char: usize,
    chunk_size: usize,
) -> usize {
    let total_chars = chars.len();
    let ideal_end = (start_char + chunk_size).min(total_chars);

    if ideal_end >= total_chars {
        return total_chars;
    }

    let lookback_start = (ideal_end.saturating_sub(LOOKBACK_WINDOW)).max(start_char + 1);
    let before = best_boundary_in_range(chars, boundaries, lookback_start, ideal_end, true);
    if let Some(boundary) = before {
        return boundary;
    }

    let lookahead_end = (ideal_end + LOOKAHEAD_WINDOW).min(total_chars);
    let after = best_boundary_in_range(chars, boundaries, ideal_end + 1, lookahead_end, false);
    if let Some(boundary) = after {
        return boundary;
    }

    nearest_safe_boundary(chars, start_char + 1, total_chars, ideal_end).unwrap_or(ideal_end)
}

fn choose_overlap_start(
    chars: &[char],
    boundaries: &[usize],
    start_char: usize,
    end_char: usize,
    overlap_size: usize,
) -> usize {
    if overlap_size == 0 || end_char <= start_char + 1 {
        return end_char;
    }

    let desired_start = end_char.saturating_sub(overlap_size).max(start_char + 1);
    let search_start = desired_start
        .saturating_sub(OVERLAP_WINDOW)
        .max(start_char + 1);
    let search_end = (desired_start + OVERLAP_WINDOW).min(end_char.saturating_sub(1));

    if search_start <= search_end
        && let Some(boundary) =
            nearest_semantic_boundary(chars, boundaries, search_start, search_end, desired_start)
    {
        return boundary;
    }

    end_char
}

fn best_boundary_in_range(
    chars: &[char],
    boundaries: &[usize],
    start_char: usize,
    end_char: usize,
    prefer_later: bool,
) -> Option<usize> {
    let mut best: Option<(BoundaryStrength, usize, usize)> = None;

    for boundary_byte in boundaries {
        let boundary_char = byte_to_char_idx(chars, *boundary_byte);
        if boundary_char < start_char || boundary_char > end_char {
            continue;
        }

        let Some(strength) = classify_boundary(chars, boundary_char) else {
            continue;
        };

        let distance = if prefer_later {
            end_char.saturating_sub(boundary_char)
        } else {
            boundary_char.saturating_sub(start_char)
        };

        match best {
            None => best = Some((strength, distance, boundary_char)),
            Some((best_strength, best_distance, best_boundary)) => {
                let should_replace = strength > best_strength
                    || (strength == best_strength
                        && (distance < best_distance
                            || (distance == best_distance
                                && ((prefer_later && boundary_char > best_boundary)
                                    || (!prefer_later && boundary_char < best_boundary)))));
                if should_replace {
                    best = Some((strength, distance, boundary_char));
                }
            }
        }
    }

    best.map(|(_, _, boundary)| boundary)
}

fn nearest_semantic_boundary(
    chars: &[char],
    boundaries: &[usize],
    start_char: usize,
    end_char: usize,
    desired_char: usize,
) -> Option<usize> {
    let mut best: Option<(BoundaryStrength, usize, usize)> = None;

    for boundary_byte in boundaries {
        let boundary_char = byte_to_char_idx(chars, *boundary_byte);
        if boundary_char < start_char || boundary_char > end_char {
            continue;
        }

        let Some(strength) = classify_boundary(chars, boundary_char) else {
            continue;
        };

        let distance = boundary_char.abs_diff(desired_char);
        match best {
            None => best = Some((strength, distance, boundary_char)),
            Some((best_strength, best_distance, best_boundary)) => {
                let should_replace = strength > best_strength
                    || (strength == best_strength
                        && (distance < best_distance
                            || (distance == best_distance && boundary_char < best_boundary)));
                if should_replace {
                    best = Some((strength, distance, boundary_char));
                }
            }
        }
    }

    best.map(|(_, _, boundary)| boundary)
}

fn nearest_safe_boundary(
    chars: &[char],
    start_char: usize,
    end_char: usize,
    desired_char: usize,
) -> Option<usize> {
    if start_char > end_char {
        return None;
    }

    let mut best_whitespace: Option<(usize, usize)> = None;

    for boundary in start_char..=end_char {
        let Some(strength) = classify_boundary(chars, boundary) else {
            continue;
        };

        if strength == BoundaryStrength::Whitespace {
            let distance = boundary.abs_diff(desired_char);
            match best_whitespace {
                None => best_whitespace = Some((distance, boundary)),
                Some((best_distance, best_boundary)) => {
                    if distance < best_distance
                        || (distance == best_distance && boundary < best_boundary)
                    {
                        best_whitespace = Some((distance, boundary));
                    }
                }
            }
        }
    }

    if let Some((_, boundary)) = best_whitespace {
        return Some(boundary);
    }

    Some(desired_char.clamp(start_char, end_char))
}

fn classify_boundary(chars: &[char], boundary: usize) -> Option<BoundaryStrength> {
    if boundary == 0 || boundary >= chars.len() {
        return None;
    }

    let prev = chars[boundary - 1];
    let next = chars[boundary];

    if is_paragraph_break(chars, boundary) {
        return Some(BoundaryStrength::Strong);
    }

    if prev == '\n' {
        return Some(BoundaryStrength::Strong);
    }

    let semantic_prev = previous_semantic_char(chars, boundary - 1);
    if matches!(semantic_prev, Some('.' | '!' | '?')) && next.is_whitespace() {
        return Some(BoundaryStrength::Medium);
    }

    if matches!(semantic_prev, Some(';' | ':' | ',')) && next.is_whitespace() {
        return Some(BoundaryStrength::Weak);
    }

    if prev.is_whitespace() {
        return Some(BoundaryStrength::Whitespace);
    }

    None
}

fn is_paragraph_break(chars: &[char], boundary: usize) -> bool {
    if chars[boundary - 1] != '\n' {
        return false;
    }

    let mut idx = boundary;
    while idx < chars.len() {
        let ch = chars[idx];
        if ch == '\n' {
            return true;
        }
        if !ch.is_whitespace() {
            return false;
        }
        idx += 1;
    }

    false
}

fn previous_semantic_char(chars: &[char], mut idx: usize) -> Option<char> {
    loop {
        let ch = chars[idx];
        if !matches!(ch, '"' | '\'' | ')' | ']' | '}') {
            return Some(ch);
        }
        if idx == 0 {
            return None;
        }
        idx -= 1;
    }
}

fn first_non_whitespace_char(chars: &[char], mut idx: usize) -> usize {
    while idx < chars.len() && chars[idx].is_whitespace() {
        idx += 1;
    }
    idx
}

fn char_to_byte_idx(text: &str, char_idx: usize) -> usize {
    if char_idx == 0 {
        return 0;
    }

    text.char_indices()
        .nth(char_idx)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len())
}

fn byte_to_char_idx(chars: &[char], byte_idx: usize) -> usize {
    let mut total = 0;
    for (idx, ch) in chars.iter().enumerate() {
        if total == byte_idx {
            return idx;
        }
        total += ch.len_utf8();
    }
    chars.len()
}

#[cfg(test)]
mod tests {
    use super::{ChunkingConfig, ChunkingError, chunk_text};

    #[test]
    fn uses_expected_defaults() {
        let config = ChunkingConfig::default();

        assert_eq!(config.chunk_size, 1500);
        assert_eq!(config.overlap_size, 200);
    }

    #[test]
    fn rejects_zero_chunk_size() {
        let error = chunk_text(
            "hello world",
            ChunkingConfig {
                chunk_size: 0,
                overlap_size: 0,
            },
        )
        .unwrap_err();

        assert_eq!(error, ChunkingError::InvalidChunkSize);
    }

    #[test]
    fn rejects_overlap_equal_to_chunk_size() {
        let error = chunk_text(
            "hello world",
            ChunkingConfig {
                chunk_size: 200,
                overlap_size: 200,
            },
        )
        .unwrap_err();

        assert_eq!(
            error,
            ChunkingError::InvalidOverlapSize {
                chunk_size: 200,
                overlap_size: 200,
            }
        );
    }

    #[test]
    fn returns_single_chunk_for_short_input() {
        let chunks = chunk_text(
            "Small inputs should stay together.",
            ChunkingConfig {
                chunk_size: 100,
                overlap_size: 20,
            },
        )
        .unwrap();

        assert_eq!(chunks, vec!["Small inputs should stay together."]);
    }

    #[test]
    fn prefers_paragraph_boundaries() {
        let text = "First paragraph has a few sentences. It should stay grouped.\n\nSecond paragraph also has enough text to trigger chunking when the target is small.";
        let chunks = chunk_text(
            text,
            ChunkingConfig {
                chunk_size: 70,
                overlap_size: 15,
            },
        )
        .unwrap();

        assert!(chunks.len() >= 2);
        assert!(chunks[0].ends_with("grouped."));
        assert!(
            chunks[1].starts_with("Second paragraph") || chunks[1].contains("Second paragraph")
        );
    }

    #[test]
    fn prefers_sentence_boundaries() {
        let text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu.";
        let chunks = chunk_text(
            text,
            ChunkingConfig {
                chunk_size: 30,
                overlap_size: 8,
            },
        )
        .unwrap();

        assert!(chunks.len() >= 2);
        assert!(chunks[0].ends_with('.'));
        assert!(chunks[1].contains("Delta epsilon zeta.") || chunks[1].contains("Eta theta iota."));
    }

    #[test]
    fn overlap_is_approximate_and_word_safe() {
        let text = "This is the first sentence. This is the second sentence with more words. This is the third sentence with even more words.";
        let chunks = chunk_text(
            text,
            ChunkingConfig {
                chunk_size: 55,
                overlap_size: 12,
            },
        )
        .unwrap();

        assert!(chunks.len() >= 2);
        assert!(
            chunks[0].split_whitespace().last().is_some()
                && chunks[1].split_whitespace().next().is_some()
        );
        let trailing_words = normalized_words(chunks[0].as_str());
        let leading_words = normalized_words(chunks[1].as_str());
        assert!(
            trailing_words
                .iter()
                .rev()
                .take(4)
                .any(|word| word.len() > 3 && leading_words.contains(word))
        );
    }

    #[test]
    fn falls_back_to_whitespace_for_punctuation_poor_text() {
        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu";
        let chunks = chunk_text(
            text,
            ChunkingConfig {
                chunk_size: 24,
                overlap_size: 6,
            },
        )
        .unwrap();

        assert!(chunks.len() >= 2);
        assert!(!chunks.iter().any(|chunk| chunk.contains("  ")));
        assert!(chunks.iter().all(|chunk| !chunk.starts_with(' ')));
        assert!(chunks.iter().all(|chunk| !chunk.ends_with(' ')));
    }

    #[test]
    fn progresses_even_without_whitespace() {
        let text = "supercalifragilisticexpialidociousandbeyond";
        let chunks = chunk_text(
            text,
            ChunkingConfig {
                chunk_size: 10,
                overlap_size: 3,
            },
        )
        .unwrap();

        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| !chunk.is_empty()));
        assert_eq!(chunks.join(""), text);
    }

    #[test]
    fn handles_unicode_without_invalid_boundaries() {
        let text = "Здравей свят. Добре дошли в retrieval kit. Това е тест.";
        let chunks = chunk_text(
            text,
            ChunkingConfig {
                chunk_size: 24,
                overlap_size: 6,
            },
        )
        .unwrap();

        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|chunk| !chunk.is_empty()));
    }

    #[test]
    fn returns_empty_for_whitespace_only_input() {
        let chunks = chunk_text("   \n\t  ", ChunkingConfig::default()).unwrap();

        assert!(chunks.is_empty());
    }

    fn normalized_words(input: &str) -> Vec<String> {
        input
            .split_whitespace()
            .map(|word| {
                word.trim_matches(|ch: char| !ch.is_alphanumeric())
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty())
            .collect()
    }
}
