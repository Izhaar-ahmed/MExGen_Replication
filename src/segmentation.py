"""
spaCy-based text segmentation for MExGen replication.
Supports sentence, phrase, and word segmentation.
"""

import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split text into sentences. Returns list of (text, start, end) tuples."""
    nlp = _get_nlp()
    doc = nlp(text)
    return [(sent.text.strip(), sent.start_char, sent.end_char) for sent in doc.sents if sent.text.strip()]


def split_phrases(sentence: str, max_phrase_tokens: int = 10) -> list[tuple[str, int, int]]:
    """
    Split a sentence into phrases using dependency parse subtrees.
    Max phrase length = max_phrase_tokens tokens.
    """
    nlp = _get_nlp()
    doc = nlp(sentence)

    phrases = []
    used = set()

    for token in doc:
        if token.i in used:
            continue
        # Get subtree tokens
        subtree = list(token.subtree)
        if len(subtree) <= max_phrase_tokens:
            start = subtree[0].idx
            end = subtree[-1].idx + len(subtree[-1].text)
            phrase_text = sentence[start:end].strip()
            if phrase_text and not all(t.i in used for t in subtree):
                phrases.append((phrase_text, start, end))
                for t in subtree:
                    used.add(t.i)

    # If no phrases found, fallback to entire sentence
    if not phrases:
        phrases = [(sentence.strip(), 0, len(sentence))]

    return phrases


def split_words(sentence: str) -> list[tuple[str, int, int]]:
    """Split a sentence into individual words/tokens."""
    nlp = _get_nlp()
    doc = nlp(sentence)
    return [(token.text, token.idx, token.idx + len(token.text)) for token in doc if not token.is_space]


def segment_text(text: str, mode: str = "sentence") -> list[str]:
    """
    Segment text into units based on mode.
    Modes: 'sentence', 'sentence_phrase' (XSUM), 'sentence_word' (SQuAD)
    Returns list of unit text strings.
    """
    sentences = split_sentences(text)

    if mode == "sentence":
        return [s[0] for s in sentences]

    elif mode == "sentence_phrase":
        units = []
        for sent_text, _, _ in sentences:
            phrases = split_phrases(sent_text)
            if len(phrases) <= 1:
                units.append(sent_text)
            else:
                for phrase_text, _, _ in phrases:
                    units.append(phrase_text)
        return units

    elif mode == "sentence_word":
        units = []
        for sent_text, _, _ in sentences:
            words = split_words(sent_text)
            if len(words) <= 3:
                units.append(sent_text)
            else:
                # Keep sentence-level for short sentences, word-level for longer
                for word_text, _, _ in words:
                    units.append(word_text)
        return units

    else:
        raise ValueError(f"Unknown segmentation mode: {mode}")


if __name__ == "__main__":
    from src.data_loader import load_xsum

    samples = load_xsum(n_samples=5)
    for i, s in enumerate(samples):
        print(f"\n=== XSUM Sample {i} ===")
        units = segment_text(s["input"], mode="sentence_phrase")
        print(f"  {len(units)} units:")
        for j, u in enumerate(units[:5]):
            print(f"    [{j}] {u[:80]}{'...' if len(u) > 80 else ''}")
        if len(units) > 5:
            print(f"    ... and {len(units) - 5} more")
