from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import statistics
import unicodedata

from pypinyin import Style, lazy_pinyin


@dataclass(slots=True)
class WordSpan:
    text: str
    start: float
    end: float
    probability: float = 0.0


@dataclass(slots=True)
class TimedCharacter:
    char: str
    phonetic: str
    start: float
    end: float
    probability: float


@dataclass(slots=True)
class TimedLine:
    index: int
    text: str
    start: float
    end: float
    confidence: float
    matched_characters: int
    total_characters: int
    matched_phonetics: int = 0
    text_confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def clean_lyrics(raw: str) -> list[str]:
    """Return meaningful lyric lines while retaining the user's exact wording."""
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    return [line.strip() for line in normalized.split("\n") if line.strip()]


def _is_lyric_character(char: str) -> bool:
    if char.isspace():
        return False
    category = unicodedata.category(char)
    return category[0] in {"L", "N"}


def normalize_char(char: str) -> str:
    return unicodedata.normalize("NFKC", char).casefold()


def normalized_characters(text: str) -> list[str]:
    result: list[str] = []
    for char in text:
        normalized = normalize_char(char)
        result.extend(c for c in normalized if _is_lyric_character(c))
    return result


def phonetic_token(char: str) -> str:
    """Return a tone-free token so Chinese homophones share one anchor."""
    phonetic = lazy_pinyin(
        char,
        style=Style.NORMAL,
        errors=lambda value: list(value),
    )
    return (phonetic[0] if phonetic else char).casefold().replace("ü", "v")


def phonetic_tokens(text: str) -> tuple[list[str], list[str]]:
    characters = normalized_characters(text)
    return characters, [phonetic_token(char) for char in characters]


def expand_words(words: list[WordSpan]) -> list[TimedCharacter]:
    """Expand Whisper word/phrase spans into approximate per-character spans."""
    characters: list[TimedCharacter] = []
    for word in words:
        chars = normalized_characters(word.text)
        if not chars:
            continue
        duration = max(word.end - word.start, 0.04 * len(chars))
        step = duration / len(chars)
        for index, char in enumerate(chars):
            characters.append(
                TimedCharacter(
                    char=char,
                    phonetic=phonetic_token(char),
                    start=max(0.0, word.start + index * step),
                    end=max(0.0, word.start + (index + 1) * step),
                    probability=max(0.0, min(1.0, word.probability)),
                )
            )
    return characters


def _align_indices(expected: list[str], observed: list[str]) -> list[tuple[int, int, bool]]:
    """Globally align token sequences and return diagonal mappings.

    A compact traceback table keeps memory bounded for normal song lyrics. Very
    large inputs fall back to SequenceMatcher to avoid quadratic runtime.
    """
    n, m = len(expected), len(observed)
    if not n or not m:
        return []

    if n * m > 12_000_000:
        matcher = SequenceMatcher(None, expected, observed, autojunk=False)
        return [
            (block.a + offset, block.b + offset, True)
            for block in matcher.get_matching_blocks()
            for offset in range(block.size)
        ]

    width = m + 1
    directions = bytearray((n + 1) * width)
    # 1 = diagonal, 2 = up/delete expected, 3 = left/insert observed
    for i in range(1, n + 1):
        directions[i * width] = 2
    for j in range(1, m + 1):
        directions[j] = 3

    gap, match, mismatch = -2, 4, -3
    previous = [j * gap for j in range(m + 1)]
    for i, expected_token in enumerate(expected, start=1):
        current = [i * gap] + [0] * m
        row_offset = i * width
        for j, observed_token in enumerate(observed, start=1):
            diagonal = previous[j - 1] + (
                match if expected_token == observed_token else mismatch
            )
            deletion = previous[j] + gap
            insertion = current[j - 1] + gap
            best = max(diagonal, deletion, insertion)
            current[j] = best
            if diagonal == best:
                directions[row_offset + j] = 1
            elif deletion == best:
                directions[row_offset + j] = 2
            else:
                directions[row_offset + j] = 3
        previous = current

    mappings: list[tuple[int, int, bool]] = []
    i, j = n, m
    while i > 0 or j > 0:
        direction = directions[i * width + j]
        if direction == 1:
            mappings.append((i - 1, j - 1, expected[i - 1] == observed[j - 1]))
            i -= 1
            j -= 1
        elif direction == 2:
            i -= 1
        elif direction == 3:
            j -= 1
        elif i:
            i -= 1
        else:
            j -= 1
    mappings.reverse()
    return mappings


def _typical_character_duration(observed: list[TimedCharacter]) -> float:
    durations = [c.end - c.start for c in observed if 0.02 <= c.end - c.start <= 1.5]
    return max(0.06, min(0.8, statistics.median(durations) if durations else 0.22))


def _fill_character_times(
    expected_count: int,
    mappings: list[tuple[int, int, bool]],
    observed: list[TimedCharacter],
    audio_duration: float,
) -> list[tuple[float, float, bool, float]]:
    typical = _typical_character_duration(observed)
    anchors: dict[int, tuple[float, float, bool, float]] = {}
    for expected_index, observed_index, exact in mappings:
        span = observed[observed_index]
        anchors[expected_index] = (span.start, span.end, exact, span.probability)

    if not anchors:
        usable_duration = max(audio_duration, expected_count * typical)
        step = usable_duration / max(expected_count, 1)
        return [
            (i * step, min(usable_duration, (i + 1) * step), False, 0.0)
            for i in range(expected_count)
        ]

    anchor_indices = sorted(anchors)
    result: list[tuple[float, float, bool, float]] = []
    for index in range(expected_count):
        if index in anchors:
            result.append(anchors[index])
            continue

        left_candidates = [value for value in anchor_indices if value < index]
        right_candidates = [value for value in anchor_indices if value > index]
        left = left_candidates[-1] if left_candidates else None
        right = right_candidates[0] if right_candidates else None

        if left is not None and right is not None:
            left_center = (anchors[left][0] + anchors[left][1]) / 2
            right_center = (anchors[right][0] + anchors[right][1]) / 2
            ratio = (index - left) / (right - left)
            center = left_center + (right_center - left_center) * ratio
        elif left is not None:
            center = (anchors[left][0] + anchors[left][1]) / 2 + (index - left) * typical
        else:
            assert right is not None
            center = (anchors[right][0] + anchors[right][1]) / 2 - (right - index) * typical

        start = max(0.0, center - typical / 2)
        end = min(max(audio_duration, start + typical), start + typical)
        result.append((start, max(start + 0.02, end), False, 0.0))
    return result


def _repair_repeated_transitions(lines: list[TimedLine]) -> None:
    """Repair gross timestamp outliers using the rhythm of repeated lyric pairs.

    Singing ASR sometimes attaches a whole phrase to its held final syllable.
    Repeated choruses give us a strong local reference without assuming a fixed
    tempo for the entire song.
    """
    if len(lines) < 3:
        return
    original_starts = [line.start for line in lines]
    durations = [right - left for left, right in zip(original_starts, original_starts[1:])]
    rates = [
        duration / max(lines[index].total_characters, 1)
        for index, duration in enumerate(durations)
        if 1.5 <= duration <= 10.0
        and 0.15 <= duration / max(lines[index].total_characters, 1) <= 0.9
    ]
    if not rates:
        return
    typical_rate = statistics.median(rates)

    transition_samples: dict[tuple[str, str], list[float]] = {}
    for index, duration in enumerate(durations):
        key = ("".join(normalized_characters(lines[index].text)), "".join(normalized_characters(lines[index + 1].text)))
        transition_samples.setdefault(key, []).append(duration)

    for index, original_duration in enumerate(durations):
        key = ("".join(normalized_characters(lines[index].text)), "".join(normalized_characters(lines[index + 1].text)))
        samples = transition_samples[key]
        if len(samples) < 2:
            continue
        expected = max(1.2, lines[index].total_characters * typical_rate)
        plausible = [sample for sample in samples if expected * 0.55 <= sample <= expected * 1.65]
        is_outlier = original_duration < expected * 0.55 or original_duration > expected * 1.65
        if is_outlier and plausible:
            lines[index + 1].start = lines[index].start + statistics.median(plausible)


def align_lyrics(
    lyric_lines: list[str],
    words: list[WordSpan],
    audio_duration: float,
) -> tuple[list[TimedLine], dict]:
    """Align canonical lyric lines to Whisper timestamped words."""
    expected_characters: list[str] = []
    expected_phonetics: list[str] = []
    line_ranges: list[tuple[int, int]] = []
    for line in lyric_lines:
        characters, phonetics = phonetic_tokens(line)
        start = len(expected_characters)
        expected_characters.extend(characters)
        expected_phonetics.extend(phonetics)
        line_ranges.append((start, len(expected_characters)))

    reliable_words = [word for word in words if word.probability >= 0.12]
    observed = expand_words(reliable_words or words)
    if not expected_characters:
        raise ValueError("歌词中没有可对齐的文字")
    if not observed:
        raise ValueError("语音模型没有识别到可用文字，请尝试更大的模型或更清晰的人声")

    mappings = _align_indices(expected_phonetics, [char.phonetic for char in observed])
    phonetic_mappings = [mapping for mapping in mappings if mapping[2]]
    character_times = _fill_character_times(
        len(expected_characters), phonetic_mappings, observed, audio_duration
    )
    phonetic_expected = {expected_index for expected_index, _, _ in phonetic_mappings}
    exact_character_expected = {
        expected_index
        for expected_index, observed_index, _ in phonetic_mappings
        if expected_characters[expected_index] == observed[observed_index].char
    }

    provisional: list[TimedLine] = []
    for index, (line, (start_index, end_index)) in enumerate(zip(lyric_lines, line_ranges)):
        if end_index <= start_index:
            continue
        spans = character_times[start_index:end_index]
        start = min(span[0] for span in spans)
        end = max(span[1] for span in spans)
        total = end_index - start_index
        exact = sum(
            1
            for char_index in range(start_index, end_index)
            if char_index in exact_character_expected
        )
        phonetic = sum(
            1
            for char_index in range(start_index, end_index)
            if char_index in phonetic_expected
        )
        phonetic_ratio = phonetic / total
        text_ratio = exact / total
        provisional.append(
            TimedLine(
                index=index + 1,
                text=line,
                start=max(0.0, start),
                end=max(start + 0.15, end),
                confidence=round(phonetic_ratio, 3),
                matched_characters=exact,
                total_characters=total,
                matched_phonetics=phonetic,
                text_confidence=round(text_ratio, 3),
            )
        )

    _repair_repeated_transitions(provisional)

    # Enforce chronological, non-overlapping line ranges for editing software.
    for index in range(1, len(provisional)):
        minimum_start = provisional[index - 1].start + 0.15
        if provisional[index].start < minimum_start:
            provisional[index].start = minimum_start

    for index, line in enumerate(provisional):
        if index + 1 < len(provisional):
            next_start = provisional[index + 1].start
            line.end = max(line.start + 0.12, min(line.end + 0.12, next_start - 0.02))
        else:
            line.end = min(max(audio_duration, line.end), line.end + 0.2)
        line.start = round(line.start, 3)
        line.end = round(max(line.start + 0.12, line.end), 3)

    phonetic_matches = len(phonetic_expected)
    exact_matches = len(exact_character_expected)
    diagnostics = {
        "expected_characters": len(expected_characters),
        "recognized_characters": len(observed),
        "exact_matches": exact_matches,
        "phonetic_matches": phonetic_matches,
        "phonetic_confidence": round(phonetic_matches / len(expected_characters), 3),
        "text_confidence": round(exact_matches / len(expected_characters), 3),
        "overall_confidence": round(phonetic_matches / len(expected_characters), 3),
        "recognized_text": "".join(char.char for char in observed),
    }
    return provisional, diagnostics
