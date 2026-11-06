from __future__ import annotations

import pytest

from agents.voice import get_sentence_based_splitter


def _stream(text: str, chunk_size: int, min_sentence_length: int = 20) -> list[str]:
    """Feed text through the splitter the way VoiceStreamedResult._add_text does."""
    split = get_sentence_based_splitter(min_sentence_length)
    buffer = ""
    spoken: list[str] = []
    for index in range(0, len(text), chunk_size):
        buffer += text[index : index + chunk_size]
        chunk, buffer = split(buffer)
        if chunk:
            spoken.append(chunk)
    if buffer.strip():
        spoken.append(buffer.strip())
    return spoken


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 11, 15])
@pytest.mark.parametrize(
    "text",
    [
        "Dr. Smith went to Washington. He arrived at 3 p.m. sharp.",
        "Hello there friend. How are you doing today? I am fine. Goodbye now.",
        "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten.",
        "A short one. Then a much longer sentence that exceeds the minimum easily.",
    ],
    ids=["abbreviations", "questions", "many_short", "mixed_lengths"],
)
def test_streamed_text_is_spoken_without_losing_or_gluing_words(text: str, chunk_size: int) -> None:
    """Splitting must not depend on where the model's deltas happen to break.

    The buffer was stripped before splitting, which also removed the trailing space that
    separates the held-back text from the next delta. When a delta boundary landed just
    after a space, the following word was concatenated onto the previous one, so "He "
    plus "arrived" was spoken as "Hearrived".
    """
    assert " ".join(_stream(text, chunk_size)).split() == text.split()


def test_split_preserves_the_separator_before_the_next_delta() -> None:
    """The remainder must still end with the whitespace it was given."""
    split = get_sentence_based_splitter(20)

    spoken, remaining = split("This sentence is long enough to flush. He ")

    assert spoken == "This sentence is long enough to flush."
    # Without the trailing space, appending the next delta glues the words together.
    assert remaining == "He "
    assert (remaining + "arrived").split() == ["He", "arrived"]


def test_split_leaves_the_buffer_untouched_when_nothing_is_flushed() -> None:
    split = get_sentence_based_splitter(20)

    assert split("Too short. ") == ("", "Too short. ")
    assert split("   ") == ("", "   ")
    assert split("") == ("", "")


def test_split_without_trailing_whitespace_is_unchanged() -> None:
    split = get_sentence_based_splitter(20)

    assert split("This sentence is long enough to flush. He") == (
        "This sentence is long enough to flush.",
        "He",
    )


CJK_TEXTS = [
    "今天天气很好，阳光明媚。明天会下雨，记得带伞。我们",
    "今日はとても良い天気です。明日は雨が降るでしょう。傘を",
]


@pytest.mark.parametrize("chunk_size", [1, 2, 3])
@pytest.mark.parametrize("text", CJK_TEXTS, ids=["zh", "ja"])
def test_streamed_cjk_text_keeps_every_character(text: str, chunk_size: int) -> None:
    """A CJK reply must stream in sentences, and lose nothing doing it.

    中文 and 日本語 end a sentence in 。！？ with no space after it, so the
    whitespace the Latin pattern required never appeared and nothing was ever
    flushed: TTS received the whole reply only once generation had finished.
    """
    spoken = _stream(text, chunk_size)

    assert len(spoken) > 1, "nothing was flushed before the reply finished"
    assert "".join(spoken) == text


def test_split_flushes_a_cjk_sentence_without_a_space() -> None:
    split = get_sentence_based_splitter(20)

    spoken, remaining = split("今天天气很好，阳光明媚。明天会下雨，记得带伞。我们")

    assert spoken == "今天天气很好，阳光明媚。明天会下雨，记得带伞。"
    assert remaining == "我们"
    # No separator was consumed, so none may be inserted.
    assert " " not in spoken


def test_split_recognises_cjk_exclamations_and_questions() -> None:
    split = get_sentence_based_splitter(0)

    assert split("太棒了！真的吗？我们明天继续吧。好的") == (
        "太棒了！真的吗？我们明天继续吧。",
        "好的",
    )


def test_split_keeps_a_closing_bracket_with_its_sentence() -> None:
    """A boundary right after 。 would start the next chunk with 」."""
    split = get_sentence_based_splitter(0)

    spoken, remaining = split("「行く。」と言った。次に進む。まだ")

    assert spoken == "「行く。」と言った。次に進む。"
    assert remaining == "まだ"
    assert not remaining.startswith("」")


def test_split_mixes_latin_and_cjk_separators() -> None:
    split = get_sentence_based_splitter(0)

    spoken, remaining = split("Hello there. 你好世界。Next part")

    # The space consumed by the Latin boundary is restored; the CJK boundary
    # never had one.
    assert spoken == "Hello there. 你好世界。"
    assert remaining == "Next part"


def test_split_respects_min_sentence_length_for_cjk() -> None:
    assert get_sentence_based_splitter(20)("好的。嗯。我们继续") == ("", "好的。嗯。我们继续")
    assert get_sentence_based_splitter(3)("好的。嗯。我们继续") == ("好的。嗯。", "我们继续")
