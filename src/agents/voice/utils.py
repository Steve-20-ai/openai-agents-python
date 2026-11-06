import re
from collections.abc import Callable

# Latin sentences end in .!? and are separated by whitespace, so the
# whitespace has to be part of the boundary.
_LATIN_TERMINATORS = ".!?"
# Chinese and Japanese end a sentence in 。！？ and write no space after it, so
# requiring whitespace meant a CJK reply never split at all and TTS only
# started once the whole answer had been generated.
_CJK_TERMINATORS = "。！？"
# Closing quotes and brackets belong to the sentence that just ended. A
# boundary placed in front of one would start the next chunk with 」 or ”.
_CLOSING_MARKS = "」』）］】》〉”’"

_SENTENCE_BOUNDARY_RE = re.compile(
    rf"(?<=[{re.escape(_LATIN_TERMINATORS)}])\s+"
    rf"|(?<=[{re.escape(_CJK_TERMINATORS)}])(?![{re.escape(_CLOSING_MARKS)}])"
)


def _needs_separator(sentence: str) -> bool:
    """Whether a space has to follow ``sentence`` when chunks are rejoined.

    A Latin boundary consumes the whitespace it split on, so it has to be put
    back. A CJK boundary is zero-width and there was no separator to begin
    with; inserting one would put a space the model never wrote in front of
    the next Chinese or Japanese sentence.
    """
    return sentence.rstrip(_CLOSING_MARKS)[-1:] not in _CJK_TERMINATORS


def get_sentence_based_splitter(
    min_sentence_length: int = 20,
) -> Callable[[str], tuple[str, str]]:
    """Returns a function that splits text into chunks based on sentence boundaries.

    Args:
        min_sentence_length: The minimum length of a sentence to be included in a chunk.

    Returns:
        A function that splits text into chunks based on sentence boundaries.
    """

    def sentence_based_text_splitter(text_buffer: str) -> tuple[str, str]:
        """
        A function to split the text into chunks. This is useful if you want to split the text into
        chunks before sending it to the TTS model rather than waiting for the whole text to be
        processed.

        Args:
            text_buffer: The text to split.

        Returns:
            A tuple of the text to process and the remaining text buffer.
        """
        sentences = _SENTENCE_BOUNDARY_RE.split(text_buffer.strip())
        if len(sentences) >= 1:
            parts = sentences[:-1]
            combined_sentences = ""
            for index, sentence in enumerate(parts):
                if index and _needs_separator(parts[index - 1]):
                    combined_sentences += " "
                combined_sentences += sentence
            if len(combined_sentences) >= min_sentence_length:
                # Carry any trailing whitespace over to the remainder. It separates the text
                # held back from the next streamed delta, and stripping it concatenates the
                # next word onto the last one, so "He " followed by "arrived" is spoken as
                # "Hearrived".
                trailing_whitespace = text_buffer[len(text_buffer.rstrip()) :]
                remaining_text_buffer = sentences[-1] + trailing_whitespace
                return combined_sentences, remaining_text_buffer
        return "", text_buffer

    return sentence_based_text_splitter
