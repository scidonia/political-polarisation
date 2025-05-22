"""
Phrasal utilities for text processing.

This module provides functionality to split text into phrases and chunk them
for efficient processing.
"""

import spacy
from typing import List, Tuple, Optional
from pydantic import BaseModel

MAX_TEXT_LEN = 10**6


class Phrase(BaseModel):
    """A phrase with its position in the original text."""

    text: str
    start_char: int
    end_char: int

    def add_to_beginning(self, s: str):
        assert self.start_char >= len(s)
        self.text = s + self.text
        self.start_char -= len(s)

    def add_to_end(self, s: str):
        self.text += s
        self.end_char += len(s)


class PhrasalChunk(BaseModel):
    """A chunk of text composed of multiple phrases."""

    phrases: List[Phrase]

    @property
    def text(self) -> str:
        """Get the full text of this chunk by joining all phrases."""
        return "".join(phrase.text for phrase in self.phrases)

    @property
    def start_char(self) -> int:
        """Get the start character position of the first phrase in this chunk."""
        if not self.phrases:
            return 0
        return self.phrases[0].start_char

    @property
    def end_char(self) -> int:
        """Get the end character position of the last phrase in this chunk."""
        if not self.phrases:
            return 0
        return self.phrases[-1].end_char


def get_phrases(text: str, model: str = "en_core_web_sm") -> List[Phrase]:
    """
    Split text into phrases (sentences with surrounding whitespace).

    Args:
        text: The text to split into phrases
        model: The spaCy model to use for sentence detection

    Returns:
        List of Phrase objects containing (phrase_text, start_char, end_char)
    """
    nlp = spacy.load(model)
    sentences = []
    offset = 0
    original_text = text
    while text != "":
        chunk = text[:MAX_TEXT_LEN]
        remainder = text[MAX_TEXT_LEN:]
        result = nlp(chunk)
        if not result:
            sentences.append(
                Phrase(text=chunk, start_char=offset, end_char=offset + len(chunk))
            )

            if remainder == "":
                break
            else:
                continue

        chunk_sentences = list(result.sents)
        if remainder != "":
            # we're not yet at the end
            if len(chunk_sentences) == 1:
                # only one sentence was found. This happens when a sentence is longer than the max input size. Very strange. let's just consider it a sentence and move on.
                text = remainder
                next_offset = offset + len(chunk)
            else:
                # Otherwise, pop last sentence, and add it to the remainder, cause it might have been sliced in half and that's bad.
                last_sentence = chunk_sentences.pop()
                text = chunk[last_sentence.start_char :] + remainder
                next_offset = offset + last_sentence.start_char
        else:
            text = ""
            next_offset = (
                -1
            )  # this is the last loop. offset will not be used in a next iteration

        sentences += [
            Phrase(
                text=sentence.text,
                start_char=offset + sentence.start_char,
                end_char=offset + sentence.end_char,
            )
            for sentence in chunk_sentences
        ]

        offset = next_offset

    text = original_text
    phrases = []

    # Add text before first sentence
    if sentences and sentences[0].start_char > 0:
        sentences[0].add_to_beginning(text[0 : sentences[0].start_char])

    # Process each sentence and the text between sentences
    for sentence_index in range(0, len(sentences)):
        sentence = sentences[sentence_index]

        # Add the sentence
        phrases.append(sentence)

        # Add text between this sentence and the next (or end of text)
        if sentence_index != len(sentences) - 1:
            next_start = sentences[sentence_index + 1].start_char
            between_text = text[sentence.end_char : next_start]
            if between_text:
                sentence.add_to_end(between_text)
        else:
            # Add text after the last sentence
            end_text = text[sentence.end_char :]
            if end_text:
                sentence.add_to_end(end_text)

    return phrases


def chunk_phrases(phrases: List[Phrase], chunk_size: int) -> List[PhrasalChunk]:
    """
    Group phrases into chunks of approximately chunk_size characters.

    Args:
        phrases: List of Phrase objects
        chunk_size: Target size for each chunk

    Returns:
        List of PhrasalChunk objects
    """
    result = []
    current_chunk_text = ""
    current_chunk_phrases = []

    for phrase in phrases:
        if len(current_chunk_text) + len(phrase.text) > chunk_size:
            if current_chunk_text == "":
                # Weird edge case. sentence is so long it can't even fit a single chunk
                # We need to split up just phrase_text into some chunks
                amount_of_chunks = (len(phrase.text) + chunk_size - 1) // chunk_size
                length_per_chunk = (
                    len(phrase.text) + amount_of_chunks - 1
                ) // amount_of_chunks

                for i in range(0, amount_of_chunks):
                    start_pos = i * length_per_chunk
                    end_pos = min((i + 1) * length_per_chunk, len(phrase.text))
                    chunk_text = phrase.text[start_pos:end_pos]

                    # Create a sub-phrase for this chunk
                    sub_phrase = Phrase(
                        text=chunk_text,
                        start_char=phrase.start_char + start_pos,
                        end_char=phrase.start_char + end_pos,
                    )

                    result.append(PhrasalChunk(phrases=[sub_phrase]))

                # pretend there was no phrase so the code below doesn't get confused
                phrase = Phrase(text="", start_char=0, end_char=0)
            else:
                # We have at least one phrase. yay!
                result.append(PhrasalChunk(phrases=current_chunk_phrases))
                current_chunk_text = ""
                current_chunk_phrases = []

        current_chunk_text += phrase.text
        if phrase.text:  # Only add non-empty phrases
            current_chunk_phrases.append(phrase)

    if current_chunk_phrases:
        result.append(PhrasalChunk(phrases=current_chunk_phrases))

    return result
