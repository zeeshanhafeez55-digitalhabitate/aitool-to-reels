"""
Subtitle generation module - generates animated captions using Whisper.
"""

import whisper
import os
import json
from typing import List, Dict, Optional
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubtitleGenerator:
    """Generates word-level subtitles using OpenAI Whisper."""

    def __init__(self, model_size: str = None):
        self.model_size = model_size or Config.WHISPER_MODEL
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")

    def transcribe(
        self, audio_path: str, language: str = None
    ) -> Dict:
        """
        Transcribe audio/video file to text with timestamps.

        Returns:
            Whisper transcription result dict
        """
        self._load_model()
        logger.info(f"Transcribing: {audio_path}")

        options = {
            "task": "transcribe",
            "word_timestamps": True,
            "verbose": False,
        }
        if language:
            options["language"] = language

        result = self.model.transcribe(audio_path, **options)
        logger.info(f"Transcription complete: {len(result.get('segments', []))} segments")
        return result

    def get_word_timestamps(
        self, audio_path: str, language: str = None
    ) -> List[Dict]:
        """
        Get word-level timestamps.

        Returns:
            List of {word, start, end} dicts
        """
        result = self.transcribe(audio_path, language)
        words = []

        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                words.append({
                    "word": word_info["word"].strip(),
                    "start": round(word_info["start"], 3),
                    "end": round(word_info["end"], 3),
                })

        return words

    def get_subtitle_chunks(
        self,
        audio_path: str,
        max_words_per_chunk: int = 5,
        max_duration: float = 3.0,
        language: str = None
    ) -> List[Dict]:
        """
        Group words into subtitle chunks for display.

        Returns:
            List of {text, start, end, words} dicts
        """
        words = self.get_word_timestamps(audio_path, language)
        if not words:
            return []

        chunks = []
        current_chunk = {
            "words": [],
            "start": words[0]["start"],
            "end": words[0]["end"],
            "text": ""
        }

        for word in words:
            would_exceed_words = (
                len(current_chunk["words"]) >= max_words_per_chunk
            )
            would_exceed_duration = (
                word["end"] - current_chunk["start"] > max_duration
            )
            gap_too_large = (
                current_chunk["words"]
                and word["start"] - current_chunk["words"][-1]["end"] > 1.0
            )

            if would_exceed_words or would_exceed_duration or gap_too_large:
                current_chunk["text"] = " ".join(
                    w["word"] for w in current_chunk["words"]
                )
                chunks.append(current_chunk)
                current_chunk = {
                    "words": [],
                    "start": word["start"],
                    "end": word["end"],
                    "text": ""
                }

            current_chunk["words"].append(word)
            current_chunk["end"] = word["end"]

        # Add last chunk
        if current_chunk["words"]:
            current_chunk["text"] = " ".join(
                w["word"] for w in current_chunk["words"]
            )
            chunks.append(current_chunk)

        logger.info(f"Generated {len(chunks)} subtitle chunks")
        return chunks

    def generate_srt(
        self, audio_path: str, output_path: str, language: str = None
    ) -> str:
        """Generate SRT subtitle file."""
        chunks = self.get_subtitle_chunks(audio_path, language=language)

        srt_content = ""
        for i, chunk in enumerate(chunks, 1):
            start_ts = self._seconds_to_srt_time(chunk["start"])
            end_ts = self._seconds_to_srt_time(chunk["end"])
            srt_content += f"{i}\n{start_ts} --> {end_ts}\n{chunk['text']}\n\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info(f"SRT file saved: {output_path}")
        return output_path

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
