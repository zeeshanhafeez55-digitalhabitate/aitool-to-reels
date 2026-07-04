"""
Highlight extraction module - identifies the most interesting parts of a video.
"""

import cv2
import numpy as np
import librosa
from typing import List, Tuple, Dict
from config import Config
from scene_detector import SceneDetector
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HighlightExtractor:
    """Extracts the most engaging/interesting segments from a video."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.scene_detector = SceneDetector(video_path)
        self.video_duration = self._get_video_duration()

    def _get_video_duration(self) -> float:
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0

    def analyze_audio_energy(
        self, segment_duration: float = 1.0
    ) -> List[Tuple[float, float]]:
        """
        Analyze audio energy levels throughout the video.

        Returns:
            List of (timestamp, energy_score) tuples
        """
        logger.info("Analyzing audio energy...")
        try:
            y, sr = librosa.load(self.video_path, sr=22050, mono=True)
            hop_length = int(sr * segment_duration)

            energy_scores = []
            for i in range(0, len(y) - hop_length, hop_length):
                segment = y[i:i + hop_length]
                rms = np.sqrt(np.mean(segment ** 2))
                timestamp = i / sr
                energy_scores.append((timestamp, float(rms)))

            # Normalize scores
            if energy_scores:
                max_energy = max(s[1] for s in energy_scores)
                if max_energy > 0:
                    energy_scores = [
                        (t, e / max_energy) for t, e in energy_scores
                    ]

            return energy_scores
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            return []

    def detect_faces(
        self, sample_rate: int = 30
    ) -> List[Tuple[float, int, List]]:
        """
        Detect faces throughout the video.

        Returns:
            List of (timestamp, face_count, face_locations) tuples
        """
        logger.info("Detecting faces...")
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        face_data = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % sample_rate != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            timestamp = frame_idx / fps
            face_locations = [
                (int(x), int(y), int(w), int(h)) for x, y, w, h in faces
            ]
            face_data.append((timestamp, len(faces), face_locations))

        cap.release()
        logger.info(f"Face detection complete: {len(face_data)} samples")
        return face_data

    def compute_segment_scores(
        self, num_segments: int = 5, segment_duration: float = 30.0
    ) -> List[Dict]:
        """
        Score video segments based on multiple criteria.

        Returns:
            List of segment dicts sorted by score (highest first)
        """
        logger.info(f"Computing segment scores (target: {num_segments} segments)...")

        # Get analysis data
        audio_energy = self.analyze_audio_energy()
        motion_data = self.scene_detector.detect_motion_intensity()
        face_data = self.detect_faces()

        # Create time windows
        step = max(1, self.video_duration / (num_segments * 3))
        segments = []

        for start in np.arange(0, self.video_duration - segment_duration, step):
            end = start + segment_duration

            # Audio score
            audio_scores = [
                e for t, e in audio_energy if start <= t <= end
            ]
            audio_score = np.mean(audio_scores) if audio_scores else 0

            # Motion score
            motion_scores_in_range = [
                m for t, m, _ in motion_data if start <= t <= end
            ]
            motion_max = max(
                [m for _, m, _ in motion_data], default=1
            )
            motion_score = (
                np.mean(motion_scores_in_range) / motion_max
                if motion_scores_in_range and motion_max > 0
                else 0
            )

            # Face score
            face_counts = [
                f for t, f, _ in face_data if start <= t <= end
            ]
            face_score = (
                np.mean(face_counts) / max(max(face_counts, default=1), 1)
                if face_counts
                else 0
            )

            # Combined score
            combined_score = (
                Config.AUDIO_ENERGY_WEIGHT * audio_score
                + Config.MOTION_WEIGHT * motion_score
                + Config.FACE_DETECTION_WEIGHT * face_score
            )

            segments.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(segment_duration, 2),
                "score": round(combined_score, 4),
                "audio_score": round(audio_score, 4),
                "motion_score": round(motion_score, 4),
                "face_score": round(face_score, 4),
            })

        # Sort by score and remove overlapping segments
        segments.sort(key=lambda x: x["score"], reverse=True)
        selected = self._remove_overlaps(segments, num_segments)

        logger.info(f"Selected {len(selected)} highlight segments")
        return selected

    def _remove_overlaps(
        self, segments: List[Dict], max_count: int
    ) -> List[Dict]:
        """Remove overlapping segments, keeping highest-scored ones."""
        selected = []
        for seg in segments:
            if len(selected) >= max_count:
                break

            overlaps = False
            for existing in selected:
                if (
                    seg["start"] < existing["end"]
                    and seg["end"] > existing["start"]
                ):
                    overlaps = True
                    break

            if not overlaps:
                selected.append(seg)

        # Sort by start time
        selected.sort(key=lambda x: x["start"])
        return selected

    def extract_highlights(
        self,
        num_clips: int = 5,
        clip_duration: float = 30.0,
        format_preset: str = "youtube_shorts"
    ) -> List[Dict]:
        """
        Main method: Extract highlight clips from the video.

        Args:
            num_clips: Number of clips to extract
            clip_duration: Target duration for each clip
            format_preset: Output format preset name

        Returns:
            List of highlight segment dictionaries
        """
        preset = Config.FORMATS.get(format_preset, Config.FORMATS["youtube_shorts"])
        max_duration = min(clip_duration, preset["max_duration"])

        highlights = self.compute_segment_scores(
            num_segments=num_clips,
            segment_duration=max_duration
        )

        for i, h in enumerate(highlights):
            h["clip_index"] = i
            h["format"] = format_preset
            h["target_width"] = preset["width"]
            h["target_height"] = preset["height"]
            h["target_fps"] = preset["fps"]

        return highlights
