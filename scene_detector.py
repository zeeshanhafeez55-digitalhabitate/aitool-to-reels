"""
Scene detection module - detects scene changes and segments in video.
"""

import cv2
import numpy as np
from scenedetect import detect, ContentDetector, AdaptiveDetector
from typing import List, Tuple
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SceneDetector:
    """Detects scene boundaries in a video file."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.scenes: List[Tuple[float, float]] = []

    def detect_scenes(self, method: str = "content") -> List[Tuple[float, float]]:
        """
        Detect scene changes in the video.

        Args:
            method: 'content' for ContentDetector, 'adaptive' for AdaptiveDetector

        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        logger.info(f"Detecting scenes in: {self.video_path}")

        if method == "content":
            detector = ContentDetector(
                threshold=Config.SCENE_THRESHOLD,
                min_scene_len=int(Config.MIN_SCENE_LENGTH * 30)
            )
        else:
            detector = AdaptiveDetector(
                min_scene_len=int(Config.MIN_SCENE_LENGTH * 30)
            )

        scene_list = detect(self.video_path, detector)

        self.scenes = []
        for scene in scene_list:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            duration = end_time - start_time

            if Config.MIN_SCENE_LENGTH <= duration <= Config.MAX_SCENE_LENGTH:
                self.scenes.append((start_time, end_time))

        logger.info(f"Detected {len(self.scenes)} valid scenes")
        return self.scenes

    def detect_motion_intensity(self) -> List[Tuple[float, float, float]]:
        """
        Analyze motion intensity throughout the video.

        Returns:
            List of (timestamp, motion_score, frame_index) tuples
        """
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        motion_scores = []

        ret, prev_frame = cap.read()
        if not ret:
            return []

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

        frame_idx = 0
        sample_rate = max(1, int(fps / 5))  # Sample 5 frames per second

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % sample_rate != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            frame_diff = cv2.absdiff(prev_gray, gray)
            motion_score = np.mean(frame_diff)
            timestamp = frame_idx / fps

            motion_scores.append((timestamp, motion_score, frame_idx))
            prev_gray = gray

        cap.release()
        logger.info(f"Motion analysis complete: {len(motion_scores)} samples")
        return motion_scores

    def get_high_motion_segments(
        self, threshold_percentile: float = 75
    ) -> List[Tuple[float, float]]:
        """
        Get video segments with above-average motion.

        Returns:
            List of (start_time, end_time) tuples
        """
        motion_scores = self.detect_motion_intensity()
        if not motion_scores:
            return []

        scores = [s[1] for s in motion_scores]
        threshold = np.percentile(scores, threshold_percentile)

        segments = []
        in_segment = False
        start_time = 0

        for timestamp, score, _ in motion_scores:
            if score >= threshold and not in_segment:
                start_time = timestamp
                in_segment = True
            elif score < threshold and in_segment:
                if timestamp - start_time >= Config.MIN_SCENE_LENGTH:
                    segments.append((start_time, timestamp))
                in_segment = False

        logger.info(f"Found {len(segments)} high-motion segments")
        return segments
