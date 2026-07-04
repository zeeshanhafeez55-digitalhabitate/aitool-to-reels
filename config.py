"""
Configuration settings for Video to Shorts converter.
"""

import os

class Config:
    # ──────────────── OUTPUT SETTINGS ────────────────
    OUTPUT_DIR = "output"
    TEMP_DIR = "temp"

    # ──────────────── VIDEO FORMAT PRESETS ────────────────
    FORMATS = {
        "youtube_shorts": {
            "width": 1080,
            "height": 1920,
            "max_duration": 60,
            "fps": 30,
            "aspect_ratio": "9:16"
        },
        "instagram_reels": {
            "width": 1080,
            "height": 1920,
            "max_duration": 90,
            "fps": 30,
            "aspect_ratio": "9:16"
        },
        "tiktok": {
            "width": 1080,
            "height": 1920,
            "max_duration": 180,
            "fps": 30,
            "aspect_ratio": "9:16"
        },
        "square": {
            "width": 1080,
            "height": 1080,
            "max_duration": 60,
            "fps": 30,
            "aspect_ratio": "1:1"
        }
    }

    # ──────────────── SCENE DETECTION ────────────────
    SCENE_THRESHOLD = 30.0          # Sensitivity for scene detection
    MIN_SCENE_LENGTH = 2.0          # Minimum scene length in seconds
    MAX_SCENE_LENGTH = 15.0         # Maximum scene length in seconds

    # ──────────────── HIGHLIGHT DETECTION ────────────────
    AUDIO_ENERGY_WEIGHT = 0.4       # Weight for audio energy scoring
    MOTION_WEIGHT = 0.3             # Weight for motion scoring
    FACE_DETECTION_WEIGHT = 0.3     # Weight for face presence scoring
    MIN_HIGHLIGHT_DURATION = 5      # Minimum highlight clip duration
    MAX_HIGHLIGHT_DURATION = 60     # Maximum highlight clip duration

    # ──────────────── SUBTITLE SETTINGS ────────────────
    SUBTITLE_FONT = "Arial-Bold"
    SUBTITLE_FONTSIZE = 45
    SUBTITLE_COLOR = "white"
    SUBTITLE_STROKE_COLOR = "black"
    SUBTITLE_STROKE_WIDTH = 3
    SUBTITLE_POSITION = ("center", "bottom")
    SUBTITLE_BG_COLOR = None        # Set to "black" for background box
    SUBTITLE_BG_OPACITY = 0.6
    WHISPER_MODEL = "base"          # tiny, base, small, medium, large

    # ──────────────── STYLE SETTINGS ────────────────
    ZOOM_EFFECT = True
    ZOOM_FACTOR = 1.1
    FADE_DURATION = 0.5
    ADD_PROGRESS_BAR = True
    PROGRESS_BAR_COLOR = (255, 0, 100)
    PROGRESS_BAR_HEIGHT = 6

    # ──────────────── ENCODING ────────────────
    VIDEO_CODEC = "libx264"
    AUDIO_CODEC = "aac"
    VIDEO_BITRATE = "8M"
    AUDIO_BITRATE = "192k"
    PRESET = "medium"               # ultrafast, fast, medium, slow

    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
