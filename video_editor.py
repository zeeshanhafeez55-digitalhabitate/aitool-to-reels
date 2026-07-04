"""
Video editing module - handles cropping, effects, subtitles overlay, and export.
"""

import cv2
import numpy as np
from moviepy.editor import (
    VideoFileClip,
    TextClip,
    CompositeVideoClip,
    ColorClip,
    concatenate_videoclips,
    AudioFileClip,
)
from moviepy.video.fx.all import crop, resize, fadein, fadeout
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
from config import Config
from subtitle_generator import SubtitleGenerator
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoEditor:
    """Handles all video editing operations for creating short-form content."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.clip = VideoFileClip(video_path)
        self.subtitle_gen = SubtitleGenerator()

    def __del__(self):
        try:
            self.clip.close()
        except Exception:
            pass

    # ────────────────────────── CROPPING ──────────────────────────

    def smart_crop_to_vertical(
        self, clip: VideoFileClip, target_w: int = 1080, target_h: int = 1920
    ) -> VideoFileClip:
        """
        Intelligently crop horizontal video to vertical (9:16),
        centering on detected faces when possible.
        """
        target_ratio = target_w / target_h  # 0.5625 for 9:16
        clip_w, clip_h = clip.size

        # Calculate crop dimensions
        new_w = int(clip_h * target_ratio)
        new_h = clip_h

        if new_w > clip_w:
            new_w = clip_w
            new_h = int(clip_w / target_ratio)

        # Try face-based centering
        face_center = self._detect_primary_face_position(clip)

        if face_center is not None:
            cx = face_center[0]
            x1 = max(0, cx - new_w // 2)
            x1 = min(x1, clip_w - new_w)
            y1 = max(0, (clip_h - new_h) // 2)
        else:
            x1 = (clip_w - new_w) // 2
            y1 = (clip_h - new_h) // 2

        cropped = crop(
            clip, x1=x1, y1=y1, x2=x1 + new_w, y2=y1 + new_h
        )

        # Resize to exact target dimensions
        cropped = cropped.resize((target_w, target_h))
        return cropped

    def _detect_primary_face_position(
        self, clip: VideoFileClip
    ) -> Optional[Tuple[int, int]]:
        """Detect the primary face position in the first few seconds."""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        sample_times = [
            t for t in np.linspace(0, min(5, clip.duration), 10)
        ]

        face_positions = []
        for t in sample_times:
            try:
                frame = clip.get_frame(t)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, 1.1, 5, minSize=(50, 50)
                )
                for (x, y, w, h) in faces:
                    face_positions.append((x + w // 2, y + h // 2))
            except Exception:
                continue

        if face_positions:
            avg_x = int(np.mean([p[0] for p in face_positions]))
            avg_y = int(np.mean([p[1] for p in face_positions]))
            return (avg_x, avg_y)

        return None

    # ────────────────────────── EFFECTS ──────────────────────────

    def add_zoom_effect(
        self, clip: VideoFileClip, zoom_factor: float = None
    ) -> VideoFileClip:
        """Add a subtle slow zoom (Ken Burns) effect."""
        zoom_factor = zoom_factor or Config.ZOOM_FACTOR

        def zoom_func(get_frame, t):
            frame = get_frame(t)
            h, w = frame.shape[:2]

            progress = t / clip.duration
            current_zoom = 1 + (zoom_factor - 1) * progress

            new_h = int(h / current_zoom)
            new_w = int(w / current_zoom)
            y1 = (h - new_h) // 2
            x1 = (w - new_w) // 2

            cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
            resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            return resized

        return clip.fl(zoom_func)

    def add_fade_effects(
        self, clip: VideoFileClip, duration: float = None
    ) -> VideoFileClip:
        """Add fade in/out effects."""
        duration = duration or Config.FADE_DURATION
        clip = fadein(clip, duration)
        clip = fadeout(clip, duration)
        return clip

    def add_progress_bar(
        self, clip: VideoFileClip, color: Tuple = None, height: int = None
    ) -> VideoFileClip:
        """Add an animated progress bar at the top of the video."""
        color = color or Config.PROGRESS_BAR_COLOR
        height = height or Config.PROGRESS_BAR_HEIGHT
        w, h = clip.size

        def make_progress_frame(t):
            bar = np.zeros((height, w, 3), dtype=np.uint8)
            progress = t / clip.duration
            fill_width = int(w * progress)
            bar[:, :fill_width] = color
            return bar

        progress_clip = (
            ColorClip(size=(w, height), color=(0, 0, 0))
            .set_duration(clip.duration)
            .fl(lambda gf, t: make_progress_frame(t))
        )

        return CompositeVideoClip([
            clip,
            progress_clip.set_position(("center", "top"))
        ])

    # ────────────────────────── SUBTITLES ──────────────────────────

    def add_subtitles(
        self,
        clip: VideoFileClip,
        subtitle_chunks: List[Dict],
        style: str = "modern"
    ) -> VideoFileClip:
        """
        Overlay animated subtitles on the video clip.

        Args:
            clip: VideoFileClip to add subtitles to
            subtitle_chunks: List of subtitle chunk dicts from SubtitleGenerator
            style: Subtitle style ('modern', 'karaoke', 'minimal')
        """
        subtitle_clips = []
        w, h = clip.size

        for chunk in subtitle_chunks:
            if style == "karaoke":
                txt_clip = self._create_karaoke_subtitle(chunk, w, h)
            elif style == "minimal":
                txt_clip = self._create_minimal_subtitle(chunk, w, h)
            else:
                txt_clip = self._create_modern_subtitle(chunk, w, h)

            if txt_clip is not None:
                subtitle_clips.append(txt_clip)

        if subtitle_clips:
            return CompositeVideoClip([clip] + subtitle_clips)
        return clip

    def _create_modern_subtitle(
        self, chunk: Dict, video_w: int, video_h: int
    ) -> Optional[TextClip]:
        """Create a modern-style subtitle with background."""
        try:
            txt = TextClip(
                chunk["text"].upper(),
                fontsize=Config.SUBTITLE_FONTSIZE,
                color=Config.SUBTITLE_COLOR,
                font=Config.SUBTITLE_FONT,
                stroke_color=Config.SUBTITLE_STROKE_COLOR,
                stroke_width=Config.SUBTITLE_STROKE_WIDTH,
                method="caption",
                size=(video_w - 100, None),
                align="center",
            )

            txt = (
                txt
                .set_start(chunk["start"])
                .set_duration(chunk["end"] - chunk["start"])
                .set_position(("center", video_h - 350))
            )
            return txt
        except Exception as e:
            logger.warning(f"Failed to create subtitle: {e}")
            return None

    def _create_karaoke_subtitle(
        self, chunk: Dict, video_w: int, video_h: int
    ) -> Optional[CompositeVideoClip]:
        """Create karaoke-style word-by-word highlight subtitle."""
        try:
            word_clips = []
            x_offset = 50

            for word_info in chunk.get("words", []):
                # Highlighted word
                highlighted = TextClip(
                    word_info["word"].upper(),
                    fontsize=Config.SUBTITLE_FONTSIZE,
                    color="yellow",
                    font=Config.SUBTITLE_FONT,
                    stroke_color="black",
                    stroke_width=2,
                )

                # Normal word
                normal = TextClip(
                    word_info["word"].upper(),
                    fontsize=Config.SUBTITLE_FONTSIZE,
                    color="white",
                    font=Config.SUBTITLE_FONT,
                    stroke_color="black",
                    stroke_width=2,
                )

                word_clips.append({
                    "normal": normal,
                    "highlighted": highlighted,
                    "start": word_info["start"],
                    "end": word_info["end"],
                })

            # For simplicity, return modern style with the full text
            return self._create_modern_subtitle(chunk, video_w, video_h)
        except Exception as e:
            logger.warning(f"Karaoke subtitle failed: {e}")
            return None

    def _create_minimal_subtitle(
        self, chunk: Dict, video_w: int, video_h: int
    ) -> Optional[TextClip]:
        """Create a minimal clean subtitle."""
        try:
            txt = TextClip(
                chunk["text"],
                fontsize=int(Config.SUBTITLE_FONTSIZE * 0.8),
                color="white",
                font="Arial",
                method="caption",
                size=(video_w - 150, None),
                align="center",
            )
            txt = (
                txt
                .set_start(chunk["start"])
                .set_duration(chunk["end"] - chunk["start"])
                .set_position(("center", video_h - 300))
            )
            return txt
        except Exception as e:
            logger.warning(f"Minimal subtitle failed: {e}")
            return None

    # ────────────────────────── CLIP CREATION ──────────────────────────

    def create_short_clip(
        self,
        start: float,
        end: float,
        format_preset: str = "youtube_shorts",
        add_subtitles: bool = True,
        subtitle_style: str = "modern",
        add_effects: bool = True,
        output_path: str = None,
    ) -> str:
        """
        Create a single short clip from the source video.

        Args:
            start: Start time in seconds
            end: End time in seconds
            format_preset: Output format preset
            add_subtitles: Whether to add subtitles
            subtitle_style: Style of subtitles
            add_effects: Whether to add visual effects
            output_path: Output file path

        Returns:
            Path to the created clip
        """
        Config.ensure_dirs()
        preset = Config.FORMATS.get(
            format_preset, Config.FORMATS["youtube_shorts"]
        )

        logger.info(
            f"Creating clip: {start:.1f}s - {end:.1f}s "
            f"({format_preset})"
        )

        # Extract subclip
        subclip = self.clip.subclip(start, min(end, self.clip.duration))

        # Smart crop to vertical
        subclip = self.smart_crop_to_vertical(
            subclip, preset["width"], preset["height"]
        )

        # Add effects
        if add_effects:
            if Config.ZOOM_EFFECT:
                subclip = self.add_zoom_effect(subclip)
            subclip = self.add_fade_effects(subclip)
            if Config.ADD_PROGRESS_BAR:
                subclip = self.add_progress_bar(subclip)

        # Add subtitles
        if add_subtitles:
            try:
                # Export temp audio for transcription
                temp_audio = os.path.join(Config.TEMP_DIR, "temp_audio.wav")
                Config.ensure_dirs()
                original_subclip = self.clip.subclip(
                    start, min(end, self.clip.duration)
                )
                original_subclip.audio.write_audiofile(
                    temp_audio, verbose=False, logger=None
                )

                chunks = self.subtitle_gen.get_subtitle_chunks(temp_audio)

                # Adjust timestamps relative to clip
                for chunk in chunks:
                    chunk["start"] = max(0, chunk["start"])
                    chunk["end"] = min(chunk["end"], subclip.duration)

                subclip = self.add_subtitles(
                    subclip, chunks, style=subtitle_style
                )

                # Cleanup
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)
            except Exception as e:
                logger.warning(f"Subtitle generation failed: {e}")

        # Set FPS
        subclip = subclip.set_fps(preset["fps"])

        # Output path
        if output_path is None:
            output_path = os.path.join(
                Config.OUTPUT_DIR,
                f"short_{format_preset}_{int(start)}_{int(end)}.mp4"
            )

        # Export
        logger.info(f"Exporting: {output_path}")
        subclip.write_videofile(
            output_path,
            codec=Config.VIDEO_CODEC,
            audio_codec=Config.AUDIO_CODEC,
            bitrate=Config.VIDEO_BITRATE,
            preset=Config.PRESET,
            verbose=False,
            logger=None,
        )

        logger.info(f"✅ Clip saved: {output_path}")
        return output_path

    def create_custom_clip(
        self,
        start: float,
        end: float,
        width: int,
        height: int,
        output_path: str,
        **kwargs
    ) -> str:
        """Create a clip with custom dimensions."""
        subclip = self.clip.subclip(start, min(end, self.clip.duration))
        subclip = self.smart_crop_to_vertical(subclip, width, height)

        if kwargs.get("effects", True):
            subclip = self.add_fade_effects(subclip)

        subclip = subclip.set_fps(kwargs.get("fps", 30))

        subclip.write_videofile(
            output_path,
            codec=Config.VIDEO_CODEC,
            audio_codec=Config.AUDIO_CODEC,
            bitrate=Config.VIDEO_BITRATE,
            preset=Config.PRESET,
            verbose=False,
            logger=None,
        )

        return output_path
