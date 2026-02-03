"""
Whisper-based speech-to-text for automatic subtitle generation.

Uses OpenAI's Whisper model for high-quality transcription with
word-level timestamps for advanced animation effects.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class WordTiming:
    """Timing information for a single word."""
    word: str
    start: float
    end: float
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass 
class TranscriptionSegment:
    """A segment of transcribed text with timing."""
    text: str
    start: float
    end: float
    words: list[WordTiming] = field(default_factory=list)
    language: str = ""
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
                for w in self.words
            ],
            "language": self.language,
            "confidence": self.confidence,
        }


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str
    segments: list[TranscriptionSegment]
    language: str
    duration: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "duration": self.duration,
        }
    
    def to_srt(self) -> str:
        """Convert to SRT subtitle format."""
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start = _format_srt_time(seg.start)
            end = _format_srt_time(seg.end)
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines)


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class WhisperTranscriber:
    """
    Speech-to-text transcriber using OpenAI Whisper.
    
    Supports both the openai-whisper Python library and
    the faster-whisper implementation for better performance.
    
    Example:
        transcriber = WhisperTranscriber(model="medium")
        result = transcriber.transcribe("video.mp4")
        
        for segment in result.segments:
            print(f"{segment.start:.2f} - {segment.end:.2f}: {segment.text}")
    """
    
    AVAILABLE_MODELS = [
        "tiny", "tiny.en",
        "base", "base.en",
        "small", "small.en",
        "medium", "medium.en",
        "large", "large-v1", "large-v2", "large-v3",
    ]
    
    def __init__(
        self,
        model: str = "medium",
        device: str = "auto",
        compute_type: str = "auto",
        language: Optional[str] = None,
        use_faster_whisper: bool = True,
    ):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (auto, cpu, cuda, mps)
            compute_type: Compute type for faster-whisper (auto, float16, int8)
            language: Force specific language (None for auto-detect)
            use_faster_whisper: Use faster-whisper if available
        """
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.use_faster_whisper = use_faster_whisper
        
        self._model = None
        self._backend = None
    
    def _load_model(self):
        """Lazy-load the Whisper model."""
        if self._model is not None:
            return
        
        # Try faster-whisper first
        if self.use_faster_whisper:
            try:
                from faster_whisper import WhisperModel
                
                device = self.device
                if device == "auto":
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        device = "cpu"  # faster-whisper doesn't support MPS directly
                    else:
                        device = "cpu"
                
                compute_type = self.compute_type
                if compute_type == "auto":
                    compute_type = "float16" if device == "cuda" else "int8"
                
                self._model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=compute_type,
                )
                self._backend = "faster-whisper"
                logger.info(f"Loaded faster-whisper model: {self.model_name} on {device}")
                return
                
            except ImportError:
                logger.info("faster-whisper not available, falling back to openai-whisper")
        
        # Fall back to openai-whisper
        try:
            import whisper
            
            self._model = whisper.load_model(self.model_name)
            self._backend = "openai-whisper"
            logger.info(f"Loaded openai-whisper model: {self.model_name}")
            
        except ImportError:
            raise ImportError(
                "No Whisper implementation found. Install either:\n"
                "  pip install faster-whisper  (recommended)\n"
                "  pip install openai-whisper"
            )
    
    def transcribe(
        self,
        audio_path: Path | str,
        word_timestamps: bool = True,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio/video file.
        
        Args:
            audio_path: Path to audio or video file
            word_timestamps: Include word-level timestamps
            **kwargs: Additional arguments passed to Whisper
            
        Returns:
            TranscriptionResult with segments and word timings
        """
        self._load_model()
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing: {audio_path}")
        
        # Extract audio if needed
        if audio_path.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv', '.webm'):
            audio_path = self._extract_audio(audio_path)
        
        if self._backend == "faster-whisper":
            return self._transcribe_faster_whisper(audio_path, word_timestamps, **kwargs)
        else:
            return self._transcribe_openai_whisper(audio_path, word_timestamps, **kwargs)
    
    def _extract_audio(self, video_path: Path) -> Path:
        """Extract audio from video using FFmpeg."""
        audio_path = Path(tempfile.mktemp(suffix=".wav"))
        
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", "16000",  # 16kHz for Whisper
            "-ac", "1",  # Mono
            "-y",  # Overwrite
            str(audio_path)
        ]
        
        logger.debug(f"Extracting audio: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
        
        return audio_path
    
    def _transcribe_faster_whisper(
        self,
        audio_path: Path,
        word_timestamps: bool,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe using faster-whisper."""
        segments_gen, info = self._model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=word_timestamps,
            **kwargs
        )
        
        segments = []
        full_text = []
        
        for segment in segments_gen:
            words = []
            if word_timestamps and segment.words:
                words = [
                    WordTiming(
                        word=w.word.strip(),
                        start=w.start,
                        end=w.end,
                        confidence=w.probability if hasattr(w, 'probability') else 1.0,
                    )
                    for w in segment.words
                ]
            
            segments.append(TranscriptionSegment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                words=words,
                language=info.language,
            ))
            full_text.append(segment.text.strip())
        
        return TranscriptionResult(
            text=" ".join(full_text),
            segments=segments,
            language=info.language,
            duration=info.duration,
        )
    
    def _transcribe_openai_whisper(
        self,
        audio_path: Path,
        word_timestamps: bool,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe using openai-whisper."""
        result = self._model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=word_timestamps,
            **kwargs
        )
        
        segments = []
        
        for segment in result.get("segments", []):
            words = []
            if word_timestamps and "words" in segment:
                words = [
                    WordTiming(
                        word=w["word"].strip(),
                        start=w["start"],
                        end=w["end"],
                        confidence=w.get("probability", 1.0),
                    )
                    for w in segment["words"]
                ]
            
            segments.append(TranscriptionSegment(
                text=segment["text"].strip(),
                start=segment["start"],
                end=segment["end"],
                words=words,
                language=result.get("language", ""),
            ))
        
        return TranscriptionResult(
            text=result.get("text", "").strip(),
            segments=segments,
            language=result.get("language", ""),
            duration=segments[-1].end if segments else 0,
        )
    
    def transcribe_to_srt(
        self,
        audio_path: Path | str,
        output_path: Optional[Path | str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe and optionally save as SRT file.
        
        Args:
            audio_path: Input audio/video file
            output_path: Optional path to save SRT file
            **kwargs: Additional transcription arguments
            
        Returns:
            SRT formatted string
        """
        result = self.transcribe(audio_path, **kwargs)
        srt_content = result.to_srt()
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(srt_content, encoding="utf-8")
            logger.info(f"Saved SRT to: {output_path}")
        
        return srt_content
