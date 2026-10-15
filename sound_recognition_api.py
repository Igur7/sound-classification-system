from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from analyze_sound import (
    Clock,
    FingerprintLibrary,
    FingerprintMatcher,
    FourierFeatureExtractor,
    RecognitionFusion,
    RecognitionSessionFusion,
    SoundIdentificationResult,
    identify_sound_from_wav_paths,
)


@dataclass(frozen=True)
class SessionRecognitionReport:
    result: SoundIdentificationResult
    per_window_results: list[tuple[Path, SoundIdentificationResult]]

    @property
    def votes(self) -> Counter:
        return Counter(result.label for _, result in self.per_window_results)

    def correct_count(self, expected_label: str) -> int:
        return self.votes[expected_label]

    @property
    def unknown_count(self) -> int:
        return self.votes["unknown"]

    def wrong_known_count(self, expected_label: str) -> int:
        return sum(
            count
            for label, count in self.votes.items()
            if label not in {expected_label, "unknown"}
        )


class MissionSoundRecognizer:
    """Final mission-facing recognizer for top-mic WAV windows.

    This keeps the current Fourier/fingerprint recognition pipeline, then adds
    session-level fusion over consecutive top microphone windows.
    """

    def __init__(
        self,
        reference_dir: Path = Path("sounds_wav"),
        window_ms: int = 5000,
        min_relative_score_margin: float = 0.20,
    ) -> None:
        self.reference_dir = reference_dir
        self.window_ms = window_ms
        self.clock = Clock()
        self.extractor = FourierFeatureExtractor()
        self.matcher = FingerprintMatcher(
            minRelativeScoreMargin=min_relative_score_margin,
        )
        self.window_fusion = RecognitionFusion()
        self.session_fusion = RecognitionSessionFusion()
        self.library = FingerprintLibrary(self.extractor)
        self.library.loadTemplates(
            self.reference_dir,
            templateWindowMs=self.window_ms,
            templateHopMs=self.window_ms,
        )

    def identify_window(self, wav_path: Path) -> SoundIdentificationResult:
        return identify_sound_from_wav_paths(
            sourcePaths=[wav_path],
            windowMs=self.window_ms,
            fingerprintLibrary=self.library,
            clock=self.clock,
            fourierExtractor=self.extractor,
            fingerprintMatcher=self.matcher,
            recognitionFusion=self.window_fusion,
        )

    def identify_session(self, wav_paths: list[Path]) -> SessionRecognitionReport:
        if not wav_paths:
            raise ValueError("At least one WAV path is required.")

        per_window_results = [
            (wav_path, self.identify_window(wav_path))
            for wav_path in wav_paths
        ]
        result = self.session_fusion.fuse(
            [result for _, result in per_window_results]
        )
        return SessionRecognitionReport(
            result=result,
            per_window_results=per_window_results,
        )
