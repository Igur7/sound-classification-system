from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

SOUND_NAMES = {
    "01": "cieknaca_woda",
    "02": "pozar_lasu",
    "03": "armatohaubica_krab",
    "04": "bomba_lotnicza",
    "05": "kolumna_pancerna",
    "06": "bron_maszynowa",
}

TARGET_SR = 16000
FRAME_SIZE = 2048
HOP_SIZE = 512
EPS = 1e-8
MAX_PEAKS = 600
DEFAULT_TEMPLATE_WINDOW_MS = 5000
DEFAULT_TEMPLATE_HOP_MS = 5000

def list_sound_files(sounds_dir: Path) -> list[Path]:
    sound_files = sorted(sounds_dir.glob("*.wav"))

    if not sound_files:
        raise FileNotFoundError(
            f"Nie znaleziono plikow WAV w katalogu: {sounds_dir}"
        )

    return sound_files

def get_sound_name(file_path: Path) -> str:
    number = file_path.stem.split("_", 1)[0]

    if number not in SOUND_NAMES:
        raise KeyError(f"Nieznany prefiks probki w pliku: {file_path.name}")

    return SOUND_NAMES[number]

def resample_audio(
    samples: np.ndarray,
    sourceSampleRateHz: int,
    targetSampleRateHz: int,
) -> np.ndarray:
    if sourceSampleRateHz == targetSampleRateHz:
        return samples.astype(np.float32)

    gcd = np.gcd(sourceSampleRateHz, targetSampleRateHz)
    up = targetSampleRateHz // gcd
    down = sourceSampleRateHz // gcd
    return signal.resample_poly(samples, up, down).astype(np.float32)


def load_audio(file_path: Path) -> tuple[np.ndarray, int]:
    sample_rate, samples = wavfile.read(file_path)

    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype == np.int32:
        samples = samples.astype(np.float32) / 2147483648.0
    elif np.issubdtype(samples.dtype, np.floating):
        samples = samples.astype(np.float32)
    else:
        raise TypeError(f"Nieobslugiwany typ danych WAV: {samples.dtype}")

    samples = resample_audio(samples, sample_rate, TARGET_SR)
    sample_rate = TARGET_SR

    samples = samples - np.mean(samples)

    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak

    return samples, sample_rate


def slice_audio_windows(
    samples: np.ndarray,
    sampleRateHz: int,
    windowMs: int,
    hopMs: int | None = None,
) -> list[np.ndarray]:
    if windowMs <= 0:
        raise ValueError("windowMs musi byc dodatnie.")

    if hopMs is None:
        hopMs = windowMs

    if hopMs <= 0:
        raise ValueError("hopMs musi byc dodatnie.")

    window_sample_count = int(sampleRateHz * windowMs / 1000)
    hop_sample_count = int(sampleRateHz * hopMs / 1000)

    if window_sample_count <= 0 or hop_sample_count <= 0:
        raise ValueError("Rozmiar okna i krok musza dawac dodatnia liczbe probek.")

    if len(samples) <= window_sample_count:
        padded = np.pad(samples, (0, max(0, window_sample_count - len(samples))))
        return [padded.astype(np.float32)]

    windows = []

    for start in range(0, len(samples) - window_sample_count + 1, hop_sample_count):
        end = start + window_sample_count
        windows.append(samples[start:end].astype(np.float32))

    return windows

@dataclass
class AudioWindow:
    samples: np.ndarray
    sampleRateHz: int
    timestampUs: int
    micId: int

@dataclass
class RecognitionBatch:
    windows: list[AudioWindow]
    timestampUs: int


@dataclass
class SpectrumFrame:
    frequenciesHz: np.ndarray
    magnitudes: np.ndarray


@dataclass
class SpectrogramFrame:
    timeBinsS: np.ndarray
    frequencyBinsHz: np.ndarray
    powerDb: np.ndarray


@dataclass
class Peak:
    timeS: float
    frequencyHz: float
    amplitudeDb: float


@dataclass
class FingerprintTemplate:
    peaks: list[Peak]
    sourcePath: Path | None = None
    windowIndex: int = 0
    timeValuesS: np.ndarray = field(init=False, repr=False)
    frequencyValuesHz: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.timeValuesS = np.asarray(
            [peak.timeS for peak in self.peaks],
            dtype=np.float32,
        )
        self.frequencyValuesHz = np.asarray(
            [peak.frequencyHz for peak in self.peaks],
            dtype=np.float32,
        )


@dataclass
class MicClassificationResult:
    micId: int
    label: str
    confidence: float
    matchScore: float


@dataclass
class SoundIdentificationResult:
    label: str
    confidence: float
    winningMicId: int
    method: str


class IMicDriver(ABC):
    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def readWindow(self, durationMs: int) -> AudioWindow:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass


class Clock:
    def nowUs(self) -> int:
        return time.time_ns() // 1000
    
class RecognitionMicDriver(IMicDriver):
    def __init__(self, micId: int) -> None:
        self.micId = micId

    def initialize(self) -> None:
        pass

    def start(self) -> None:
        pass

    def readWindow(self, durationMs: int) -> AudioWindow:
        pass

    def stop(self) -> None:
        pass


class WavMicDriver(IMicDriver):
    def __init__(self, micId: int, sourcePath: Path, clock: Clock) -> None:
        self.micId = micId
        self.sourcePath = sourcePath
        self.clock = clock
        self.samples = None
        self.sampleRateHz = None
        self.cursor = 0
        self.running = False

    def initialize(self) -> None:
        samples, sample_rate = load_audio(self.sourcePath)
        self.samples = samples
        self.sampleRateHz = sample_rate
        self.cursor = 0

    def start(self) -> None:
        if self.samples is None or self.sampleRateHz is None:
            self.initialize()

        self.cursor = 0
        self.running = True

    def readWindow(self, durationMs: int) -> AudioWindow:
        if not self.running:
            raise RuntimeError(f"Mikrofon {self.micId} nie zostal uruchomiony.")

        if self.samples is None or self.sampleRateHz is None:
            raise RuntimeError(f"Mikrofon {self.micId} nie zostal zainicjalizowany.")

        sample_count = int(self.sampleRateHz * durationMs / 1000)
        start = self.cursor
        end = start + sample_count

        window_samples = self.samples[start:end]

        if len(window_samples) < sample_count:
            window_samples = np.pad(window_samples, (0, sample_count - len(window_samples)))

        self.cursor = end

        return AudioWindow(
            samples=window_samples.astype(np.float32),
            sampleRateHz=self.sampleRateHz,
            timestampUs=self.clock.nowUs(),
            micId=self.micId,
        )

    def stop(self) -> None:
        self.running = False


def initialize_and_start_mics(mics: list[IMicDriver]) -> None:
    if not mics:
        raise ValueError("Lista mikrofonow nie moze byc pusta.")

    for mic in mics:
        mic.initialize()
        mic.start()


def stop_mics(mics: list[IMicDriver]) -> None:
    for mic in mics:
        mic.stop()


def build_wav_mics(
    sourcePaths: list[Path],
    clock: Clock | None = None,
    firstMicId: int = 1,
) -> tuple[list[WavMicDriver], Clock]:
    if not sourcePaths:
        raise ValueError("Trzeba podac przynajmniej jeden plik WAV.")

    if clock is None:
        clock = Clock()

    mics = [
        WavMicDriver(firstMicId + index, source_path, clock)
        for index, source_path in enumerate(sourcePaths)
    ]

    return mics, clock

class RecognitionCoordinator:
    def __init__(self, mics: list[IMicDriver], clock: Clock) -> None:
        self.mics = mics
        self.clock = clock

    def captureBatch(self, windowMs: int) -> RecognitionBatch:
        windows = []

        for mic in self.mics:
            window = mic.readWindow(windowMs)
            windows.append(window)

        return RecognitionBatch(
            windows=windows,
            timestampUs=self.clock.nowUs(),
        )

class FourierFeatureExtractor:
    def computeSpectrum(self, window: AudioWindow) -> SpectrumFrame:
        windowed_samples = window.samples * np.hanning(len(window.samples)) #podobno okno ładnie obicna szumy na końcach 
        #np.fft.rft, mamy tylko jednowymiarową tablice więc imo nie ma co liczyć full
        frequencies = np.fft.rfftfreq(len(windowed_samples), d=1.0 / window.sampleRateHz)
        fft_values = np.fft.rfft(windowed_samples)
        magnitudes = np.abs(fft_values)

        return SpectrumFrame(
            frequenciesHz=frequencies,
            magnitudes=magnitudes,
        )

    def computeSpectrogram(self, window: AudioWindow) -> SpectrogramFrame:
        frequencies, times, stft_matrix = signal.stft(
            window.samples,
            fs=window.sampleRateHz,
            window="hann",
            nperseg=FRAME_SIZE,
            noverlap=FRAME_SIZE - HOP_SIZE,
            boundary=None,
        )

        power = np.abs(stft_matrix) ** 2
        power_db = 10.0 * np.log10(power + EPS)

        return SpectrogramFrame(
            timeBinsS=times,
            frequencyBinsHz=frequencies,
            powerDb=power_db,
        )


    def extractPeaks(self, spec: SpectrogramFrame) -> list[Peak]:
        local_max = ndimage.maximum_filter(spec.powerDb, size=(3, 3))
        peak_mask = spec.powerDb == local_max

        amplitude_threshold = np.percentile(spec.powerDb, 90)
        peak_mask &= spec.powerDb >= amplitude_threshold

        freq_indices, time_indices = np.where(peak_mask)

        peaks = []
        for freq_index, time_index in zip(freq_indices, time_indices):
            peaks.append(
                Peak(
                    timeS=float(spec.timeBinsS[time_index]),
                    frequencyHz=float(spec.frequencyBinsHz[freq_index]),
                    amplitudeDb=float(spec.powerDb[freq_index, time_index]),
                )
            )

        peaks.sort(key=lambda peak: peak.amplitudeDb, reverse=True)
        return peaks[:MAX_PEAKS]


class FingerprintLibrary:
    def __init__(self, extractor: FourierFeatureExtractor) -> None:
        self.extractor = extractor
        self.templates: dict[str, list[FingerprintTemplate]] = {}

    def addTemplate(
        self,
        label: str,
        peaks: list[Peak],
        sourcePath: Path | None = None,
        windowIndex: int = 0,
    ) -> None:
        if label not in self.templates:
            self.templates[label] = []

        self.templates[label].append(
            FingerprintTemplate(
                peaks=peaks,
                sourcePath=sourcePath,
                windowIndex=windowIndex,
            )
        )

    def loadTemplates(
        self,
        sounds_dir: Path,
        templateWindowMs: int = DEFAULT_TEMPLATE_WINDOW_MS,
        templateHopMs: int = DEFAULT_TEMPLATE_HOP_MS,
    ) -> None:
        self.templates = {}
        sound_files = list_sound_files(sounds_dir)

        for sound_file in sound_files:
            label = get_sound_name(sound_file)
            samples, sample_rate = load_audio(sound_file)

            template_windows = slice_audio_windows(
                samples=samples,
                sampleRateHz=sample_rate,
                windowMs=templateWindowMs,
                hopMs=templateHopMs,
            )

            for window_index, window_samples in enumerate(template_windows):
                window = AudioWindow(
                    samples=window_samples,
                    sampleRateHz=sample_rate,
                    timestampUs=0,
                    micId=-1,
                )

                spectrogram = self.extractor.computeSpectrogram(window)
                peaks = self.extractor.extractPeaks(spectrogram)
                self.addTemplate(
                    label=label,
                    peaks=peaks,
                    sourcePath=sound_file,
                    windowIndex=window_index,
                )

    def getLabels(self) -> list[str]:
        return list(self.templates.keys())

class FingerprintMatcher:
    def __init__(
        self,
        frequencyToleranceHz: float = 80.0,
        timeToleranceS: float = 0.08,
        timeOffsetBinS: float = 0.04,
        minRelativeScoreMargin: float = 0.20,
    ) -> None:
        self.frequencyToleranceHz = frequencyToleranceHz
        self.timeToleranceS = timeToleranceS
        self.timeOffsetBinS = timeOffsetBinS
        self.minRelativeScoreMargin = minRelativeScoreMargin
    #najlepsza próbka = ta, która ma najwięcej zgodnych pików z aktualnym sygnałem

    def _buildPeakArrays(
        self,
        peaks: list[Peak],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not peaks:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty

        query_times = np.asarray([peak.timeS for peak in peaks], dtype=np.float32)
        query_frequencies = np.asarray(
            [peak.frequencyHz for peak in peaks],
            dtype=np.float32,
        )

        return query_times, query_frequencies

    def _countMatches(
        self,
        queryTimesS: np.ndarray,
        queryFrequenciesHz: np.ndarray,
        template: FingerprintTemplate,
    ) -> int:
        if queryTimesS.size == 0 or template.timeValuesS.size == 0:
            return 0

        frequency_matches = np.abs(
            queryFrequenciesHz[:, None] - template.frequencyValuesHz[None, :]
        ) <= self.frequencyToleranceHz

        query_indices, template_indices = np.where(frequency_matches)
        if query_indices.size == 0:
            return 0

        time_deltas = (
            template.timeValuesS[template_indices] - queryTimesS[query_indices]
        )
        offset_bins = np.round(time_deltas / self.timeOffsetBinS).astype(np.int32)
        unique_bins, counts = np.unique(offset_bins, return_counts=True)

        best_match_count = 0
        candidate_bins = unique_bins[np.argsort(counts)[-3:]]

        for offset_bin in candidate_bins:
            time_offset_s = float(offset_bin) * self.timeOffsetBinS
            time_matches = np.abs(
                (queryTimesS[:, None] + time_offset_s)
                - template.timeValuesS[None, :]
            ) <= self.timeToleranceS
            match_count = int(
                np.count_nonzero(np.any(frequency_matches & time_matches, axis=1))
            )

            if match_count > best_match_count:
                best_match_count = match_count

        return best_match_count
    
    def match(
        self,
        micId: int,
        peaks: list[Peak],
        library: FingerprintLibrary,
    ) -> MicClassificationResult:
        query_times, query_frequencies = self._buildPeakArrays(peaks)
        best_label = "unknown"
        best_score = 0
        second_best_score = 0

        for label, templates in library.templates.items():
            label_best_score = 0

            for template in templates:
                score = self._countMatches(query_times, query_frequencies, template)

                if score > label_best_score:
                    label_best_score = score

            if label_best_score > best_score:
                second_best_score = best_score
                best_score = label_best_score
                best_label = label
            elif label_best_score > second_best_score:
                second_best_score = label_best_score

        confidence = 0.0
        if peaks:
            confidence = best_score / len(peaks)

        relative_margin = 0.0
        if best_score > 0:
            relative_margin = (best_score - second_best_score) / best_score

        if relative_margin < self.minRelativeScoreMargin:
            best_label = "unknown"

        return MicClassificationResult(
            micId=micId,
            label=best_label,
            confidence=min(confidence, 1.0),
            matchScore=float(best_score),
        )

 #sumujemy confidance z miców w jednej grupie i tam gdzie największa suma to bierzemy
    
class RecognitionFusion:
    def fuse(self, results: list[MicClassificationResult]) -> SoundIdentificationResult:
        if not results:
            return SoundIdentificationResult(
                label="unknown",
                confidence=0.0,
                winningMicId=-1,
                method="empty_results",
            )

        grouped_results: dict[str, list[MicClassificationResult]] = {}

        for result in results:
            if result.label not in grouped_results:
                grouped_results[result.label] = []

            grouped_results[result.label].append(result)

        best_label = "unknown"
        best_group_confidence = -1.0
        best_group_score = -1.0

        for label, group in grouped_results.items():
            total_confidence = sum(item.confidence for item in group)
            total_score = sum(item.matchScore for item in group)

            if total_confidence > best_group_confidence:
                best_label = label
                best_group_confidence = total_confidence
                best_group_score = total_score
            elif total_confidence == best_group_confidence and total_score > best_group_score:
                best_label = label
                best_group_confidence = total_confidence
                best_group_score = total_score

        winning_group = grouped_results[best_label]
        best_mic_result = max(
            winning_group,
            key=lambda item: (item.confidence, item.matchScore),
        )

        final_confidence = best_group_confidence / len(results)

        return SoundIdentificationResult(
            label=best_label,
            confidence=min(final_confidence, 1.0),
            winningMicId=best_mic_result.micId,
            method="sum_confidence_then_score",
        )


class RecognitionSessionFusion:
    def fuse(self, results: list[SoundIdentificationResult]) -> SoundIdentificationResult:
        if not results:
            return SoundIdentificationResult(
                label="unknown",
                confidence=0.0,
                winningMicId=-1,
                method="empty_session",
            )

        known_results = [
            result for result in results
            if result.label != "unknown"
        ]
        voting_results = known_results if known_results else results

        grouped_results: dict[str, list[SoundIdentificationResult]] = {}

        for result in voting_results:
            if result.label not in grouped_results:
                grouped_results[result.label] = []

            grouped_results[result.label].append(result)

        best_label = "unknown"
        best_vote_count = -1
        best_confidence_sum = -1.0

        for label, group in grouped_results.items():
            vote_count = len(group)
            confidence_sum = sum(item.confidence for item in group)

            if vote_count > best_vote_count:
                best_label = label
                best_vote_count = vote_count
                best_confidence_sum = confidence_sum
            elif vote_count == best_vote_count and confidence_sum > best_confidence_sum:
                best_label = label
                best_vote_count = vote_count
                best_confidence_sum = confidence_sum

        winning_group = grouped_results[best_label]
        representative_result = max(
            winning_group,
            key=lambda item: item.confidence,
        )

        vote_ratio = len(winning_group) / len(voting_results)
        average_confidence = sum(item.confidence for item in winning_group) / len(winning_group)
        final_confidence = min(1.0, 0.5 * vote_ratio + 0.5 * average_confidence)

        return SoundIdentificationResult(
            label=best_label,
            confidence=final_confidence,
            winningMicId=representative_result.winningMicId,
            method="majority_vote_then_confidence",
        )

class RecognitionMissionController:
    def __init__(
        self,
        recognitionCoordinator: RecognitionCoordinator,
        fourierExtractor: FourierFeatureExtractor,
        fingerprintMatcher: FingerprintMatcher,
        recognitionFusion: RecognitionFusion,
        fingerprintLibrary: FingerprintLibrary,
    ) -> None:
        self.recognitionCoordinator = recognitionCoordinator
        self.fourierExtractor = fourierExtractor
        self.fingerprintMatcher = fingerprintMatcher
        self.recognitionFusion = recognitionFusion
        self.fingerprintLibrary = fingerprintLibrary

    def identifySound(self, windowMs: int) -> SoundIdentificationResult:
        batch = self.recognitionCoordinator.captureBatch(windowMs)

        mic_results = []

        for window in batch.windows:
            spectrogram = self.fourierExtractor.computeSpectrogram(window)
            peaks = self.fourierExtractor.extractPeaks(spectrogram)

            mic_result = self.fingerprintMatcher.match(
                micId=window.micId,
                peaks=peaks,
                library=self.fingerprintLibrary,
            )
            mic_results.append(mic_result)

        return self.recognitionFusion.fuse(mic_results)


def build_recognition_controller(
    mics: list[IMicDriver],
    fingerprintLibrary: FingerprintLibrary,
    clock: Clock | None = None,
    fourierExtractor: FourierFeatureExtractor | None = None,
    fingerprintMatcher: FingerprintMatcher | None = None,
    recognitionFusion: RecognitionFusion | None = None,
    autoStartMics: bool = True,
) -> RecognitionMissionController:
    if not mics:
        raise ValueError("Trzeba podac przynajmniej jeden mikrofon.")

    if clock is None:
        clock = Clock()

    if autoStartMics:
        initialize_and_start_mics(mics)

    if fourierExtractor is None:
        fourierExtractor = FourierFeatureExtractor()

    if fingerprintMatcher is None:
        fingerprintMatcher = FingerprintMatcher()

    if recognitionFusion is None:
        recognitionFusion = RecognitionFusion()

    return RecognitionMissionController(
        recognitionCoordinator=RecognitionCoordinator(mics, clock),
        fourierExtractor=fourierExtractor,
        fingerprintMatcher=fingerprintMatcher,
        recognitionFusion=recognitionFusion,
        fingerprintLibrary=fingerprintLibrary,
    )


def identify_sound_from_wav_paths(
    sourcePaths: list[Path],
    windowMs: int,
    fingerprintLibrary: FingerprintLibrary,
    clock: Clock | None = None,
    fourierExtractor: FourierFeatureExtractor | None = None,
    fingerprintMatcher: FingerprintMatcher | None = None,
    recognitionFusion: RecognitionFusion | None = None,
) -> SoundIdentificationResult:
    wav_mics, used_clock = build_wav_mics(sourcePaths, clock=clock)

    try:
        controller = build_recognition_controller(
            mics=wav_mics,
            fingerprintLibrary=fingerprintLibrary,
            clock=used_clock,
            fourierExtractor=fourierExtractor,
            fingerprintMatcher=fingerprintMatcher,
            recognitionFusion=recognitionFusion,
            autoStartMics=True,
        )
        return controller.identifySound(windowMs)
    finally:
        stop_mics(wav_mics)
