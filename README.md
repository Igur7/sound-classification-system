# RPi sound recognition bundle

Ten folder zawiera pliki potrzebne na Raspberry Pi do finalnego rozpoznawania dzwieku bez ML.

## Zawartosc

```text
rpi_sound_recognition/
  analyze_sound.py                  rdzen Fourier/fingerprint
  sound_recognition_api.py          mission-level API z glosowaniem sesji
  serial_top_session_capture.py     odbior danych z ESP, top mic only
  recognize_session.py              CLI do rozpoznania zapisanej sesji
  requirements.txt                  zaleznosci Pythona
  sounds_wav/*.wav                  biblioteka referencyjna fingerprintow
```

System uzywa tylko mikrofonu `top`.

## Instalacja

```bash
python -m pip install -r requirements.txt
```

## Zbieranie danych z ESP

ESP wysyla pojedynczy strumien PCM16 z mikrofonu `top`.

Przyklad nagrania 60 sekund:

```bash
python serial_top_session_capture.py \
  --port /dev/ttyACM0 \
  --baud 4000000 \
  --sample-rate 40100 \
  --output-dir sessions \
  --label woda \
  --duration-seconds 60
```

Zapisuje:

```text
sessions/woda/top/capture_1.wav
sessions/woda/top/capture_2.wav
...
```

Okna maja domyslnie:

```text
window = 5 s
overlap = 1 s
```

## Rozpoznawanie sesji

```bash
python recognize_session.py --session sessions/woda
```

Wynik:

```text
FINAL label=<klasa> confidence=<confidence> votes={<glosy>} windows=<liczba_okien>
```

## Uzycie jako API

```python
from pathlib import Path
from sound_recognition_api import MissionSoundRecognizer

recognizer = MissionSoundRecognizer()
paths = sorted(Path("sessions/woda/top").glob("capture_*.wav"))
report = recognizer.identify_session(paths)

print(report.result.label)
print(report.result.confidence)
print(report.votes)
```

## Klasy

Etykiety zwracane przez obecny system:

```text
cieknaca_woda
pozar_lasu
armatohaubica_krab
bomba_lotnicza
kolumna_pancerna
bron_maszynowa
unknown
```
