# Audio Monitoring Module

## Overview

The audio monitoring module adds voice and sound detection capabilities to the proctoring system. It detects multiple voices, background noise, and suspicious audio patterns during exams.

---

## Features

### 1. **Speech Detection**
- Detects when the exam-taker is speaking
- Triggers violation after 30 seconds of continuous speech
- Uses RMS (Root Mean Square) analysis

### 2. **Multiple Voice Detection**
- Identifies when multiple people are talking
- Uses frequency spectrum analysis to detect different voice patterns
- **CRITICAL violation** - triggers after 15 seconds
- Indicates unauthorized assistance

### 3. **Background Noise Detection**
- Detects TV, music, or other background sounds
- Analyzes energy distribution across frequency spectrum
- Triggers violation after 20 seconds of continuous noise

### 4. **Audio Evidence Recording**
- Automatically records audio when violations occur
- Saves WAV files for review
- Maximum 50 seconds per recording

---

## How It Works

### Audio Analysis Techniques

#### 1. **RMS (Root Mean Square) Analysis**
Measures the overall loudness of audio:
- **Silence**: RMS < 500
- **Speech**: RMS > 1500
- **Multiple Voices**: RMS > 3000
- **Background Noise**: RMS > 2500

#### 2. **Frequency Spectrum Analysis (FFT)**
Analyzes voice frequencies:
- **Low Frequency** (85-430 Hz): Fundamental voice frequencies
- **Mid Frequency** (430-1290 Hz): Voice harmonics
- **High Frequency** (1290-2580 Hz): Consonants and sibilants

Multiple voices create more frequency peaks across all ranges.

#### 3. **Peak Detection**
Counts significant peaks in frequency spectrum:
- Single voice: 5-10 peaks
- Multiple voices: 15+ peaks
- Background noise: Uniform distribution

---

## Installation

### Install PyAudio

#### Windows:
```bash
pip install pyaudio
```

If you get an error, download the wheel file:
```bash
pip install pipwin
pipwin install pyaudio
```

#### Linux:
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

#### macOS:
```bash
brew install portaudio
pip install pyaudio
```

---

## Usage

### Standalone Audio Monitoring

Test audio monitoring independently:

```bash
python audio_monitoring.py
```

**Output:**
```
Status: SILENT          | RMS:   234 | Avg:   245 | Speech:   0 | Multi:   0 | Noise:   0
Status: SPEECH          | RMS:  1823 | Avg:  1654 | Speech:  12 | Multi:   0 | Noise:   0
Status: MULTIPLE VOICES | RMS:  3421 | Avg:  3102 | Speech:   0 | Multi:   8 | Noise:   0
```

Press `Ctrl+C` to stop and generate report.

---

### Integrated System (Video + Audio)

Run complete proctoring with both video and audio:

```bash
python integrated_proctoring_system.py
```

This combines:
- Face detection
- Eye tracking
- Pupil detection
- Gaze monitoring
- Object detection (YOLO)
- **Audio monitoring** (new!)

---

## Violation Types

### Critical Violations (Auto-terminate at 10)
| Violation | Threshold | Description |
|-----------|-----------|-------------|
| **MULTIPLE_VOICES** | 15 seconds | Multiple people detected speaking |

### Medium Violations (Warning only)
| Violation | Threshold | Description |
|-----------|-----------|-------------|
| **SPEECH_DETECTED** | 30 seconds | Prolonged speech detected |
| **BACKGROUND_NOISE** | 20 seconds | Continuous background noise |

---

## Configuration

### Adjust Thresholds

Edit `audio_monitoring.py`:

```python
# RMS Thresholds
self.SILENCE_THRESHOLD = 500      # Below this = silence
self.SPEECH_THRESHOLD = 1500      # Above this = speech
self.MULTIPLE_VOICE_THRESHOLD = 3000  # Above this = multiple voices
self.NOISE_THRESHOLD = 2500       # Background noise level

# Violation Thresholds (in chunks, ~1 chunk per second)
self.SPEECH_VIOLATION_THRESHOLD = 30          # 30 seconds
self.MULTIPLE_VOICE_VIOLATION_THRESHOLD = 15  # 15 seconds
self.NOISE_VIOLATION_THRESHOLD = 20           # 20 seconds
```

### Cooldown Period

```python
self.violation_cooldown = 5.0  # 5 seconds between violations
```

---

## Output Files

### Audio Violations Folder
```
audio_violations/
├── session_20251207_143022_20251207_143045.wav  # Violation recording
├── session_20251207_143022_20251207_143112.wav
└── audio_report_20251207_143022.txt             # Session report
```

### Sample Report
```
================================================================================
AUDIO MONITORING REPORT
================================================================================

Session ID: 20251207_143022
Start Time: 2025-12-07 14:30:22
End Time: 2025-12-07 14:35:18
Duration: 0:04:56

Total Audio Violations: 3

Violation Summary:
  - MULTIPLE_VOICES: 2
  - SPEECH_DETECTED: 1

================================================================================
DETAILED VIOLATION LOG
================================================================================

1. MULTIPLE_VOICES [CRITICAL]
   Time: 14:31:45
   Details: Multiple people detected speaking (RMS: 3421)

2. SPEECH_DETECTED [MEDIUM]
   Time: 14:33:12
   Details: Prolonged speech detected (RMS: 1823)

3. MULTIPLE_VOICES [CRITICAL]
   Time: 14:34:56
   Details: Multiple people detected speaking (RMS: 3654)
```

---

## Technical Details

### Audio Settings
- **Sample Rate**: 44100 Hz (CD quality)
- **Channels**: 1 (Mono)
- **Format**: 16-bit PCM
- **Chunk Size**: 1024 samples
- **Processing Interval**: ~0.1 seconds

### Performance
- **CPU Usage**: Low (~2-5%)
- **Memory**: Minimal (circular buffers)
- **Latency**: <100ms
- **Runs in Background**: Separate thread

---

## Integration with Video System

The integrated system displays audio status in real-time:

### Audio Panel (Top Right)
```
┌─────────────────────────┐
│   AUDIO MONITOR         │
├─────────────────────────┤
│ Status: MULTIPLE VOICES │
│ Level: 3421             │
│ Multi-Voice: 8/15       │
│ Speech: 0/30            │
│ Noise: 0/20             │
└─────────────────────────┘
```

### Color Coding
- **Green**: Silent/Normal
- **Orange**: Speech or Noise detected
- **Red**: Multiple voices (critical)

---

## Troubleshooting

### "No Default Input Device"
**Problem**: No microphone detected

**Solution**:
1. Check microphone is connected
2. Set default microphone in system settings
3. Grant microphone permissions to Python

### "Input Overflowed"
**Problem**: Audio buffer overflow

**Solution**: Already handled with `exception_on_overflow=False`

### High False Positives
**Problem**: Too many speech violations

**Solution**: Increase thresholds:
```python
self.SPEECH_THRESHOLD = 2000  # Increase from 1500
self.SPEECH_VIOLATION_THRESHOLD = 45  # Increase from 30
```

### Not Detecting Multiple Voices
**Problem**: Multiple people talking but not detected

**Solution**: Decrease threshold:
```python
self.MULTIPLE_VOICE_THRESHOLD = 2500  # Decrease from 3000
```

---

## Advantages

1. **Real-time Detection**: Processes audio every 0.1 seconds
2. **Low Resource Usage**: Runs in background thread
3. **Evidence Recording**: Saves audio clips of violations
4. **Smart Cooldowns**: Prevents spam violations
5. **Frequency Analysis**: Accurate multi-voice detection
6. **Configurable**: Easy to adjust thresholds
7. **Standalone or Integrated**: Works independently or with video

---

## Limitations

1. **Microphone Quality**: Better microphones = better detection
2. **Background Noise**: Very noisy environments may cause false positives
3. **Voice Similarity**: Similar-pitched voices harder to distinguish
4. **Distance**: Voices far from microphone may not be detected
5. **Language Independent**: Works with any language

---

## Future Enhancements

1. **Voice Recognition**: Identify specific speakers
2. **Keyword Detection**: Flag specific words/phrases
3. **Emotion Analysis**: Detect stress or anxiety
4. **Direction Detection**: Determine where sound is coming from
5. **Machine Learning**: Train on exam audio patterns
6. **Real-time Transcription**: Convert speech to text

---

## Testing

### Test with Different Scenarios

1. **Silent**: No talking - should show "SILENT"
2. **Single Voice**: Talk alone - should show "SPEECH" after 30s
3. **Multiple Voices**: Have conversation - should trigger "MULTIPLE_VOICES" after 15s
4. **Background Noise**: Play music/TV - should show "BACKGROUND NOISE" after 20s

---

## API Reference

### AudioMonitor Class

#### Methods

**`__init__()`**
- Initializes audio monitoring system
- Sets up PyAudio stream
- Creates violation tracking

**`start_monitoring()`**
- Opens audio stream
- Returns: `True` if successful

**`stop_monitoring()`**
- Closes audio stream
- Cleans up resources

**`process_audio()`**
- Processes one audio chunk
- Returns: Dictionary with status and metrics

**`log_violation(vtype, severity, details)`**
- Logs audio violation
- Starts recording evidence

**`generate_report()`**
- Creates text report
- Returns: Report file path

---

## License

Same as main project (MIT License)

---

## Support

For issues or questions about audio monitoring:
1. Check microphone permissions
2. Verify PyAudio installation
3. Test with standalone mode first
4. Adjust thresholds for your environment
