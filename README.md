---
title: Coqui Xtts Demo
emoji: 🐢
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.15.0
app_file: gradio_app.py
pinned: false
license: mpl-2.0
short_description: Coqui-XTTS Text-to-Speech Demo with Vietnamese
---

# Coqui XTTS Demo

A multilingual Text-to-Speech demo application supporting 17 languages with an OpenAI-compatible API server.

## Features

- **Gradio Web UI** - Interactive web interface with three modes: Built-in Voice, Reference Voice, and Clone Your Voice
- **OpenAI-Compatible API** - REST API server with multi-speaker synthesis support
- **17 Languages** - English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Hungarian, Korean, Japanese, Vietnamese
- **Multi-Speaker Synthesis** - 115+ speakers (101 built-in + 14 custom)
- **Vietnamese Support** - Text normalization, number-to-words conversion, abbreviations

---

## Quick Start

### Gradio Web Interface

```bash
python gradio_app.py
```

Access at `http://localhost:7860`

### API Server

```bash
python xtts_oai_server/xtts_server.py
```

Server runs at `http://localhost:8088`

---

## API Reference

### POST /v1/audio/speech

Generate speech from text with single or multiple speakers.

#### Single-Speaker Mode (Backward Compatible)

```bash
curl -X POST 'http://localhost:8088/v1/audio/speech' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello world", "speaker": "Aaron Dreschner"}' \
  --output output.wav
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize (max 3000 chars) |
| `speaker` | string | Yes | Speaker ID |
| `language` | string | No | Language code (auto-detected if omitted) |

#### Multi-Speaker Mode with Tags

```bash
curl -X POST 'http://localhost:8088/v1/audio/speech' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "[storyteller_1] Once upon a time... [silence 1.5s] [young_man_1] Hello there!"
  }' \
  --output multi.wav
```

**Tag Syntax:**

| Tag | Example | Description |
|-----|---------|-------------|
| `[speaker_id]` | `[storyteller_1]` | Switch to specified speaker |
| `[silence Xs]` | `[silence 2s]` | Insert X seconds of silence |

**Note:** 1 second of silence is automatically inserted between different speakers.

---

### GET /v1/speakers

List all available speakers.

```bash
curl http://localhost:8088/v1/speakers | python -m json.tool
```

**Response:**

```json
{
  "speakers": [
    {
      "id": "storyteller_1",
      "source": "custom",
      "cached": true
    },
    {
      "id": "Aaron Dreschner",
      "source": "builtin",
      "cached": true
    }
  ],
  "total": 115,
  "counts": {
    "total": 115,
    "builtin": 101,
    "custom": 14
  }
}
```

**Speaker Sources:**
- `source: "builtin"` - Pre-trained speakers from XTTS model
- `source: "custom"` - Custom speakers from `./speakers/` directory

---

## Available Speakers

### Custom Speakers (14)

| ID | Description |
|----|-------------|
| `storyteller_1` | Main narrator voice (most used) |
| `storyteller_2` | Secondary narrator voice |
| `young_man_1` - `young_man_6` | Young male voices |
| `old_man_1` - `old_man_4` | Older male voices |
| `old_woman_1` | Older female voice |
| `young_woman_1` | Young female voice |

### Built-in Speakers (101)

Full list available via `GET /v1/speakers`. Popular voices include:
- `Aaron Dreschner`
- `Anna Jensen`
- `Andrew Koehn`
- `Beatrice Mac`
- `Carson Zuniga`
- And 96 more...

---

## Adding Custom Speakers

1. Place audio files in `./speakers/` directory
2. Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`
3. Filename becomes speaker ID (e.g., `hero_voice.wav` → `hero_voice`)
4. Restart server to process new speakers

**Audio Requirements:**
- Duration: 3-30 seconds
- Quality: Clear speech, minimal background noise
- Format: Any (automatically resampled)

**First startup:** ~2-5 seconds per speaker to process
**Subsequent startups:** <1 second (loads from cache)

---

## Python Client Example

```python
import requests

# API endpoint
url = "http://localhost:8088/v1/audio/speech"

# Get available speakers
speakers = requests.get("http://localhost:8088/v1/speakers").json()
custom = [s for s in speakers['speakers'] if s['source'] == 'custom']
print(f"Custom speakers: {[s['id'] for s in custom]}")

# Single-speaker synthesis
response = requests.post(url, json={
    "text": "Hello world, this is a test.",
    "speaker": "storyteller_1"
})
with open("output.wav", "wb") as f:
    f.write(response.content)

# Multi-speaker synthesis
response = requests.post(url, json={
    "text": "[storyteller_1] Chapter One. [silence 1.5s] [young_man_1] Hello there!"
})
with open("multi.wav", "wb") as f:
    f.write(response.content)
```

---

## Language Support

| Code | Language |
|------|----------|
| `en` | English |
| `es` | Spanish |
| `fr` | French |
| `de` | German |
| `it` | Italian |
| `pt` | Portuguese |
| `pl` | Polish |
| `tr` | Turkish |
| `ru` | Russian |
| `nl` | Dutch |
| `cs` | Czech |
| `ar` | Arabic |
| `zh` | Chinese |
| `hu` | Hungarian |
| `ko` | Korean |
| `ja` | Japanese |
| `vi` | Vietnamese |

Set `language: "Auto"` for automatic language detection.

---

## Generation Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `temperature` | 0.1 - 1.0 | 0.2 | Controls randomness |
| `top_p` | 0.5 - 1.0 | 0.85 | Nucleus sampling threshold |
| `top_k` | 0 - 100 | 70 | Top-k sampling |
| `repetition_penalty` | 1.0 - 50.0 | 9.0 | Prevents repetition |

---

## Error Handling

| Error | Solution |
|-------|----------|
| `Speaker not found` | Check speaker ID via `GET /v1/speakers` |
| `Invalid speaker` | Ensure speaker is available |
| `Missing text` | Provide text in request body |

---

## Directory Structure

```
.
├── gradio_app.py              # Gradio web UI
├── xtts_oai_server/
│   ├── xtts_server.py         # API server
│   ├── speaker_registry.py    # Unified speaker registry
│   ├── custom_speaker_manager.py  # Custom speaker loading
│   └── multi_speaker_inference.py # Multi-speaker synthesis
├── speakers/                  # Custom speaker audio files
├── cache/                     # Model & speaker cache
└── utils/                     # Vietnamese normalization, etc.
```

---

## Documentation

- [Multi-Speaker API Guide](MULTI_SPEAKER_GUIDE.md) - Detailed multi-speaker documentation
- [CLAUDE.md](CLAUDE.md) - Development notes and architecture

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
