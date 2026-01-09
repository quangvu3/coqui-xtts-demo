# Multi-Speaker TTS API Guide

This guide explains how to use the multi-speaker text-to-speech features added to the XTTS API server.

## Overview

The API server now supports:
- **Custom speakers** from audio files in `./speakers/` folder
- **Single-speaker mode** (backward compatible)
- **Multi-speaker mode** with embedded speaker tags `[speaker_id]`
- **Silence insertion** with `[silence Xs]` tags
- **Speaker discovery** via GET `/v1/speakers` endpoint

## Starting the Server

```bash
python xtts_oai_server/xtts_server.py
```

The server will:
1. Load the XTTS model
2. Scan `./speakers/` for audio files (.wav, .mp3, .flac, .ogg)
3. Process audio files into embeddings (first start only)
4. Cache embeddings as `.safetensors` files in `cache/speakers/custom/`
5. Load 101 built-in speakers + custom speakers
6. Start listening on `http://0.0.0.0:8088`

**First startup**: Takes ~2-5 seconds per speaker to process audio files
**Subsequent startups**: <1 second (loads from cache)

## Current Speaker Count

- **Built-in speakers**: 101
- **Custom speakers**: 22 (from `./speakers/` folder)
- **Total**: 123 speakers

## API Endpoints

### POST /v1/audio/speech (Modified)

Generate speech from text with single or multiple speakers.

#### Single-Speaker Mode (Backward Compatible)

**Request:**
```json
{
  "text": "Hello world, this is a test.",
  "speaker": "main_storyteller_1"
}
```

**Response:** WAV audio file

**Example:**
```bash
curl -X POST 'http://localhost:8088/v1/audio/speech' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello world", "speaker": "main_storyteller_1"}' \
  --output output.wav
```

#### Multi-Speaker Mode with Tags

**Request:**
```json
{
  "text": "[main_storyteller_1] Once upon a time... [silence 1s] [normal_young_man_1] Hello!"
}
```

**Tag Formats:**
- Speaker tags: `[speaker_id]` - Switch to specified speaker
- Silence tags: `[silence 2s]` or `[silence 0.5s]` - Insert silence duration

**Example:**
```bash
cat > request.json <<'EOF'
{
  "text": "[main_storyteller_1] Chapter One. [silence 1.5s] [normal_old_man_1] Who goes there? [silence 0.5s] [normal_young_woman_1] It's me!"
}
EOF

curl -X POST 'http://localhost:8088/v1/audio/speech' \
  -H 'Content-Type: application/json' \
  -d @request.json \
  --output audiobook.wav
```

### GET /v1/speakers (New)

List all available speakers.

**Request:** No parameters

**Response:**
```json
{
  "speakers": [
    {
      "id": "main_storyteller_1",
      "source": "custom",
      "cached": true
    },
    {
      "id": "Aaron Dreschner",
      "source": "builtin",
      "cached": true
    }
  ],
  "total": 123,
  "counts": {
    "total": 123,
    "builtin": 101,
    "custom": 22
  }
}
```

**Example:**
```bash
curl http://localhost:8088/v1/speakers | python -m json.tool
```

**Filter custom speakers:**
```bash
curl -s http://localhost:8088/v1/speakers | \
  python -c "import json, sys; speakers = json.load(sys.stdin)['speakers']; print('\\n'.join(s['id'] for s in speakers if s['source'] == 'custom'))"
```

## Available Custom Speakers

The following custom speakers are currently available:
- `main_storyteller_1`
- `main_storyteller_2`
- `normal_old_man_1` through `normal_old_man_5`
- `normal_old_woman_1` through `normal_old_woman_4`
- `normal_young_man_1` through `normal_young_man_6`
- `normal_young_woman_1` through `normal_young_woman_5`

## Adding New Custom Speakers

1. Place audio files (.wav, .mp3, .flac, .ogg) in the `./speakers/` folder
2. Name files with the desired speaker ID (e.g., `hero_voice.wav` → speaker ID: `hero_voice`)
3. Restart the server - it will automatically process and cache the new speakers

**Audio requirements:**
- Format: WAV, MP3, FLAC, or OGG
- Duration: 3-30 seconds recommended
- Quality: Clear speech, minimal background noise
- Sample rate: Any (will be resampled automatically)

## Usage Examples

### 1. Simple Narration with Single Speaker
```bash
curl -X POST 'http://localhost:8088/v1/audio/speech' \
  -H 'Content-Type: application/json' \
  -d '{"text": "This is a simple narration.", "speaker": "main_storyteller_1"}' \
  --output narration.wav
```

### 2. Audiobook with Multiple Voices
```bash
cat > audiobook.json <<'EOF'
{
  "text": "[main_storyteller_1] Chapter One. The old house stood at the end of the street, its windows dark and mysterious. [silence 2s] [normal_old_man_1] Who goes there? [silence 0.5s] [normal_young_woman_1] It's just me, grandfather. I've come to visit."
}
EOF

curl -X POST 'http://localhost:8088/v1/audio/speech' \
  -H 'Content-Type: application/json' \
  -d @audiobook.json \
  --output audiobook.wav
```

### 3. Dialogue Scene
```bash
cat > dialogue.json <<'EOF'
{
  "text": "[normal_young_man_1] I can't believe we're finally here! [silence 0.3s] [normal_young_woman_1] I know, right? It's amazing! [silence 0.5s] [normal_old_man_1] Children, stay close. We don't know what dangers await."
}
EOF

curl -X POST 'http://localhost:8088/v1/audio/speech' \
  -H 'Content-Type: application/json' \
  -d @dialogue.json \
  --output dialogue.wav
```

### 4. Tutorial with Pauses
```bash
cat > tutorial.json <<'EOF'
{
  "text": "[main_storyteller_1] First, prepare the ingredients. [silence 1s] Next, mix them thoroughly. [silence 1s] Finally, bake for thirty minutes."
}
EOF

curl -X POST 'http://localhost:8088/v1/audio/speech' \
  -H 'Content-Type: application/json' \
  -d @tutorial.json \
  --output tutorial.wav
```

### 5. List Available Speakers
```bash
# Get all speakers
curl http://localhost:8088/v1/speakers

# Get just speaker IDs
curl -s http://localhost:8088/v1/speakers | python -c "import json, sys; print('\\n'.join(s['id'] for s in json.load(sys.stdin)['speakers']))"

# Count by source
curl -s http://localhost:8088/v1/speakers | python -c "import json, sys; print(json.load(sys.stdin)['counts'])"
```

## Error Handling

### Common Errors

**Speaker not found:**
```json
{"error": "Speaker(s) not found: unknown_speaker"}
```
→ Check speaker ID exists using GET /v1/speakers

**Invalid speaker:**
```json
{"error": "Invalid speaker: [speaker_name]"}
```
→ Ensure speaker is available (single-speaker mode)

**Missing text:**
```json
{"error": "Missing or empty 'text' field"}
```
→ Provide text in request body

## Performance Notes

- **First speaker processing**: 2-5 seconds per speaker (one-time cost)
- **Cache loading**: <1 second for all speakers
- **Single-speaker synthesis**: ~0.5-2 seconds for short text
- **Multi-speaker synthesis**: Linear with number of segments
- **Silence generation**: Near-zero cost (just array creation)

## Technical Details

### Architecture
- **CustomSpeakerManager**: Loads and caches custom speakers from audio files
- **UnifiedSpeakerRegistry**: Merges built-in and custom speakers into single registry
- **TextParser**: Parses embedded speaker and silence tags
- **MultiSpeakerInference**: Generates audio with automatic speaker switching

### File Structure
```
./speakers/                     # Custom speaker audio files
./cache/speakers/              # Built-in speaker embeddings
./cache/speakers/custom/       # Cached custom speaker embeddings
```

### Caching
- Speaker embeddings cached as `.safetensors` files
- Cache invalidation: Automatically detects source file modifications
- Cache location: `cache/speakers/custom/<speaker_id>.safetensors`

## Troubleshooting

**Server won't start:**
- Check if model files are downloaded in `cache/`
- Ensure `speakers/` directory exists (create if missing)

**Custom speaker not found:**
- Verify audio file is in `speakers/` folder
- Check file format is supported (.wav, .mp3, .flac, .ogg)
- Restart server to process new speakers

**Audio quality issues:**
- Use high-quality reference audio (clear speech, minimal noise)
- Ensure reference audio is 3-30 seconds long
- Try different speakers for comparison

**Slow synthesis:**
- First run processes all speakers (2-5 sec each)
- Subsequent runs use cache (<1 sec total)
- Multi-speaker synthesis is slower than single-speaker

## Python Client Example

```python
import requests
import json

# API endpoint
url = "http://localhost:8088/v1/audio/speech"

# Get available speakers
speakers_response = requests.get("http://localhost:8088/v1/speakers")
speakers = speakers_response.json()['speakers']
print(f"Available speakers: {len(speakers)}")

# Single-speaker synthesis
single_speaker_data = {
    "text": "Hello world, this is a test.",
    "speaker": "main_storyteller_1"
}
response = requests.post(url, json=single_speaker_data)
with open("output_single.wav", "wb") as f:
    f.write(response.content)

# Multi-speaker synthesis
multi_speaker_data = {
    "text": "[main_storyteller_1] Once upon a time... [silence 1s] [normal_young_man_1] Hello!"
}
response = requests.post(url, json=multi_speaker_data)
with open("output_multi.wav", "wb") as f:
    f.write(response.content)

print("Audio files generated successfully!")
```

## Summary

The multi-speaker TTS API supports:
- ✅ 123 total speakers (101 built-in + 22 custom)
- ✅ Backward-compatible single-speaker mode
- ✅ Multi-speaker synthesis with embedded tags
- ✅ Configurable silence insertion
- ✅ Automatic speaker caching for fast startup
- ✅ Speaker discovery via GET /v1/speakers

For questions or issues, check the server logs in `server.log`.
