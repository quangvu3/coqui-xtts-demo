# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Coqui XTTS (Text-to-Speech) demo application supporting 17 languages with Vietnamese newly added. The project provides two interfaces:
- **Gradio web UI** (`gradio_app.py`) - Interactive web interface with three modes: Built-in Voice, Reference Voice, and Clone Your Voice
- **OpenAI-compatible API server** (`xtts_oai_server/xtts_server.py`) - aiohttp-based server exposing `/v1/audio/speech` endpoint

Both interfaces use the same underlying XTTS model from HuggingFace (`jimmyvu/xtts`) with support for voice cloning, audio enhancement, and multilingual synthesis.

## Running the Application

### Gradio Web Interface
```bash
python gradio_app.py
```
This launches an interactive web UI on the default Gradio port with three tabs for different TTS modes.

### OpenAI-Compatible API Server
```bash
python xtts_oai_server/xtts_server.py
```
Starts an aiohttp server on port 8088 with the endpoint `/v1/audio/speech`.

### Installing Dependencies
```bash
pip install -r requirements.txt
```

The application uses a custom fork of Coqui-XTTS from GitHub (`git+https://github.com/quangvu3/coqui-xtts.git`).

## Architecture

### Model Loading and Inference Flow

1. **Model initialization** happens at startup in both `gradio_app.py` and `xtts_server.py`:
   - Downloads model from HuggingFace (`jimmyvu/xtts`) to `cache/` directory
   - Loads safetensors checkpoint (not .pth files)
   - Moves to CUDA if available
   - Downloads unidic dictionary for Japanese text processing

2. **Text preprocessing pipeline**:
   - Language detection via `langdetect` (or use specified language)
   - Vietnamese text normalization (`utils/vietnamese_normalization.py`):
     - Converts numbers to Vietnamese words (handles both US and Vietnamese number formats)
     - Replaces abbreviations with full Vietnamese words
     - Converts currency symbols and Roman numerals
   - Sentence tokenization using `underthesea.sent_tokenize`
   - Sentence merging (`utils/sentence.py`) to ensure optimal length (min 12 words, max 250 chars)
   - Optional sentence splitting for very long sentences

3. **Audio generation**:
   - Speaker conditioning via either:
     - Built-in speaker ID from `xtts_model.speaker_manager.speakers`
     - Reference audio file with `get_conditioning_latents()`
   - Optional DeepFilterNet audio enhancement for reference audio (gradio only)
   - Per-sentence inference with configurable generation parameters (temperature, top_p, top_k, repetition_penalty)
   - Dynamic length penalty based on text length
   - Silence padding between sentences (API server only, 500ms default)

### Key Utilities

**`utils/vietnamese_normalization.py`**
Comprehensive Vietnamese text normalization:
- Detects number format (Vietnamese: `1.234,5` vs US: `1,234.5`)
- Converts numbers to Vietnamese words with proper grammar rules (e.g., "mười", "mươi", "lẻ", "lăm", "mốt")
- Extensive abbreviation dictionary for Vietnamese terms (HĐND, UBND, TPHCM, etc.)
- Currency symbol conversion (₫, $, €, ¥, etc.)
- Roman numeral to integer conversion

**`utils/sentence.py`**
Two sentence merging strategies:
- `merge_sentences()`: Forward-backward pass merging
- `merge_sentences_balanced()`: Bidirectional balanced merging (used in API server)
- `split_sentence()`: Recursive sentence splitting at delimiters or word boundaries

**`utils/logger.py`**
Logging configuration (examine this file for logging setup details)

**`utils/spaces.py`**
Fallback decorator when not running on HuggingFace Spaces

### Directory Structure

```
cache/                      # Model cache (gitignored)
├── model.safetensors      # XTTS model weights
├── config.json            # Model configuration
├── vocab.json             # Tokenizer vocabulary
├── audio_samples/         # Sample reference audio files
├── audio_enhances/        # Enhanced reference audio cache
├── temp/                  # Temporary audio files
└── speakers/              # Speaker embeddings

xtts_oai_server/           # OpenAI-compatible API server
└── xtts_server.py         # aiohttp server implementation

utils/                     # Shared utilities
├── vietnamese_normalization.py
├── sentence.py
├── logger.py
└── spaces.py
```

## Important Implementation Details

### Gradio UI vs API Server Differences

**Gradio (`gradio_app.py`)**:
- Uses `@spaces.GPU` decorator for GPU allocation on HuggingFace Spaces
- Three synthesis modes: Built-in Voice, Reference Voice, Clone Your Voice
- Optional DeepFilterNet enhancement for reference audio
- Text splitting disabled in inference (`enable_text_splitting=False`) - uses custom `split_sentence()` instead
- No silence padding between sentences

**API Server (`xtts_oai_server/xtts_server.py`)**:
- No `@spaces` decorator (standalone deployment)
- Only supports built-in speaker mode via `speaker` parameter
- Text splitting enabled in inference (`enable_text_splitting=True`)
- Adds 500ms silence between sentences via `sentence_silence_ms` parameter
- Uses `merge_sentences_balanced()` instead of `merge_sentences()`
- Implements `calculate_keep_len()` hack for short sentences to trim audio

### Language Support

Supported languages: English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Hungarian, Korean, Japanese, Vietnamese

Auto-detection: Set `language='Auto'` to use `langdetect` for automatic language detection.

### Generation Parameters

Key parameters for controlling synthesis quality:
- `temperature` (0.1-1.0, default 0.3 Gradio / 0.2 API): Controls randomness
- `top_p` (0.5-1.0, default 0.85): Nucleus sampling threshold
- `top_k` (0-100, default 50 Gradio / 70 API): Top-k sampling
- `repetition_penalty` (1.0-50.0, default 10.0 Gradio / 9.0 API): Prevents repetition
- `length_penalty`: Dynamically calculated based on text length

### Model and Data Paths

- HuggingFace model repo: `jimmyvu/xtts`
- Model downloads to: `{APP_DIR}/cache/`
- Default speaker: `"Aaron Dreschner"`
- Audio sample rate: 24000 Hz
- Max input text length: 3000 characters

## Common Patterns

### Adding a new language
1. Add to `language_dict` with language code
2. Handle sentence tokenization in `inference()` (special logic for CJK languages)
3. Consider adding language-specific normalization in `utils/`

### Modifying audio enhancement
Audio enhancement is only implemented in Gradio UI via DeepFilterNet. See `generate_speech()` around line 169-181 for the enhancement logic.

### Adjusting sentence merging behavior
Modify parameters in `merge_sentences()` or `merge_sentences_balanced()`:
- `min_words`: Minimum words per sentence (default 12)
- `max_chars`: Maximum characters per sentence (default 250)
