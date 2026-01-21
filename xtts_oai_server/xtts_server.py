import os
import sys
import time
import site
import subprocess
import tempfile

import asyncio
from aiohttp import web

import torch
import torchaudio
import numpy as np

from underthesea import sent_tokenize
from df.enhance import enhance, init_df, load_audio, save_audio

from huggingface_hub import snapshot_download

from langdetect import detect

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(APP_DIR)

from utils.vietnamese_normalization import normalize_vietnamese_text
from utils.logger import setup_logger
from utils.sentence import split_sentence, merge_sentences, merge_sentences_balanced
from utils.length_penalty import calculate_length_penalty
from utils.audio_trimmer import trim_audio, validate_audio_length, TrimConfig

# Import multi-speaker support modules
from xtts_oai_server.custom_speaker_manager import CustomSpeakerManager
from xtts_oai_server.speaker_registry import UnifiedSpeakerRegistry
from xtts_oai_server.text_parser import TextParser
from xtts_oai_server.multi_speaker_inference import MultiSpeakerInference
from xtts_oai_server.soundtrack_manager import SoundtrackManager

import warnings
warnings.filterwarnings("ignore")

logger = setup_logger(__file__)

df_model, df_state = None, None

checkpoint_dir=f"{APP_DIR}/cache"
temp_dir=f"{APP_DIR}/cache/temp/"
sample_audio_dir=f"{APP_DIR}/cache/audio_samples/"
enhance_audio_dir=f"{APP_DIR}/cache/audio_enhances/"
for d in [checkpoint_dir, temp_dir, sample_audio_dir, enhance_audio_dir]:
    os.makedirs(d, exist_ok=True)

language_dict = {'English': 'en', 'Español (Spanish)': 'es', 'Français (French)': 'fr', 
                 'Deutsch (German)': 'de', 'Italiano (Italian)': 'it', 'Português (Portuguese)': 'pt', 
                 'Polski (Polish)': 'pl', 'Türkçe (Turkish)': 'tr', 'Русский (Russian)': 'ru', 
                 'Nederlands (Dutch)': 'nl', 'Čeština (Czech)': 'cs', 'العربية (Arabic)': 'ar', '中文 (Chinese)': 'zh-cn',
                 'Magyar nyelv (Hungarian)': 'hu', '한국어 (Korean)': 'ko', '日本語 (Japanese)': 'ja', 
                 'Tiếng Việt (Vietnamese)': 'vi', 'Auto': 'auto'}

default_language = 'Auto'
language_codes = [v for _, v in language_dict.items()]
def lang_detect(text):
    try:
        lang = detect(text)
        if lang == 'zh-tw':
            return 'zh-cn'
        return lang if lang in language_codes else 'en'
    except:
        return 'en'

input_text_max_length = 3000
use_deepspeed = False

# Initialize audio trimming configuration
TRIM_CONFIG = TrimConfig()

xtts_model = None
def load_model():
    global xtts_model
    
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    model_safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    model_legacy_path = os.path.join(checkpoint_dir, "model.pth")
    if not os.path.exists(model_safetensors_path) and not os.path.exists(model_legacy_path):
        repo_id = "jimmyvu/xtts"
        snapshot_download(repo_id=repo_id, 
                        local_dir=checkpoint_dir, 
                        allow_patterns=["*.safetensors", "*.wav", "*.json"], 
                        ignore_patterns="*.pth")

    config = XttsConfig()
    config.load_json(os.path.join(checkpoint_dir, "config.json"))
    xtts_model = Xtts.init_from_config(config)

    logger.info("Loading model...")
    xtts_model.load_checkpoint(
        config, checkpoint_dir=checkpoint_dir, use_deepspeed=use_deepspeed
    )
    if torch.cuda.is_available():
        xtts_model.cuda()
    logger.info(f"Successfully loaded model from {checkpoint_dir}")

load_model()

def download_unidic():
    site_package_path = site.getsitepackages()[0]
    unidic_path = os.path.join(site_package_path, "unidic", "dicdir")
    if not os.path.exists(unidic_path):
        logger.info("Downloading unidic...")
        subprocess.call([sys.executable, "-m", "unidic", "download"])

download_unidic()

# Initialize multi-speaker support
logger.info("Initializing multi-speaker support...")

custom_speakers_dir = f"{APP_DIR}/speakers"
custom_cache_dir = f"{APP_DIR}/cache/speakers/custom"
os.makedirs(custom_cache_dir, exist_ok=True)

# Initialize soundtrack manager
soundtracks_dir = f"{APP_DIR}/soundtracks"
os.makedirs(soundtracks_dir, exist_ok=True)
soundtrack_manager = SoundtrackManager(soundtrack_folder=soundtracks_dir)
logger.info(f"Soundtrack manager initialized with {soundtrack_manager.get_soundtrack_count()} soundtracks")

custom_speaker_manager = CustomSpeakerManager(
    xtts_model=xtts_model,
    speakers_dir=custom_speakers_dir,
    cache_dir=custom_cache_dir
)
custom_speaker_manager.scan_and_load_speakers()

speaker_registry = UnifiedSpeakerRegistry(
    xtts_model=xtts_model,
    custom_speaker_manager=custom_speaker_manager
)
speaker_registry.build_registry()

text_parser = TextParser(speaker_registry)

# We'll initialize multi_speaker_engine after defining the inference function
multi_speaker_engine = None

logger.info(f"Multi-speaker support initialized with {len(speaker_registry.list_all_speakers())} speakers")

default_speaker_id = "Aaron Dreschner"

# Cache for last used speaker ID - used as fallback when no speaker is specified in multi-chunk requests
last_used_speaker_id = default_speaker_id

def synthesize_speech(input_text, speaker_id, temperature=0.3, top_p=0.85, top_k=50, repetition_penalty=29.0, language='Auto'):
    """Process text and generate audio."""
    global xtts_model
    
    start = time.time()
    logger.info(f"Start processing text: {input_text[:30]}... [length: {len(input_text)}]")
    logger.info(f"Speaker ID: {speaker_id}")
    # inference
    wav_array, num_of_tokens = inference(input_text=input_text, 
                          language=language,
                          speaker_id=speaker_id, 
                          gpt_cond_latent=None, 
                          speaker_embedding=None, 
                          temperature=temperature, 
                          top_p=top_p, 
                          top_k=top_k, 
                          repetition_penalty=float(repetition_penalty))
    end = time.time()
    processing_time = end - start
    tokens_per_second = num_of_tokens/processing_time
    logger.info(f"End processing text: {input_text[:30]}")
    message = f"💡 {tokens_per_second:.1f} tok/s • {num_of_tokens} tokens • in {processing_time:.2f} seconds"
    logger.info(message)
    return (xtts_model.config.audio.sample_rate, wav_array)


def add_silence_to_wav_array(wav_array, sample_rate=24000, silence_ms=1000):
    """Add silence to the end of a wav numpy array"""
    # Convert torch tensor to numpy if needed
    if hasattr(wav_array, 'cpu'):  # torch tensor
        wav_array = wav_array.cpu().numpy()

    # Handle 2D array input (shape: [1, samples] or [channels, samples])
    if isinstance(wav_array, np.ndarray) and wav_array.ndim > 1:
        wav_array = wav_array.squeeze()

    # Calculate number of silence samples needed
    silence_samples = int((silence_ms / 1000.0) * sample_rate)

    # Create silence array (zeros)
    silence = np.zeros(silence_samples, dtype=wav_array.dtype)

    # Concatenate original audio with silence
    return np.concatenate([wav_array, silence])

def inference(input_text, language, speaker_id=None, gpt_cond_latent=None, speaker_embedding=None,
              temperature=0.3, top_p=0.85, top_k=50, repetition_penalty=29.0, sentence_silence_ms=500):
    """
    Generate speech from text with silence padding options.
    
    Args:
        input_text: Text to synthesize
        language: Target language
        speaker_id: Speaker identifier for voice cloning
        gpt_cond_latent: GPT conditioning latent (optional)
        speaker_embedding: Speaker embedding (optional)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty factor
        sentence_silence_ms: Silence to add after each sentence (milliseconds)
    
    Returns:
        tuple: (final_wav_array, num_of_tokens)
    """
    language_code = lang_detect(input_text) if language == 'Auto' else language_dict.get(language, 'en')
    
    # Split text by sentence
    if language_code in ["ja", "zh-cn"]:
        sentences = input_text.split("。")
    else:
        sentences = sent_tokenize(input_text)

    # merge short sentences to next/prev ones
    sentences = merge_sentences_balanced(sentences)

    if speaker_id is not None:
        # Use speaker_registry to support both built-in and custom speakers
        gpt_cond_latent, speaker_embedding = speaker_registry.get_speaker(speaker_id)

    # max_text_length is used for split_sentence() only
    max_text_length = 180

    # inference
    out_wavs = []
    num_of_tokens = 0

    for i, sentence in enumerate(sentences):
        if len(sentence.strip()) == 0:
            continue

        lang = lang_detect(sentence) if language == 'Auto' else language_code
        if lang == 'vi':
            sentence = normalize_vietnamese_text(sentence)

        # Split sentence if too long (matches Gradio app pattern)
        txts = split_sentence(sentence, max_text_length=max_text_length)
        for txt in txts:
            text_tokens = torch.IntTensor(xtts_model.tokenizer.encode(txt, lang=lang)).unsqueeze(0).to(xtts_model.device)
            num_of_tokens += text_tokens.shape[-1]
            logger.info(f"[{lang}] {txt}")

            try:
                out = xtts_model.inference(
                    text=txt,
                    language=lang,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=calculate_length_penalty(text_length=len(txt), max_length=max_text_length, exponent=2.0),
                    enable_text_splitting=True,
                )

                # For short text (<11 words), validate length and retry if over-generated
                sentence_wav = validate_audio_length(
                    audio=out["wav"],
                    text=txt,
                    language=lang,
                    sample_rate=xtts_model.config.audio.sample_rate,
                    inference_fn=xtts_model.inference,
                    word_threshold=15,
                    length_tolerance=1.2,
                    max_retries=10,
                    config=TRIM_CONFIG,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    enable_text_splitting=True,
                )

                # Add silence after each sentence (except the last one)
                if sentence_silence_ms > 0 and i < len(sentences) - 1:
                    sentence_wav = add_silence_to_wav_array(
                        sentence_wav,
                        sample_rate=xtts_model.config.audio.sample_rate,
                        silence_ms=sentence_silence_ms
                    )

                out_wavs.append(sentence_wav)

            except Exception as e:
                logger.error(f"Error processing text: {txt} - {e}")

    # Concatenate all sentences
    if out_wavs:
        final_wav = np.concatenate(out_wavs)
    else:
        # Return empty array if no audio was generated
        final_wav = np.array([])
    
    return final_wav, num_of_tokens


# Initialize multi-speaker engine now that inference function is defined
multi_speaker_engine = MultiSpeakerInference(
    xtts_model=xtts_model,
    speaker_registry=speaker_registry,
    inference_fn=inference,
    soundtrack_manager=soundtrack_manager
)
logger.info("Multi-speaker inference engine initialized")


async def handle_speech_request(request):
    """Handles the /v1/audio/speech endpoint with multi-speaker support."""
    global last_used_speaker_id

    try:
        # Validate the request's content
        request_data = await request.json()
        text_to_speak = request_data.get('text')
        language = request_data.get('language', 'Auto')


        if not text_to_speak:
            return web.json_response({"error": "Missing or empty 'text' field"}, status=400)

        # Check for embedded speaker tags
        has_tags = text_parser.has_tags(text_to_speak)

        if has_tags:
            # Auto mode: parse tags and use multi-speaker synthesis
            try:
                logger.info("Using multi-speaker mode (tags detected)")

                # Get default speaker for segments without tags
                # Use cached speaker as fallback when request doesn't specify one
                default_speaker = request_data.get('speaker', last_used_speaker_id)

                # Parse text into segments
                segments = text_parser.parse_text(text_to_speak, default_speaker=default_speaker)

                # Validate all speakers exist
                text_parser.validate_speakers(segments)

                # Log segment info
                stats = text_parser.segment_stats(segments)
                logger.info(f"Parsed {stats['total_segments']} segments: "
                           f"{stats['speech_segments']} speech, "
                           f"{stats['silence_segments']} silence, "
                           f"{stats['soundtrack_segments']} soundtrack, "
                           f"{stats['unique_speakers']} unique speakers")

                # Multi-speaker synthesis
                wav_array, _ = multi_speaker_engine.synthesize_segments(
                    segments,
                    language=language,
                    temperature=0.3,
                    top_p=0.85,
                    top_k=50,
                    repetition_penalty=29.0,
                    sentence_silence_ms=500
                )

                sample_rate = xtts_model.config.audio.sample_rate

                # Update cached speaker with the last speech segment's speaker
                speech_segments = [s for s in segments if s['type'] == 'speech']
                if speech_segments:
                    last_used_speaker_id = speech_segments[-1]['speaker_id']
                    logger.info(f"Updated cached speaker to: {last_used_speaker_id}")

            except ValueError as e:
                # Speaker validation or parsing error
                logger.error(f"Multi-speaker error: {e}")
                return web.json_response({"error": str(e)}, status=400)

        else:
            # Single speaker mode (backward compatible)
            logger.info("Using single-speaker mode")

            # Use cached speaker as fallback when request doesn't specify one
            speaker_id = request_data.get('speaker', last_used_speaker_id)

            # Check if speaker exists in registry
            if not speaker_registry.speaker_exists(speaker_id):
                return web.json_response(
                    {"error": f"Invalid speaker: [{speaker_id}]"},
                    status=400
                )

            # Use existing synthesize_speech function
            sample_rate, wav_array = synthesize_speech(text_to_speak, speaker_id=speaker_id, language=language)

            # Update cached speaker
            last_used_speaker_id = speaker_id
            logger.info(f"Updated cached speaker to: {last_used_speaker_id}")

        # Save and return audio (common for both modes)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name
            torchaudio.save(tmp_file_path, torch.tensor(wav_array).unsqueeze(0), sample_rate)

            # Check if audio file was created
            if not os.path.exists(tmp_file_path):
                return web.json_response({"error": "Failed to generate audio"}, status=500)

            # Prepare the response
            response = web.FileResponse(
                tmp_file_path,
                headers=[('Content-Disposition', 'attachment; filename="speech.wav"')]
            )

            return response

    except Exception as e:
        logger.error(f"Error in handle_speech_request: {e}", exc_info=True)
        return web.json_response({"error": f"An error occurred: {str(e)}"}, status=500)


async def handle_speakers_list(request):
    """Handle GET /v1/speakers endpoint to list all available speakers."""
    try:
        speakers = speaker_registry.list_all_speakers()

        # Format response
        response_data = {
            'speakers': [
                {
                    'id': s['id'],
                    'source': s['source'],
                    'cached': s.get('cached', True)
                }
                for s in speakers
            ],
            'total': len(speakers)
        }

        # Add speaker count by source
        counts = speaker_registry.get_speaker_count()
        response_data['counts'] = counts

        return web.json_response(response_data)

    except Exception as e:
        logger.error(f"Error listing speakers: {e}", exc_info=True)
        return web.json_response({"error": str(e)}, status=500)


async def main():
    app = web.Application()
    app.router.add_post('/v1/audio/speech', handle_speech_request)
    app.router.add_get('/v1/speakers', handle_speakers_list)

    runner = web.AppRunner(app)
    await runner.setup()
    port = 8088
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    print(f"Server started on port {port}")

    try:
        await asyncio.Future()  # Keep the event loop running
    except KeyboardInterrupt:
        print("Server stopped.")
        await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())