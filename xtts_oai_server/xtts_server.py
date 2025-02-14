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
from utils.sentence import split_sentence, merge_sentences

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

xtts_model = None
def load_model():
    global xtts_model
    
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    repo_id = "jimmyvu/xtts"
    snapshot_download(repo_id=repo_id, 
                      local_dir=checkpoint_dir, 
                      allow_patterns=["*.safetensors", "*.wav", "*.json"], 
                      ignore_patterns="*.pth")

    config = XttsConfig()
    config.load_json(os.path.join(checkpoint_dir, "config.json"))
    xtts_model = Xtts.init_from_config(config)

    logger.info("Loading model...")
    xtts_model.load_safetensors_checkpoint(
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

default_speaker_id = "Aaron Dreschner"

def synthesize_speech(input_text, speaker_id, temperature=0.3, top_p=0.85, top_k=50, repetition_penalty=10.0, language='Auto'):
    """Process text and generate audio."""
    global xtts_model
    
    start = time.time()
    logger.info(f"Start processing text: {input_text[:30]}... [length: {len(input_text)}]")
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
    return (24000, wav_array)


def inference(input_text, language, speaker_id=None, gpt_cond_latent=None, speaker_embedding=None, temperature=0.3, top_p=0.85, top_k=50, repetition_penalty=10.0):
    language_code = lang_detect(input_text) if language == 'Auto' else language_dict.get(language, 'en')
    # Split text by sentence
    if language_code in ["ja", "zh-cn"]:
        sentences = input_text.split("。")
    else:
        sentences = sent_tokenize(input_text)
    # merge short sentences to next/prev ones
    sentences = merge_sentences(sentences)
    
    # set dynamic length penalty from -1.0 to 1,0 based on text length
    max_text_length = 180
    dynamic_length_penalty = lambda text_length: (2 * (min(max_text_length, text_length) / max_text_length)) - 1

    if speaker_id is not None:
        gpt_cond_latent, speaker_embedding = xtts_model.speaker_manager.speakers[speaker_id].values()

    # inference
    out_wavs = []
    num_of_tokens = 0
    for sentence in sentences:
        if len(sentence.strip()) == 0:
            continue
        lang = lang_detect(sentence) if language == 'Auto' else language_code
        if lang == 'vi':
            sentence = normalize_vietnamese_text(sentence)
        text_tokens = torch.IntTensor(xtts_model.tokenizer.encode(sentence, lang=lang)).unsqueeze(0).to(xtts_model.device)
        num_of_tokens += text_tokens.shape[-1]
        logger.info(f"[{lang}] {sentence}")
        try:
            out = xtts_model.inference(
                text=sentence,
                language=lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=dynamic_length_penalty(len(sentence)),
                enable_text_splitting=True,
            )
            out_wavs.append(out["wav"])
        except Exception as e:
            logger.error(f"Error processing text: {sentence} - {e}")            
    return np.concatenate(out_wavs), num_of_tokens


async def handle_speech_request(request):
    """Handles the /v1/audio/speech endpoint, generating audio from text."""
    try:
        # Important: Validate the request's content
        request_data = await request.json()
        text_to_speak = request_data.get('text')
        
        if not text_to_speak:
            return web.json_response({"error": "Missing or empty 'text' field"}, status=400)

        speaker_id = request_data.get('speaker', default_speaker_id)

        # Initialize the text-to-speech engine.
        if speaker_id not in xtts_model.speaker_manager.speakers:
            return web.json_response({"Error": f"Invalid speaker: [{speaker_id}]"}, status=400)
        
        sample_rate, wav_array = synthesize_speech(text_to_speak, speaker_id=speaker_id)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name
            torchaudio.save(tmp_file_path, torch.tensor(wav_array).unsqueeze(0), sample_rate)

            # Crucial: Check if audio file was created.
            if not os.path.exists(tmp_file_path):
                return web.json_response({"Error": "Failed to generate audio"}, status=500)

            # Prepare the response
            response = web.FileResponse(
                tmp_file_path,
                headers=[('Content-Disposition', 'attachment; filename="speech.wav"')]
            )

            return response
    except Exception as e:
        print(e)
        return web.json_response({"error": f"An error occurred: {str(e)}"}, status=500)


async def main():
    app = web.Application()
    app.router.add_post('/v1/audio/speech', handle_speech_request)

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