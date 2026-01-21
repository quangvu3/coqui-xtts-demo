import os
import sys
import time
import hashlib
import site
import subprocess

import gradio as gr
import torch
import torchaudio
import numpy as np

from underthesea import sent_tokenize
from df.enhance import enhance, init_df, load_audio, save_audio

from huggingface_hub import snapshot_download

from langdetect import detect

from utils.vietnamese_normalization import normalize_vietnamese_text
from utils.logger import setup_logger
from utils.sentence import split_sentence, merge_sentences
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

APP_DIR = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir=f"{APP_DIR}/cache"
temp_dir=f"{APP_DIR}/cache/temp/"
sample_audio_dir=f"{APP_DIR}/cache/audio_samples/"
enhance_audio_dir=f"{APP_DIR}/cache/audio_enhances/"
speakers_dir=f"{APP_DIR}/cache/speakers/"
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

input_text_max_length = 100000
use_deepspeed = False

# Initialize audio trimming configuration
TRIM_CONFIG = TrimConfig()

try:
    import spaces
except ImportError:
    from utils import spaces

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

# MultiSpeakerInference will be initialized after inference() function is defined
multi_speaker_engine = None

logger.info(f"Multi-speaker support initialized with {len(speaker_registry.list_all_speakers())} speakers")

default_speaker_reference_audio = os.path.join(sample_audio_dir, 'harvard.wav')
default_speaker_id = "Aaron Dreschner"

def validate_input(input_text, language):
    log_messages = ""
    if len(input_text) > input_text_max_length:
        gr.Warning("Text is too long! Please provide a shorter text.")
        log_messages += "Text is too long! Please provide a shorter text.\n"
        return log_messages
    
    language_code = language_dict.get(language, 'en')
    logger.info(f"Language [{language}], code: [{language_code}]")
    lang = lang_detect(input_text) if language == 'Auto' else language_code
    if (lang not in ['ja', 'kr', 'zh-cn'] and len(input_text.split()) < 2) or \
        (lang in ['ja', 'kr', 'zh-cn'] and len(input_text) < 2):
        gr.Warning("Text is too short! Please provide a longer text.")
        log_messages += "Text is too short! Please provide a longer text.\n"
        
    return log_messages
    
@spaces.GPU
def synthesize_speech(input_text, speaker_id, temperature=0.3, top_p=0.85, top_k=50, repetition_penalty=10.0, language='Auto'):
    """Process text and generate audio."""
    global xtts_model
    log_messages = validate_input(input_text, language)
    if log_messages: 
        return None, log_messages

    start = time.time()
    logger.info(f"Start processing text: {input_text[:30]}... [length: {len(input_text)}]")

    # Check if Auto mode with tags
    if speaker_id == "Auto" and text_parser.has_tags(input_text):
        # Multi-speaker mode
        logger.info("Using multi-speaker mode (Auto + tags detected)")
        try:
            segments = text_parser.parse_text(input_text)
            text_parser.validate_speakers(segments)

            wav_array, num_of_tokens = multi_speaker_engine.synthesize_segments(
                segments,
                language=language,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                sentence_silence_ms=500  # Add 0.5s silence between sentences
            )
        except ValueError as e:
            gr.Warning(f"Multi-speaker error: {e}")
            log_messages += f"Error: {e}"
            return None, log_messages
    else:
        # Single-speaker mode (existing logic)
        logger.info("Using single-speaker mode")
        wav_array, num_of_tokens = inference(input_text=input_text,
                              language=language,
                              speaker_id=speaker_id if speaker_id != "Auto" else default_speaker_id,
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
    log_messages += message
    return (24000, wav_array), log_messages
    

@spaces.GPU
def generate_speech(input_text, speaker_reference_audio, enhance_speech, temperature=0.3, top_p=0.85, top_k=50, repetition_penalty=10.0, language='Auto'):
    """Process text and generate audio."""
    global df_model, df_state, xtts_model
    log_messages = validate_input(input_text, language)
    if log_messages: 
        return None, log_messages
    
    if not speaker_reference_audio:
        gr.Warning("Please provide at least one reference audio!")
        log_messages += "Please provide at least one reference audio!\n"
        return None, log_messages

    start = time.time()
    logger.info(f"Start processing text: {input_text[:30]}... [length: {len(input_text)}]")
    
    if enhance_speech:
        logger.info("Enhancing reference audio...")
        _, audio_file = os.path.split(speaker_reference_audio)
        enhanced_audio_path = os.path.join(enhance_audio_dir, f"{audio_file}.enh.wav")
        if not os.path.exists(enhanced_audio_path):
            if not df_model:
                df_model, df_state, _ = init_df()
            audio, _ = load_audio(speaker_reference_audio, sr=df_state.sr())
            # denoise audio
            enhanced_audio = enhance(df_model, df_state, audio)
            # save enhanced audio
            save_audio(enhanced_audio_path, enhanced_audio, sr=df_state.sr())
        speaker_reference_audio = enhanced_audio_path

    gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
        audio_path=speaker_reference_audio,
        gpt_cond_len=xtts_model.config.gpt_cond_len,
        max_ref_length=xtts_model.config.max_ref_len,
        sound_norm_refs=xtts_model.config.sound_norm_refs,
    )
    
    # inference
    wav_array, num_of_tokens = inference(input_text=input_text, 
                          language=language,
                          speaker_id=None, 
                          gpt_cond_latent=gpt_cond_latent, 
                          speaker_embedding=speaker_embedding, 
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
    log_messages += message
    return (24000, wav_array), log_messages


def inference(input_text, language, speaker_id=None, gpt_cond_latent=None, speaker_embedding=None, temperature=0.3, top_p=0.85, top_k=50, repetition_penalty=29.0, sentence_silence_ms=500):
    # If a language is specified, use it, otherwise detect it.
    # This is used for sentence splitting.
    lang_for_split = language_dict.get(language, 'en') if language != 'Auto' else lang_detect(input_text)

    # Split text by sentence
    if lang_for_split in ["ja", "zh-cn"]:
        sentences = input_text.split("。")
    else:
        sentences = sent_tokenize(input_text)
    # merge short sentences to next/prev ones
    sentences = merge_sentences(sentences)

    # max_text_length is used for split_sentence() only
    max_text_length = 180

    if speaker_id is not None:
        # Use speaker_registry to support both built-in and custom speakers
        gpt_cond_latent, speaker_embedding = speaker_registry.get_speaker(speaker_id)

    # inference
    out_wavs = []
    num_of_tokens = 0
    for i, sentence in enumerate(sentences):
        if len(sentence.strip()) == 0:
            continue
        
        # If a language is specified, use it, otherwise detect from the sentence.
        # This is used for inference.
        lang_for_inference = language_dict.get(language, 'en') if language != 'Auto' else lang_detect(sentence)

        if lang_for_inference == 'vi':
            sentence = normalize_vietnamese_text(sentence)
        text_tokens = torch.IntTensor(xtts_model.tokenizer.encode(sentence, lang=lang_for_inference)).unsqueeze(0).to(xtts_model.device)
        num_of_tokens += text_tokens.shape[-1]
        txts = split_sentence(sentence, max_text_length=max_text_length)
        for txt in txts:
            logger.info(f"[{lang_for_inference}] {txt}")
            try:
                out = xtts_model.inference(
                    text=txt,
                    language=lang_for_inference,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty*1.0,
                    length_penalty=calculate_length_penalty(text_length=len(txt), max_length=max_text_length, exponent=2.0),
                    enable_text_splitting=True,
                )

                # Trim audio to remove excess silence and over-generation
                # For short text (<11 words), validate length and retry if over-generated
                trimmed_wav = validate_audio_length(
                    audio=out["wav"],
                    text=txt,
                    language=lang_for_inference,
                    sample_rate=24000,
                    inference_fn=xtts_model.inference,
                    word_threshold=11,
                    length_tolerance=1.3,
                    max_retries=5,
                    config=TRIM_CONFIG,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty*1.0,
                    enable_text_splitting=True,
                )
                out_wavs.append(trimmed_wav)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
        
        # Add silence after each sentence (except the last one)
        if sentence_silence_ms > 0 and i < len(sentences) - 1:
            silence_samples = int((sentence_silence_ms / 1000.0) * 24000)
            silence = np.zeros(silence_samples, dtype=np.float32)
            out_wavs.append(silence)

    return np.concatenate(out_wavs), num_of_tokens


#  Initialize multi-speaker engine now that inference function is defined
multi_speaker_engine = MultiSpeakerInference(
    xtts_model=xtts_model,
    speaker_registry=speaker_registry,
    inference_fn=inference,
    soundtrack_manager=soundtrack_manager
)
logger.info("Multi-speaker inference engine initialized")

def build_gradio_ui():
    """Builds and launches the Gradio UI."""
    
    default_prompt = ("Hi, I am a multilingual text-to-speech AI model.\n"
                      "Bonjour, je suis un modèle d'IA de synthèse vocale multilingue.\n"
                      "Hallo, ich bin ein mehrsprachiges Text-zu-Sprache KI-Modell.\n"
                      "Ciao, sono un modello di intelligenza artificiale di sintesi vocale multilingue.\n"
                      "Привет, я многоязычная модель искусственного интеллекта, преобразующая текст в речь.\n"
                      "Xin chào, tôi là một mô hình AI chuyển đổi văn bản thành giọng nói đa ngôn ngữ.\n")
        
    
    def update_help_text(speaker):
        """Update helper text visibility based on speaker selection."""
        if speaker == "Auto":
            return gr.update(visible=True, value="**Auto mode enabled**: Use tags like `[speaker_id] text`, `[silence 2s]`, and `[soundtrack 10s fadeout:3s]` in your text. Example: `[soundtrack 10s] [narrator] Once upon a time... [silence 1s] [hero] Hello!`")
        else:
            return gr.update(visible=False)

    with gr.Blocks(title="Coqui XTTS Demo", theme='jimmyvu/small_and_pretty') as ui:
        gr.Markdown(
          """
          # 🐸 Coqui-XTTS Text-to-Speech Demo
          Convert text to speech with advanced voice cloning and enhancement. 
          Support 17 languages, \u2605 **Vietnamese** \u2605 newly added.
          """
        )

        with gr.Tab("Built-in Voice"):
          with gr.Row():
            with gr.Column():
                input_text = gr.Text(label="Enter Text Here", 
                                     placeholder="Write the text you want to synthesize...", 
                                     value=default_prompt,
                                     lines=5, 
                                     max_length=input_text_max_length)
                
                speaker_id = gr.Dropdown(label="Speaker", choices=["Auto"] + [s["id"] for s in speaker_registry.list_all_speakers()], value=default_speaker_id)
                help_text = gr.Markdown(visible=False)
                language = gr.Dropdown(label="Target Language", choices=[k for k in language_dict.keys()], value=default_language)
                synthesize_button = gr.Button("Generate Speech")
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio")
                log_output = gr.Text(label="Log Output")

        # Connect speaker dropdown to help text
        speaker_id.change(update_help_text, inputs=[speaker_id], outputs=[help_text])


        with gr.Tab("Reference Voice"):
          with gr.Row():
            with gr.Column():
                input_text_generate = gr.Text(label="Enter Text Here", 
                                     placeholder="Write the text you want to synthesize...", 
                                     lines=5, 
                                     max_length=input_text_max_length)
                speaker_reference_audio = gr.Audio(
                    label="Speaker reference audio:",
                    type="filepath",
                    editable=False,
                    min_length=3,
                    max_length=300,
                    value=default_speaker_reference_audio
                )
                enhance_speech = gr.Checkbox(label="Enhance Reference Audio", value=False)
                language_generate = gr.Dropdown(label="Target Language", choices=[k for k in language_dict.keys()], value=default_language)
                generate_button = gr.Button("Generate Speech")
            with gr.Column():
                audio_output_generate = gr.Audio(label="Generated Audio")
                log_output_generate = gr.Text(label="Log Output")

        with gr.Tab("Clone Your Voice"):
          with gr.Row():
            with gr.Column():
                input_text_mic = gr.Text(label="Enter Text Here", 
                                     placeholder="Write the text you want to synthesize...", 
                                     lines=5, 
                                     max_length=input_text_max_length)
                mic_ref_audio = gr.Audio(label="Record Reference Audio", sources=["microphone"])
                enhance_speech_mic = gr.Checkbox(label="Enhance Reference Audio", value=True)
                language_mic = gr.Dropdown(label="Target Language", choices=[k for k in language_dict.keys()], value=default_language)
                generate_button_mic = gr.Button("Generate Speech")
            with gr.Column():
                audio_output_mic = gr.Audio(label="Generated Audio")
                log_output_mic = gr.Text(label="Log Output")


        def process_mic_and_generate(input_text_mic, mic_ref_audio, enhance_speech_mic, temperature, top_p, top_k, repetition_penalty, language_mic):
              if mic_ref_audio:
                  data = str(time.time()).encode("utf-8")
                  hash = hashlib.sha1(data).hexdigest()[:10]
                  output_path = os.path.join(temp_dir, (f"mic_{hash}.wav"))

                  torch_audio = torch.from_numpy(mic_ref_audio[1].astype(float))
                  try:
                      torchaudio.save(output_path, torch_audio.unsqueeze(0), mic_ref_audio[0])
                      return generate_speech(input_text_mic, output_path, enhance_speech_mic, temperature, top_p, top_k, repetition_penalty, language_mic)
                  except Exception as e:
                      logger.error(f"Error saving audio file: {e}")
                      return None, f"Error saving audio file: {e}"
              else:
                  return None, "Please record an audio!"

        with gr.Tab("Advanced Settings"):
            with gr.Row():
                with gr.Column():
                    temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, value=0.3, step=0.05)
                    repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=50.0, value=29.0, step=1.0)
                    
                with gr.Column():
                    top_p = gr.Slider(label="Top P", minimum=0.5, maximum=1.0, value=0.85, step=0.05)
                    top_k = gr.Slider(label="Top K", minimum=0, maximum=100, value=50, step=5)
        
        synthesize_button.click(
            synthesize_speech,
            inputs=[input_text, speaker_id, temperature, top_p, top_k, repetition_penalty, language],
            outputs=[audio_output, log_output],
        )
        
        generate_button.click(
            generate_speech,
            inputs=[input_text_generate, speaker_reference_audio, enhance_speech, temperature, top_p, top_k, repetition_penalty, language_generate],
            outputs=[audio_output_generate, log_output_generate],
        )

        generate_button_mic.click(
            process_mic_and_generate,
            inputs=[input_text_mic, mic_ref_audio, enhance_speech_mic, temperature, top_p, top_k, repetition_penalty, language_mic],
            outputs=[audio_output_mic, log_output_mic],
        )
        
    return ui

if __name__ == "__main__":
    ui = build_gradio_ui()
    ui.launch(debug=False)
