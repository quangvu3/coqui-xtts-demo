import os
import time
import uuid
import hashlib
from pathlib import Path

import gradio as gr
import torch
import torchaudio
import numpy as np
import spaces

from huggingface_hub import hf_hub_download

from utils.vietnamese_normalization import normalize_vietnamese_text
from utils.logger import setup_logger
from utils.sentence import split_sentence, merge_sentences

from underthesea import sent_tokenize

from df.enhance import enhance, init_df, load_audio, save_audio

import warnings
warnings.filterwarnings("ignore")

logger = setup_logger(__file__)

df_model, df_state = None, None

checkpoint_dir="/tmp/xtts/model/"
temp_dir="/tmp/xtts/temp/"
enhance_audio_dir="/tmp/xtts/enhance/"
for d in [checkpoint_dir, temp_dir, enhance_audio_dir]:
    os.makedirs(d, exist_ok=True)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

language_dict = {'English': 'en', 'Spanish': 'es', 'French': 'fr', 
                 'German': 'de', 'Italian': 'it', 'Portuguese': 'pt', 
                 'Polish': 'pl', 'Turkish': 'tr', 'Russian': 'ru', 
                 'Dutch': 'nl', 'Czech': 'cs', 'Arabic': 'ar', 'Chinese': 'zh-cn', 
                 'Hungarian': 'hu', 'Korean': 'ko', 'Japanese': 'ja', 
                 'Hindi': 'hi', 'Vietnamese': 'vi'}

input_text_max_length = 2000

use_deepspeed = False
# if use_deepspeed:
#    from utils.cuda_toolkit import install_cuda_toolkit
#    logger.info("Installing CUDA toolkit...")
#    install_cuda_toolkit()

xtts_model = None
def load_model():
    global xtts_model
    
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    repo_id = "jimmyvu/xtts"
    
    model_files = ["config.json", "vocab.json", "harvard.wav", "model.pth"]
    for filename in model_files:
        if not os.path.exists(os.path.join(checkpoint_dir, filename)):
            logger.info(f"Downloading {filename} from Hugging Face...")
            hf_hub_download(repo_id=repo_id, 
                        filename=filename, 
                        local_dir=checkpoint_dir)

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

gpt_cond_latent_cache = {}
@spaces.GPU
def generate_speech(input_text, speaker_reference_audio, enhance_speech, temperature=0.3, top_p=0.85, top_k=50, repetition_penalty=10.0, language='English', *args):
    """Process text and generate audio."""
    global df_model, df_state, xtts_model
    log_messages = ""
    if len(input_text) > input_text_max_length:
        log_messages += "Text is too long! Please provide a shorter text.\n"
        return None, log_messages

    if len(input_text) < 2:
        log_messages += "Text is too short! Please provide a longer text.\n"
        return None, log_messages
    
    if not speaker_reference_audio:
        log_messages += "Please provide at least one reference audio!\n"
        return None, log_messages

    start = time.time()
    logger.info(f"Start processing text: {input_text[:30]}... [length: {len(input_text)}]")
    
    language_code = language_dict.get(language, 'en')
    if language_code == 'vi':
        input_text = normalize_vietnamese_text(input_text)

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

    speaker_hash = hashlib.sha1(speaker_reference_audio.encode("utf-8")).hexdigest()
    if speaker_hash not in gpt_cond_latent_cache:
        gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
            audio_path=speaker_reference_audio,
            gpt_cond_len=xtts_model.config.gpt_cond_len,
            max_ref_length=xtts_model.config.max_ref_len,
            sound_norm_refs=xtts_model.config.sound_norm_refs,
        )
        gpt_cond_latent_cache[speaker_hash] = (gpt_cond_latent, speaker_embedding)
    else:
        gpt_cond_latent, speaker_embedding = gpt_cond_latent_cache[speaker_hash]
    
    # Split text by sentence
    if language_code in ["ja", "zh-cn"]:
        sentences = input_text.split("。")
    else:
        sentences = sent_tokenize(input_text)
    # merge short sentences to next/prev ones
    sentences = merge_sentences(sentences)
    # inference
    wav_array = inference(sentences, language_code, gpt_cond_latent, speaker_embedding, temperature, top_p, top_k, repetition_penalty)
    end = time.time()
    logger.info(f"End processing text: {input_text[:30]}... Processing time: {end - start:.2f}s")
    log_messages += f"Processing time: {end - start:.2f}s"
    return (24000, wav_array), log_messages


def inference(sentences, language_code, gpt_cond_latent, speaker_embedding, temperature, top_p, top_k, repetition_penalty):
    # set dynamic length penalty from -1.0 to 1,0 based on text length
    max_text_length = 180
    dynamic_length_penalty = lambda text_length: (2 * (min(max_text_length, text_length) / max_text_length)) - 1
     # inference
    out_wavs = []
    for sentence in sentences:
        if len(sentence.strip()) == 0:
            continue
        # split too long sentence
        texts = split_sentence(sentence) if len(sentence) > max_text_length else [sentence]
        for text in texts:
            # logger.info(f"Processing text: {text}")
            try:
                out = xtts_model.inference(
                    text=text,
                    language=language_code,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=dynamic_length_penalty(len(text)),
                    enable_text_splitting=False,
                )
                out_wavs.append(out["wav"])
            except Exception as e:
                logger.error(f"Error processing text: {text} - {e}")
                
    return np.concatenate(out_wavs)

def build_gradio_ui():
    """Builds and launches the Gradio UI."""
    theme=gr.Theme.from_hub('JohnSmith9982/small_and_pretty')
    setattr(theme, 'button_secondary_background_fill', '#fcd53f')
    setattr(theme, 'checkbox_border_color', '#02c160')
    setattr(theme, 'input-border-width', '1px')
    setattr(theme, 'input-background-fill', '#ffffff')
    setattr(theme, 'input-background-fill_focus', '#ffffff')
    setattr(theme, 'input-border-color', '#d1d5db')
    setattr(theme, 'input-border-color_focus', '#fcd53f')

    default_prompt = ("Hi, I'm a multilingual text-to-speech model. "
                      "I can speak German: Hallo, wie geht es dir? "
                      "And French: Bonjour mesdames et messieurs. "
                      "I can also speak Vietnamese, Xin chào các bạn Việt Nam, "
                      "and many other languages. ")
        
    with gr.Blocks(title="Coqui XTTS Demo", theme=theme) as ui:
        gr.Markdown(
          """
          # 🐸 Coqui-XTTS Text-to-Speech Demo
          Convert text to speech with advanced voice cloning and enhancement. 
          Support 18 languages, \u2605 **Vietnamese** \u2605 newly added.
          """
        )

        with gr.Tab("Text to Speech"):
          with gr.Row():
            with gr.Column():
                input_text = gr.Text(label="Enter Text Here", 
                                     placeholder="Write the text you want to convert...", 
                                     value=default_prompt,
                                     lines=5, 
                                     max_length=input_text_max_length)
                speaker_reference_audio = gr.Audio(
                    label="Speaker reference audio:",
                    type="filepath",
                    editable=False,
                    min_length=3,
                    max_length=300,
                    value=os.path.join(checkpoint_dir, 'harvard.wav')
                )
                enhance_speech = gr.Checkbox(label="Enhance Reference Audio", value=False)
                language = gr.Dropdown(label="Target Language", choices=[k for k in language_dict.keys()], value="English")
                generate_button = gr.Button("Generate Speech")
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio")
                log_output = gr.Text(label="Log Output")

        with gr.Tab("Clone Your Voice"):
          with gr.Row():
            with gr.Column():
                input_text_mic = gr.Text(label="Enter Text Here", 
                                     placeholder="Write the text you want to convert...", 
                                     lines=5, 
                                     max_length=input_text_max_length)
                mic_ref_audio = gr.Audio(label="Record Reference Audio", sources=["microphone"])
                enhance_speech_mic = gr.Checkbox(label="Enhance Reference Audio", value=True)
                language_mic = gr.Dropdown(label="Target Language", choices=[k for k in language_dict.keys()], value="English")
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
                    repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=50.0, value=9.5, step=1.0)
                    
                with gr.Column():
                    top_p = gr.Slider(label="Top P", minimum=0.5, maximum=1.0, value=0.85, step=0.05)
                    top_k = gr.Slider(label="Top K", minimum=0, maximum=100, value=50, step=5)
                    
        generate_button.click(
            generate_speech,
            inputs=[input_text, speaker_reference_audio, enhance_speech, temperature, top_p, top_k, repetition_penalty, language],
            outputs=[audio_output, log_output],
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
