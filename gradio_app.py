import os
import time
import uuid
import hashlib
from pathlib import Path

import gradio as gr
import torch
import torchaudio

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import hf_hub_download

from utils.vietnamese_normalization import normalize_vietnamese_text
from utils.logger import setup_logger

logger = setup_logger(__file__)

xtts_model = None
checkpoint_dir="/tmp/xtts/model/"
temp_dir="/tmp/xtts/temp/"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_AUDIO = os.path.join(APP_DIR, "samples", "nam-calm.wav")

language_dict = {'English': 'en', 'Spanish': 'es', 'French': 'fr', 
                 'German': 'de', 'Italian': 'it', 'Portuguese': 'pt', 
                 'Polish': 'pl', 'Turkish': 'tr', 'Russian': 'ru', 
                 'Dutch': 'nl', 'Czech': 'cs', 'Arabic': 'ar', 'Simplified Chinese': 'zh-cn', 
                 'Hungarian': 'hu', 'Korean': 'ko', 'Japanese': 'ja', 
                 'Hindi': 'hi', 'Vietnamese': 'vi'}

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model():
    global xtts_model
    repo_id = "jimmyvu/xtts"
    
    logger.info("Downloading model from Hugging Face...")
    model_files = ["config.json", "vocab.json", "model.pth"]
    for filename in model_files:
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=checkpoint_dir)

    config = XttsConfig()
    config.load_json(os.path.join(checkpoint_dir, "config.json"))
    xtts_model = Xtts.init_from_config(config)

    logger.info("Loading model...")
    xtts_model.load_checkpoint(
        config, checkpoint_dir=checkpoint_dir, use_deepspeed=True
    )
    if torch.cuda.is_available():
        xtts_model.cuda()
    logger.info(f"Successfully loaded model from {checkpoint_dir}")

def generate_speech(input_text, speaker_reference_audio, enhance_speech, temperature=0.3, top_p=0.85, top_k=50, repetition_penalty=10.0, language='English', *args):
    """Process text and generate audio."""
    log_messages = ""
    if not speaker_reference_audio:
        log_messages += "Please provide at least one reference audio!\n"
        return None, log_messages

    if not xtts_model:
        load_model()

    language_code = language_dict.get(language, 'en')
    if language_code == 'vi':
        input_text = normalize_vietnamese_text(input_text)
    
    gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
        audio_path=speaker_reference_audio,
        gpt_cond_len=xtts_model.config.gpt_cond_len,
        max_ref_length=xtts_model.config.max_ref_len,
        sound_norm_refs=xtts_model.config.sound_norm_refs,
    )

    # inference
    out = xtts_model.inference(
        text=input_text,
        language=language_code,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty*1.0,
        enable_text_splitting=True,
    )
    return (24000, out["wav"]), log_messages
    

def build_gradio_ui():
    """Builds and launches the Gradio UI."""
    with gr.Blocks(title="Coqui XTTS Demo", theme="JohnSmith9982/small_and_pretty") as ui:

        gr.Markdown(
          """
          # 🐸 Coqui-XTTS Text-to-Speech Demo
          Convert text to speech with advanced voice cloning and enhancement.
          """
        )

        with gr.Tab("Text to Speech"):
          with gr.Row():
            with gr.Column():
                input_text = gr.Text(label="Enter Text Here", placeholder="Write the text you want to convert...", lines=5)
                speaker_reference_audio = gr.Audio(
                    label="Speaker reference audio:",
                    value=REFERENCE_AUDIO,
                    type="filepath",
                    editable=False,
                )
                language = gr.Dropdown(label="Target Language", choices=[
                        'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Polish', 
                        'Turkish', 'Russian', 'Dutch', 'Czech', 'Arabic', "Simplified Chinese", 
                        'Hungarian', 'Korean', 'Japanese', 'Hindi', 'Vietnamese',
                    ], value="English")
                with gr.Accordion("Advanced settings", open=False):
                    enhance_speech = gr.Checkbox(label="Enhance Reference Speech", value=False)
                    temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, value=0.3, step=0.05)
                    top_p = gr.Slider(label="Top P", minimum=0.5, maximum=1.0, value=0.85, step=0.05)
                    top_k = gr.Slider(label="Top K", minimum=0, maximum=100, value=50, step=10)
                    repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=50.0, value=10.0, step=1.0)
                    
                generate_button = gr.Button("Generate Speech")
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio")
                log_output = gr.Text(label="Log Output")

          generate_button.click(
            generate_speech,
            inputs=[input_text, speaker_reference_audio, enhance_speech, temperature, top_p, top_k, repetition_penalty, language],
            outputs=[audio_output, log_output],
          )

        with gr.Tab("Clone Your Voice"):
          with gr.Row():
            with gr.Column():
                input_text_mic = gr.Text(label="Enter Text Here", placeholder="Write the text you want to convert...", lines=5)
                mic_ref_audio = gr.Audio(label="Record Reference Audio", sources=["microphone"])
                language_mic = gr.Dropdown(label="Target Language", choices=[
                    'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Polish', 
                    'Turkish', 'Russian', 'Dutch', 'Czech', 'Arabic', "Simplified Chinese", 
                    'Hungarian', 'Korean', 'Japanese', 'Hindi', 'Vietnamese',
                  ], value="English")
                with gr.Accordion("Advanced settings", open=False):
                    enhance_speech_mic = gr.Checkbox(label="Enhance Reference Speech", value=True)
                    temperature_mic = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, value=0.3, step=0.05)
                    top_p_mic = gr.Slider(label="Top P", minimum=0.5, maximum=1.0, value=0.85, step=0.05)
                    top_k_mic = gr.Slider(label="Top K", minimum=0, maximum=100, value=50, step=10)
                    repetition_penalty_mic = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=50.0, value=10.0, step=1.0)
                generate_button_mic = gr.Button("Generate Speech")
            with gr.Column():
                audio_output_mic = gr.Audio(label="Generated Audio")
                log_output_mic = gr.Text(label="Log Output")

          
          def process_mic_and_generate(input_text_mic, mic_ref_audio, enhance_speech_mic, temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic):
              if mic_ref_audio:
                  data = str(time.time()).encode("utf-8")
                  hash = hashlib.sha1(data).hexdigest()[:10]
                  output_path = os.path.join(temp_dir, (f"mic_{hash}.wav"))

                  torch_audio = torch.from_numpy(mic_ref_audio[1].astype(float))
                  try:
                      torchaudio.save(str(output_path), torch_audio.unsqueeze(0), mic_ref_audio[0])
                      return generate_speech(input_text_mic, [Path(output_path)], enhance_speech_mic, temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic)
                  except Exception as e:
                      logger.error(f"Error saving audio file: {e}")
                      return None, f"Error saving audio file: {e}"
              else:
                  return None, "Please record an audio!"

          generate_button_mic.click(
            process_mic_and_generate,
            inputs=[input_text_mic, mic_ref_audio, enhance_speech_mic, temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic],
            outputs=[audio_output_mic, log_output_mic],
          )
        
    return ui

if __name__ == "__main__":
    ui = build_gradio_ui()
    ui.launch(debug=True)
