import os
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
os.makedirs(checkpoint_dir, exist_ok=True)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_AUDIO = os.path.join(APP_DIR, "samples", "nam-calm.wav")

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

def process_text_and_generate_speech(input_text, speaker_reference_audio, enhance_speech, temperature, top_p, top_k, repetition_penalty, language, *args):
    """Process text and generate audio."""
    log_messages = ""
    if not speaker_reference_audio:
        log_messages += "Please provide at least one reference audio!\n"
        return None, log_messages

    if language == 'vi':
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
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty*1.0,
        enable_text_splitting=True,
    )
    # out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
    # torchaudio.save("speech.wav", out["wav"], 24000)
    return (24000, out["wav"]), log_messages
    

def build_gradio_ui():
    """Builds and launches the Gradio UI."""
    with gr.Blocks(title="Coqui XTTS Demo", theme="soft") as ui:

        gr.Markdown(
          """
          # 🐸 Coqui-XTTS Text-to-Speech Demo
          Convert text to speech with advanced voice cloning and enhancement.
          """
        )

        with gr.Tab("Text to Speech"):
          with gr.Row():
            with gr.Column():
                input_text = gr.Text(label="Enter Text Here", placeholder="Write the text you want to convert...")
                speaker_reference_audio = gr.Audio(
                    label="Speaker reference audio:",
                    value=REFERENCE_AUDIO,
                    type="filepath",
                )
                with gr.Accordion("Advanced settings", open=False):
                    enhance_speech = gr.Checkbox(label="Enhance Reference Speech", value=False)
                    temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, value=0.3, step=0.05)
                    top_p = gr.Slider(label="Top P", minimum=0.5, maximum=1.0, value=0.85, step=0.05)
                    top_k = gr.Slider(label="Top K", minimum=0, maximum=100, value=50, step=10)
                    repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=50.0, value=10.0, step=5.0)
                    language = gr.Dropdown(label="Target Language", choices=[
                        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
                        "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi", "vi", "auto",
                    ], value="auto")
                generate_button = gr.Button("Generate Speech")
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio")
                log_output = gr.Text(label="Log Output")

          generate_button.click(
            process_text_and_generate_speech,
            inputs=[input_text, speaker_reference_audio, enhance_speech, temperature, top_p, top_k, repetition_penalty, language],
            outputs=[audio_output, log_output],
          )

        with gr.Tab("File to Speech"):
          with gr.Row():
            with gr.Column():
              file_input = gr.File(label="Text / Ebook File", file_types=["text", ".epub"])
              ref_audio_files_file = gr.Files(label="Reference Audio Files", file_types=["audio"])
              with gr.Accordion("Advanced settings", open=False):
                  speed_file = gr.Slider(label="Playback speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
                  enhance_speech_file = gr.Checkbox(label="Enhance Reference Speech", value=False)
                  temperature_file = gr.Slider(label="Temperature", minimum=0.5, maximum=1.0, value=0.75, step=0.05)
                  top_p_file = gr.Slider(label="Top P", minimum=0.5, maximum=1.0, value=0.85, step=0.05)
                  top_k_file = gr.Slider(label="Top K", minimum=0, maximum=100, value=50, step=10)
                  repetition_penalty_file = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=10.0, value=5.0, step=0.5)
                  language_file = gr.Dropdown(label="Target Language", choices=[
                      "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
                      "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi", "vi", "auto",
                  ], value="auto")
              generate_button_file = gr.Button("Generate Speech from File")
            with gr.Column():
                audio_output_file = gr.Audio(label="Generated Audio")
                log_output_file = gr.Text(label="Log Output")

          def process_file_and_generate(file_input, ref_audio_files_file, speed_file, enhance_speech_file, temperature_file, top_p_file, top_k_file, repetition_penalty_file, language_file):
              if file_input:
                  file_extension = Path(file_input.name).suffix
                  if file_extension == '.epub':
                      input_text = extract_text_from_epub(file_input.name)
                  elif file_extension == '.txt':
                      input_text = text_from_file(file_input.name)
                  else:
                      return None, "Unsupported file format, it needs to be either .epub or .txt"

                  return process_text_and_generate(input_text, ref_audio_files_file, speed_file, enhance_speech_file,
                                                   temperature_file, top_p_file, top_k_file, repetition_penalty_file, language_file)
              else:
                  return None, "Please provide an .epub or .txt file!"

          generate_button_file.click(
            process_file_and_generate,
            inputs=[file_input, ref_audio_files_file, speed_file, enhance_speech_file, temperature_file, top_p_file, top_k_file, repetition_penalty_file, language_file],
            outputs=[audio_output_file, log_output_file],
          )

        with gr.Tab("Clone With Microfone"):
          with gr.Row():
            with gr.Column():
              input_text_mic = gr.Text(label="Enter Text Here", placeholder="Write the text you want to convert...")
              mic_ref_audio = gr.Audio(label="Record Reference Audio", sources=["microphone"])

              with gr.Accordion("Advanced settings", open=False):
                  speed_mic = gr.Slider(label="Playback speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
                  enhance_speech_mic = gr.Checkbox(label="Enhance Reference Speech", value=True)
                  temperature_mic = gr.Slider(label="Temperature", minimum=0.5, maximum=1.0, value=0.75, step=0.05)
                  top_p_mic = gr.Slider(label="Top P", minimum=0.5, maximum=1.0, value=0.85, step=0.05)
                  top_k_mic = gr.Slider(label="Top K", minimum=0, maximum=100, value=50, step=10)
                  repetition_penalty_mic = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=10.0, value=5.0, step=0.5)
                  language_mic = gr.Dropdown(label="Target Language", choices=[
                      "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
                      "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi", "vi", "auto",
                  ], value="auto")
              generate_button_mic = gr.Button("Generate Speech")
            with gr.Column():
                audio_output_mic = gr.Audio(label="Generated Audio")
                log_output_mic = gr.Text(label="Log Output")

          import hashlib

          def process_mic_and_generate(input_text_mic, mic_ref_audio, speed_mic, enhance_speech_mic, temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic):
              if mic_ref_audio:
                  data = str(time.time()).encode("utf-8")
                  hash = hashlib.sha1(data).hexdigest()[:10]
                  output_path = temp_dir / (f"mic_{hash}.wav")

                  torch_audio = torch.from_numpy(mic_ref_audio[1].astype(float))
                  try:
                      torchaudio.save(str(output_path), torch_audio.unsqueeze(0), mic_ref_audio[0])
                      return process_text_and_generate(input_text_mic, [Path(output_path)], speed_mic, enhance_speech_mic, temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic)
                  except Exception as e:
                      logger.error(f"Error saving audio file: {e}")
                      return None, f"Error saving audio file: {e}"
              else:
                  return None, "Please record an audio!"

          generate_button_mic.click(
            process_mic_and_generate,
            inputs=[input_text_mic, mic_ref_audio, speed_mic, enhance_speech_mic, temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic],
            outputs=[audio_output_mic, log_output_mic],
          )
        
    return ui

if __name__ == "__main__":
    load_model()
    ui = build_gradio_ui()
    ui.launch(debug=True)
