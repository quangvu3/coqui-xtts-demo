## Project Overview

This project is a Gradio-based web application that serves as a demonstration for the Coqui XTTS (Text-to-Speech) model. It allows users to convert text into speech using a variety of voices and languages, with a special focus on Vietnamese. The application supports advanced features like voice cloning from an audio sample and synthesizing text with multiple speakers in a single audio file.

The project has two main parts:
1.  A **Gradio Web UI** (`gradio_app.py`) for interactive use.
2.  An **OpenAI-compatible web server** (`xtts_oai_server/xtts_server.py`) that exposes a REST API for programmatic access.

The core technologies used are Python, Gradio, PyTorch, and Coqui-TTS. The application is configured to download a specific XTTS model version from Hugging Face Hub.

## Building and Running

### 1. Install Dependencies

The project's dependencies are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

This will install all the necessary Python packages, including a custom version of `coqui-xtts` from a GitHub repository.

### 2. Running the Gradio Web UI

To start the interactive Gradio web application, run the following command:

```bash
python gradio_app.py
```

This will launch a local web server, and you can access the user interface in your web browser at the URL provided in the console (usually `http://127.0.0.1:7860`).

### 3. Running the OpenAI-compatible Server

To start the web server that provides the OpenAI-compatible API, run the following command:

```bash
python xtts_oai_server/xtts_server.py
```

The server will start on port `8088` by default. You can then send requests to the API endpoints, such as `/v1/audio/speech` and `/v1/speakers`.

## Development Conventions

*   **Model Caching**: The application downloads and caches the XTTS model in the `cache` directory.
*   **Custom Speakers**: Custom speaker voices can be added to the `speakers` directory. The application will automatically detect and load them.
*   **Multi-speaker Synthesis**: The application can synthesize text with multiple speakers by using tags in the input text, like `[speaker_id] text`. For example: `[main_storyteller_1] Once upon a time... [normal_young_man_1] Hello!`.
*   **Vietnamese Text Normalization**: The application includes a utility for normalizing Vietnamese text before synthesis.
*   **Code Structure**: The project is organized into a main Gradio application, a separate server module, and a `utils` directory for shared helper functions.
