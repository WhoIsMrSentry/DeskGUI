# DeskGUI

DeskGUI is a PyQt5-based desktop control and monitoring interface developed for the SentryBOT robot platform. It provides real-time video streaming, voice commands, face and object detection, robot state monitoring, and integrations with LLMs (large language models) to help you manage your robot easily.

## Features

- **Real-time Video Streaming:** Watch live camera feed from the robot.
- **Voice Commands & TTS:** Send commands via microphone and listen to spoken responses.
- **Face & Object Detection:** Advanced vision modules for detecting faces, objects, age, and emotions.
- **Bluetooth Audio Server:** Route the robot's audio I/O through your PC.
- **Robot State Monitoring:** Monitor connection status, eye color, personality, and other robot states in real time.
- **LLM / Gemini Integration:** Chat and command support using Ollama, Gemini, and other LLMs.
- **Theme Support:** Switch between dark, light, and red themes.
- **Advanced Logging and Error Handling:** Detailed logs and error panel for troubleshooting.
- **Multi-language Support:** Multiple language support and automatic language detection.
- **Hand & Finger Gesture Control:** Control the robot with camera-based hand gestures.
- **Animation Control:** Manage LED and servo animations on the robot.
- **Age & Emotion Estimation:** Approximate age and emotional expression from face images.

## Installation

### System Requirements

- **OS:** Windows 10/11, Ubuntu 20.04+, or macOS 10.15+
- **Python:** 3.8+ (3.10 recommended)
- **Microphone:** Required for STT (speech-to-text)
- **Speakers:** For TTS output
- **Camera:** Optional, local camera for testing
- **GPU:** Optional — NVIDIA GPU recommended for some vision tasks

### Setup Steps

1. Clone the repository and install Python dependencies:
   ```powershell
   # Create and activate a virtual environment (recommended)
   python -m venv venv
   .\venv\Scripts\activate

   # Install required packages
   pip install -r requirements.txt

   # Optional extra packages for image processing
   pip install mediapipe cvzone tensorflow
   ```

2. Place the required model files for vision processing:
   - `encodings.pickle`: face recognition encoding file (example provided)
   - `haarcascade_frontalface_default.xml`: OpenCV face detector
   - `hey_sen_tree_bot.onnx`: wake-word / detection model

   Also update `MODELS_DIR` in `modules/vision/__init__.py` if needed:
   ```python
   MODELS_DIR = r"C:\path\to\your\models"
   ```

3. Start the GUI:
   ```powershell
   python run_gui.py --robot-ip <ROBOT_IP_ADDRESS>
   ```
   Or start both GUI and audio server together:
   ```powershell
   python run_all.py
   ```

## Command Line Arguments

### run_gui.py

- `--robot-ip` - Robot IP address (default: 192.168.137.52)
- `--video-port` - Video stream port (default: 8000)
- `--command-port` - Command port (default: 8090)
- `--ollama-url` - Ollama API URL (default: http://localhost:11435)
- `--ollama-model` - Ollama model to use (default: SentryBOT:4b)
- `--encodings-file` - Face encodings file (default: encodings.pickle)
- `--bluetooth-server` - Bluetooth audio server IP (default: 192.168.1.100)
- `--enable-fastapi` - Enable FastAPI support
- `--retry-on-error` - Auto-restart on error
- `--log-file` - Log file (default: sentry_gui.log)
- `--debug` - Show debug information

### run_audio_server.py

- `--host` - Host to bind to (default: 0.0.0.0)
- `--tts-port` - TTS service port (default: 8095)
- `--speech-port` - Speech recognition port (default: 8096)
- `--fastapi-port` - FastAPI websocket port (default: 8098)
- `--use-fastapi` - Use FastAPI for performance
- `--device-name` - Microphone device name
- `--device-index` - Microphone device index
- `--list-devices` - List available microphones
- `--voice-idx` - TTS voice index (default: 0)
- `--auto-start-speech` - Auto-start speech recognition
- `--language` - Speech recognition language (e.g., en-US, tr-TR)
- `--test-audio` - Test audio output on startup
- `--verbose` - Verbose logging

### run_all.py

- `--robot-ip` - Robot IP address
- `--video-port` - Video stream port
- `--command-port` - Command port
- `--ollama-url` - Ollama API URL (default port 11435)
- `--encodings-file` - Face encodings file
- `--debug` - Show debug info
- `--theme` - App theme (light, dark, auto)
- `--xtts` - Start XTTS API in a separate terminal (Windows)

## TTS (Text-to-Speech) Configuration

### Piper TTS Setup

1. Download Piper TTS: https://github.com/rhasspy/piper

   Example Windows steps:
   ```powershell
   mkdir C:\Users\<USER>\piper
   cd C:\Users\<USER>\piper
   $url = "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_windows_amd64.zip"
   Invoke-WebRequest -Uri $url -OutFile "piper.zip"
   Expand-Archive -Path "piper.zip" -DestinationPath "."
   ```

2. Download required voice models:
   ```powershell
   mkdir C:\Users\<USER>\piper\tr-TR
   cd C:\Users\<USER>\piper\tr-TR
   $model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/sinem/medium/tr_TR-sinem-medium.onnx"
   $json_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/sinem/medium/tr_TR-sinem-medium.onnx.json"
   Invoke-WebRequest -Uri $model_url -OutFile "tr_TR-sinem-medium.onnx"
   Invoke-WebRequest -Uri $json_url -OutFile "tr_TR-sinem-medium.onnx.json"
   ```

3. Place models under:
   - Windows: `C:\Users\<USER>\piper\<LANG>\<MODEL>.onnx`
   - Linux: `~/piper/<LANG>/<MODEL>.onnx`

4. (Optional) Test:
   ```powershell
   cd C:\Users\<USER>\piper
   .\piper.exe --model .\tr-TR\tr_TR-sinem-medium.onnx --output_file test.wav --text "Hello, this is a robot voice."
   ```

5. In the GUI, set TTS provider to `piper`.

### XTTS (XTalker TTS) Setup

1. Create a virtual environment and install dependencies:
   ```powershell
   mkdir C:\Users\<USER>\xTTS
   cd C:\Users\<USER>\xTTS
   python -m venv tts_env
   .\tts_env\Scripts\Activate.ps1
   pip install TTS uvicorn fastapi python-multipart
   ```

2. Example `xtts.py` API server (FastAPI) is provided in the original README.

3. Start the server with Uvicorn or via a helper batch script.

4. In the GUI, set TTS provider to `xtts` and configure the reference voice file path.

## Usage

- Provide robot IP and ports via CLI arguments.
- Control video, audio, animations, and commands from the GUI.
- Configure advanced settings, API keys, and LLM options from the GUI settings.

## Dependencies

- Python 3.8+
- PyQt5
- OpenCV

## Troubleshooting

### Ollama 404 (/api\generate) Error
On some Windows setups, `os.path.join` may insert backslashes into URLs and cause requests like `http://localhost:11435/api\generate`, which results in 404.

Fixes:
1. Use `http://localhost:11435` or `http://localhost:11435/api` in the GUI Ollama URL field (no trailing backslash).
2. Ensure your code uses string concatenation for URLs or a corrected join implementation.
3. Test with PowerShell:
   ```powershell
   Invoke-RestMethod -Method Post -Uri http://localhost:11435/api/generate -Body (@{model='SentryBOT:4b'; prompt='test'; stream=$false} | ConvertTo-Json) -ContentType 'application/json'
   ```

### NumPy / onnxruntime DLL Errors
If you see `ImportError: DLL load failed while importing _framework_bindings` or similar, NumPy 2.x may be incompatible with older onnxruntime versions.

Fix:
1. `pip install 'numpy<2' --upgrade --force-reinstall`
2. `pip install --upgrade onnxruntime` (or `onnxruntime-gpu` for GPU)
3. Run `python debug_imports.py` for diagnostics.

### Empty / Very Long LLM Responses
If responses are empty, the initial request may have timed out. Increase timeout in settings. Long responses may be truncated — adjust `resp_length` and output truncation settings in `desk_gui_app.py`.

### Missing Translation Module / Errors
If translation is disabled, the input may be sent to the LLM untranslated. Ensure `modules/translate_helper.py` is present and `googletrans` (or chosen library) is installed.

## Recommended Packages (optional)

```powershell
pip install PyQt5 opencv-python-headless face_recognition numpy sounddevice pyaudio pyttsx3 gtts requests pubsub pygame onnxruntime pydub langdetect fastapi uvicorn
```

For advanced vision features:

```powershell
pip install mediapipe cvzone tensorflow keras
```

## Project Layout

- `desk_gui.py`, `run_gui.py`, `run_all.py`: main entry points and GUI launchers
- `modules/`: audio, vision, command helpers, robot data listener, and helpers
- `modules/gui/desk_gui_app.py`: central GUI application
- `modules/vision/`: vision processing (face, object, finger, age/emotion detection)
- `encodings.pickle`, `haarcascade_frontalface_default.xml`: model and helper files

## Modules and Components

DeskGUI is modular and composed of many modules. Key modules include:

- `desk_gui_app.py`: main GUI application and interface
- `audio_manager.py`: manages audio I/O and devices
- `audio_thread_manager.py`: manages audio threads
- `command_sender.py`: sends TCP commands to the robot
- `command_helpers.py`: utilities for building and handling commands
- `face_detector.py`: face detection and recognition
- `gemini_helper.py`: Google Gemini integration helper
- `motion_detector.py`: motion detection from camera frames
- `remote_video_stream.py`: receives and processes video from the robot
- `robot_data_listener.py`: listens to robot state messages
- `speech_input.py`: speech recognition handling
- `tracking.py`: position calculations for object/face tracking
- `translate_helper.py`: translation utilities
- `tts.py`: text-to-speech integration

## Vision Modules (modules/vision)

- `age_emotion.py`: age and emotion estimation from faces
- `finger_tracking.py`: hand and finger gesture recognition
- `object_detection.py`: object detection (e.g., TensorFlow-based)
- `object_tracking.py`: tracking detected objects over time

## Launchers

- `run_gui.py`: starts only the GUI
- `run_audio_server.py`: starts only the audio server
- `run_all.py`: starts GUI and audio server together

## LLM (Language Model) Integration

### Ollama

DeskGUI integrates with Ollama by default.

1. Install Ollama (Windows example):
   ```powershell
   winget install Ollama.Ollama
   ```
2. Pull a model:
   ```powershell
   ollama pull [MODEL_NAME]
   ```
3. Configure `--ollama-url` and `--ollama-model` when launching.

### Gemini AI

To use Google Gemini, obtain API credentials from Google AI Studio and configure them in the GUI settings.

## Advanced Features

### Face Recognition
Store person face encodings in `encodings.pickle` for face recognition workflows.



(End of README English translation)
