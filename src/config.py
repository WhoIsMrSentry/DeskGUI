import os
import sys
from pathlib import Path

# Define project root (parent of 'src')
SRC_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SRC_DIR.parent

# Define standard paths
CONFIG_DIR = ROOT_DIR / "config"
MODELS_DIR = ROOT_DIR / "models"
SRC_GUI_DIR = SRC_DIR / "gui"
SRC_AUDIO_DIR = SRC_DIR / "audio"
SRC_VISION_DIR = SRC_DIR / "vision"
SRC_COMMS_DIR = SRC_DIR / "comms"
SRC_UTILS_DIR = SRC_DIR / "utils"

# Config Files
PERSONALITIES_FILE = CONFIG_DIR / "personalities.json"
PRIORITY_ANIMATIONS_FILE = CONFIG_DIR / "priority_animations.json"
PRIORITY_PERSONS_FILE = CONFIG_DIR / "priority_persons.json"
SETTINGS_FILE = CONFIG_DIR / "settings.json" # Future use

# Model Files
ENCODINGS_FILE = MODELS_DIR / "encodings.pickle"
HAARCASCADE_FILE = MODELS_DIR / "haarcascade_frontalface_default.xml"
WAKE_WORD_MODEL = MODELS_DIR / "hey_sen_tree_bot.onnx"

def get_model_path(filename: str) -> str:
    """Helper to get absolute path of a model file."""
    return str(MODELS_DIR / filename)

def get_config_path(filename: str) -> str:
    """Helper to get absolute path of a config file."""
    return str(CONFIG_DIR / filename)

# Ensure src is in python path if needed (though running as module handles this)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
