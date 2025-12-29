import queue
import sys
import src.config as config
import cv2
import requests
import os
import threading
import socket
import tempfile
import io
import random
import sounddevice as sd
sd.default.device = 1  # veya uygun mikrofon ID'si
import struct
import traceback
import math
import gtts
import pyttsx3
import re
import pickle
from pydub import AudioSegment # Ses dosyalarını birleştirmek için
import numpy as np
import subprocess
import langdetect
import shutil
import json
import pygame
import pyaudio
import sounddevice as sd
import numpy as np    
try:
    import onnxruntime as ort  # Wake word veya diğer ONNX modelleri için
    ONNXRUNTIME_AVAILABLE = True
except Exception as _ort_err:  # NumPy 2 uyumsuzluğu veya binary hatası
    ort = None  # type: ignore
    ONNXRUNTIME_AVAILABLE = False
    print(f"UYARI: onnxruntime import edilemedi: {_ort_err}\n  -> Çözüm: 'pip install \"numpy<2\" --force-reinstall' veya 'pip install --upgrade onnxruntime' (NumPy 2 uyumlu sürüm). Wake word devre dışı.")
try:
    from cvzone.HandTrackingModule import HandDetector
except ImportError:
    HandDetector = None
    print("Warning: cvzone or mediapipe not found. Hand tracking disabled.")
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Q_ARG, QMetaObject  # <-- Add this import
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QSize, pyqtSlot
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QGroupBox, QFormLayout, QLineEdit, QTextEdit,
    QSizePolicy, QTabWidget, QGridLayout, QRadioButton, 
    QCheckBox, QComboBox, QSlider, QMessageBox, QSpinBox, QTableWidget, QTableWidgetItem,QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox, QLineEdit, QComboBox, QMessageBox
)
from pubsub import pub
from langdetect import detect, LangDetectException
import time

# --- Monkey Patch for dlib Unicode Path Issue ---
# Fixes RuntimeError: Unable to open ... on Windows paths with non-ASCII chars
try:
    import face_recognition_models
    import shutil
    import tempfile
    import os
    
    if os.name == 'nt':
        _temp_dir = tempfile.gettempdir()
        
        def _patch_model(func_name, filename):
            try:
                # Get usage of original function
                orig_func = getattr(face_recognition_models, func_name)
                _original_path = orig_func()
                _safe_path = os.path.join(_temp_dir, filename)
                
                # Check copying needs
                if os.path.exists(_original_path):
                    if not os.path.exists(_safe_path) or os.path.getsize(_safe_path) != os.path.getsize(_original_path):
                        print(f"System: Patching {filename} to safe path...")
                        shutil.copy2(_original_path, _safe_path)
                    
                    # Apply monkey patch
                    setattr(face_recognition_models, func_name, lambda: _safe_path)
                    # print(f"System: Patched {func_name}.")
            except Exception as e:
                print(f"Warning: Failed to patch {filename}: {e}")

        # Patch all 4 models used by face_recognition
        _patch_model('pose_predictor_model_location', "shape_predictor_68_face_landmarks.dat")
        _patch_model('pose_predictor_five_point_model_location', "shape_predictor_5_face_landmarks.dat")
        _patch_model('cnn_face_detector_model_location', "mmod_human_face_detector.dat")
        _patch_model('face_recognition_model_location', "dlib_face_recognition_resnet_model_v1.dat")
        print("System: dlib model info patched for unicode paths.")

except Exception as _patch_err:
    print(f"Warning: Failed to apply dlib Unicode patch: {_patch_err}")

import face_recognition
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

try:
    from src.vision.remote.train_model import TrainModel
    TRAIN_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import TrainModel module: {e}")
    TRAIN_MODEL_AVAILABLE = False

# Yeni Görüntü İşleme Modülleri
try:
    from src.vision.finger_tracking import FingerTracking
    from src.vision.age_emotion import AgeEmotionDetector
    from src.vision.object_detection import ObjectDetector
    from src.vision.object_tracking import ObjectTracker
    VISION_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"UYARI: Görüntü işleme modülleri yüklenemedi: {e}")
    VISION_MODULES_AVAILABLE = False
    # Modül sınıflarını None olarak tanımla ki referans hataları olmasın
    FingerTracking = None
    AgeEmotionDetector = None
    ObjectDetector = None
    ObjectTracker = None    

try:
    from src.utils.translate_helper import TranslateHelper
    TRANSLATE_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Uyarı:Çeviri Modülü Yüklenemedi: {e}")  
    VISION_MODULES_AVAILABLE= False
    TranslateHelper = None

try:
    from src.utils.gemini_helper import GeminiHelper
    GEMINI_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Uyarı: Gemini Modülü Yüklenemedi: {e}")
    GEMINI_MODULE_AVAILABLE = False
    GeminiHelper = None



# Modül importları
from src.audio.speech_input import SpeechInput
from src.audio.audio_manager import AudioManager
from src.audio.audio_thread_manager import AudioThreadManager
from src.comms.remote_video_stream import RemoteVideoStream
from src.comms.command_sender import CommandSender
from src.vision.face_detector import FaceDetector
from src.vision.motion_detector import MotionDetector
from src.vision.tracking import Tracking
from src.comms.robot_data_listener import RobotDataListener

class DeskGUI(QtWidgets.QMainWindow):
    # UI güncellemeleri için PyQt sinyalleri tanımlayalım
    update_output_signal = QtCore.pyqtSignal(str, bool)
    clear_output_signal = QtCore.pyqtSignal()
    set_input_text_signal = QtCore.pyqtSignal(str)
    update_thinking_signal = QtCore.pyqtSignal()
    request_completed_signal = QtCore.pyqtSignal()
    update_status_signal = QtCore.pyqtSignal(str, str)
    update_audio_status_signal = QtCore.pyqtSignal(str, str)
    update_mic_level_signal = QtCore.pyqtSignal(float)
    speech_status_signal = QtCore.pyqtSignal(bool, str)

    # --- YENİ SİNYALLER (Eğitim Thread'i için) ---
    log_signal = QtCore.pyqtSignal(str) # Thread'den log mesajı göndermek için
    update_priority_combo_signal = QtCore.pyqtSignal() # Priority combo box'ı güncellemek için
    training_complete_signal = QtCore.pyqtSignal(bool, str, str) # Eğitim tamamlandı (başarı durumu, mesaj, backup_dosyası
    # Thread güvenliği için yeni sinyaller
    show_error_message_signal = QtCore.pyqtSignal(str, str)

    # --- NEW Signals for Robot Data Listener ---
    update_robot_status_ui = QtCore.pyqtSignal(dict) # To update UI elements with robot status
    update_log_from_robot = QtCore.pyqtSignal(str)  # To add robot logs to GUI log

    # --- Theme Switcher Methods ---
    def create_theme_switcher(self):
        """Sağ üst köşe için üç renkli daireden oluşan tema switcher widget'ı oluşturur."""
        from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton
        from PyQt5.QtGui import QColor, QPainter, QBrush
        from PyQt5.QtCore import QSize

        class ColorCircleButton(QPushButton):
            def __init__(self, color, theme_name, parent=None):
                super().__init__(parent)
                self.color = QColor(color)
                self.theme_name = theme_name
                self.setCheckable(True)
                self.setFixedSize(32, 32)
                self.setStyleSheet("border: none;")

            def paintEvent(self, event):
                painter = QPainter(self)
                painter.setRenderHint(QPainter.Antialiasing)
                rect = self.rect().adjusted(4, 4, -4, -4)
                brush = QBrush(self.color)
                painter.setBrush(brush)
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawEllipse(rect)
                if self.isChecked():
                    painter.setPen(QColor('#FFD700'))  # Altın sarısı kenar
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawEllipse(rect.adjusted(-2, -2, 2, 2))
                painter.end()

        switcher_widget = QWidget()
        switcher_layout = QHBoxLayout()
        switcher_layout.setContentsMargins(0, 0, 0, 0)
        switcher_layout.setSpacing(8)
        switcher_widget.setLayout(switcher_layout)

        self.theme_buttons = {}
        theme_defs = [
            ("#88001b", "red"),
            ("#222", "dark"),
            ("#fff", "light")
        ]
        for color, theme in theme_defs:
            btn = ColorCircleButton(color, theme)
            btn.clicked.connect(lambda checked, t=theme: self.on_theme_circle_clicked(t))
            switcher_layout.addWidget(btn)
            self.theme_buttons[theme] = btn

        # Başlangıçta seçili temayı işaretle
        self.update_theme_switcher_highlight(getattr(self, 'theme', 'red'))
        return switcher_widget

    def update_theme_switcher_highlight(self, theme):
        """Seçili temanın dairesini vurgula."""
        for t, btn in self.theme_buttons.items():
            btn.setChecked(t == theme)

    def on_theme_circle_clicked(self, theme):
        self.theme = theme
        self.apply_theme(theme)
        self.update_theme_switcher_highlight(theme)

    def add_theme_switcher_to_gui(self):
        """Tema switcher widget'ını sağ üst köşeye ekler."""
        switcher = self.create_theme_switcher()
        # Ana layout'un sağ üstüne ekle
        # Sağ panelin en üstüne ekleyeceğiz
        if hasattr(self, 'main_h_layout'):
            # Sağ paneli bul
            right_widget = self.main_h_layout.itemAt(1).widget()
            if right_widget:
                right_layout = right_widget.layout()
                right_layout.insertWidget(0, switcher, 0, QtCore.Qt.AlignRight)
        self.theme_switcher_widget = switcher
    @QtCore.pyqtSlot(str, str)
    def _show_warning_message_box(self, title, message):
        """Shows a warning message box (thread-safe)."""
        QtWidgets.QMessageBox.warning(self, title, message)

    @QtCore.pyqtSlot(str, str)
    def _show_critical_message_box(self, title, message):
        """Shows a critical message box (thread-safe)."""
        QtWidgets.QMessageBox.critical(self, title, message)

    @QtCore.pyqtSlot(str, str)
    def _show_info_message_box(self, title, message):
        """Shows an information message box (thread-safe)."""
        QtWidgets.QMessageBox.information(self, title, message)

    def apply_dark_red_theme(self):
        """Koyu kırmızı tema uygula (örn. #88001b)."""
        dark_red = "#88001b"
        dark_red2 = "#88001b"
        text_color = "#f5f5f5"
        accent = "#b71c1c"
        # Uygulama genelinde stil uygula
        style = f'''
            QWidget {{
                background-color: {dark_red};
                color: {text_color};
            }}
            QGroupBox {{
                border: 2px solid {accent};
                border-radius: 8px;
                margin-top: 10px;
                background-color: {dark_red2};
            }}
            QGroupBox:title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: {text_color};
            }}
            QPushButton {{
                background-color: {accent};
                color: {text_color};
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #a31515;
            }}
            QLineEdit, QTextEdit, QComboBox, QSpinBox {{
                background-color: #2d0a13;
                color: {text_color};
                border: 1px solid {accent};
                border-radius: 4px;
            }}
            QLabel {{
                color: {text_color};
            }}
        '''
        QtWidgets.QApplication.instance().setStyleSheet(style)

    def apply_theme(self, theme):
        """Tema seçimini uygula: 'light', 'dark', 'auto', 'red'"""
        if theme == 'dark':
            QtWidgets.QApplication.instance().setStyle('Fusion')
            dark_palette = QtGui.QPalette()
            dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
            dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
            dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
            dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
            dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
            dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
            dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
            dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
            dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
            dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
            dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
            dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
            dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
            QtWidgets.QApplication.instance().setPalette(dark_palette)
            QtWidgets.QApplication.instance().setStyleSheet("")
        elif theme == 'light':
            QtWidgets.QApplication.instance().setStyle('Fusion')
            QtWidgets.QApplication.instance().setPalette(QtGui.QPalette())
            QtWidgets.QApplication.instance().setStyleSheet("")
        elif theme == 'red':
            # Tema seçimini uygula
            self.apply_dark_red_theme()
        elif theme == 'auto':
            # Basit: sistem teması veya varsayılanı uygula
            QtWidgets.QApplication.instance().setStyle('Fusion')
            QtWidgets.QApplication.instance().setPalette(QtGui.QPalette())
            QtWidgets.QApplication.instance().setStyleSheet("")
        else:
            self.apply_dark_red_theme()

    # show_gemini_settings_menu metodunu tamamen değiştiriyoruz
    def show_gemini_settings_menu(self):
        """Gemini ayarları için bir PyQt5 ayar menüsü açar."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Gemini API Ayarları")
        layout = QFormLayout(dlg)

        # API Anahtarı Girişi
        # Dialog alanlarını self'te tutarak değerlerin kalmasını sağlayabiliriz
        if not hasattr(self, '_gemini_api_key_input'):
            self._gemini_api_key_input = QLineEdit(self)
            self._gemini_api_key_input.setPlaceholderText("Gemini API Anahtarınızı buraya girin")
            self._gemini_api_key_input.setEchoMode(QLineEdit.Password)
        # Mevcut değeri göster
        self._gemini_api_key_input.setText(getattr(self, 'gemini_api_key', ''))
        layout.addRow("API Anahtarı:", self._gemini_api_key_input)

        # Gemini Model Seçimi
        if not hasattr(self, '_gemini_model_combo'):
            self._gemini_model_combo = QComboBox()
            # API'den çekmek ideal ama sabit liste de kullanılabilir.
            # Genellikle kullanılan ve GUI'de sunulabilecek modeller:
            gemini_models_available = [
                "gemini-2.5-flash-preview-04-17",
                "gemini-2.5-pro-preview-05-06",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                # Daha fazla model eklenebilir. Model isimlerinin API'deki tam isimlerle eşleştiğinden emin olun.
                # Örn: 'gemini-1.5-flash-preview-04-17' gibi isimler yerine 'gemini-1.5-flash-latest' daha güncel bir takma ad olabilir.
                # GUI'deki listeden seçilen değeri API'ye göndereceğiz.
            ]
            self._gemini_model_combo.addItems(gemini_models_available)
        # Mevcut modeli seç
        current_gem_model = getattr(self, 'gemini_model_name', 'gemini-1.5-pro-latest')
        index = self._gemini_model_combo.findText(current_gem_model)
        if index != -1:
            self._gemini_model_combo.setCurrentIndex(index)
        else:
            # Eğer kayıtlı model listede yoksa, listeye ekleyip seçin (veya logla)
            self.log(f"Kayıtlı Gemini modeli '{current_gem_model}' listede bulunamadı. Listeye ekleniyor.")
            self._gemini_model_combo.insertItem(0, current_gem_model)
            self._gemini_model_combo.setCurrentIndex(0)


        layout.addRow("Gemini Model:", self._gemini_model_combo)

        # Diğer parametreler (Temperature, Top-K, Top-P, System Instruction, Safety Settings)
        # Bu widget'ları da self'te tutabiliriz veya her seferinde oluşturabiliriz.
        # Değerlerin kalması için self'te tutmak daha iyi:
        if not hasattr(self, '_gemini_temp_spin'): self._gemini_temp_spin = QDoubleSpinBox(); self._gemini_temp_spin.setRange(0.0, 2.0); self._gemini_temp_spin.setSingleStep(0.01); self._gemini_temp_spin.setValue(getattr(self, 'gemini_temperature', 1.0))
        layout.addRow("Temperature:", self._gemini_temp_spin)

        if not hasattr(self, '_gemini_topk_spin'): self._gemini_topk_spin = QSpinBox(); self._gemini_topk_spin.setRange(1, 100); self._gemini_topk_spin.setValue(getattr(self, 'gemini_top_k', 32)) # Gemini için varsayılanları kontrol edin
        layout.addRow("Top-K:", self._gemini_topk_spin)

        if not hasattr(self, '_gemini_topp_spin'): self._gemini_topp_spin = QDoubleSpinBox(); self._gemini_topp_spin.setRange(0.0, 1.0); self._gemini_topp_spin.setSingleStep(0.01); self._gemini_topp_spin.setValue(getattr(self, 'gemini_top_p', 1.0))
        layout.addRow("Top-P:", self._gemini_topp_spin)

        if not hasattr(self, '_gemini_instr_edit'): self._gemini_instr_edit = QTextEdit(); self._gemini_instr_edit.setPlaceholderText("İsteğe bağlı sistem talimatı"); self._gemini_instr_edit.setPlainText(getattr(self, 'gemini_system_instruction', ''))
        layout.addRow("System Instruction:", self._gemini_instr_edit)

        if not hasattr(self, '_gemini_safety_edit'): self._gemini_safety_edit = QTextEdit(); self._gemini_safety_edit.setPlaceholderText("Örn: [{'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_ONLY_HIGH'}]");
        current_safety_settings = getattr(self, 'gemini_safety_settings', None)
        if current_safety_settings is None:
            self._gemini_safety_edit.setPlainText("")
        elif isinstance(current_safety_settings, (list, dict)):
             try:
                self._gemini_safety_edit.setPlainText(json.dumps(current_safety_settings, indent=2, ensure_ascii=False))
             except Exception: # JSON serileştirme hatası durumunda
                self._gemini_safety_edit.setPlainText(str(current_safety_settings))
        else: # Beklenmeyen tip
            self._gemini_safety_edit.setPlainText(str(current_safety_settings))

        layout.addRow("Safety Settings (JSON):", self._gemini_safety_edit)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addRow(btns)

        def on_accept():
            new_api_key = self._gemini_api_key_input.text().strip()
            new_model_name = self._gemini_model_combo.currentText()
            new_temperature = self._gemini_temp_spin.value()
            new_top_k = self._gemini_topk_spin.value()
            new_top_p = self._gemini_topp_spin.value()
            new_system_instruction = self._gemini_instr_edit.toPlainText().strip()

            # API anahtarını hemen kaydet
            self.gemini_api_key = new_api_key

            # API Anahtarı kontrolü - Anahtar boşsa uyarı verip çık
            if not self.gemini_api_key:
                QMessageBox.warning(dlg, "Eksik API Anahtarı", "Lütfen Gemini API anahtarınızı girin.")
                return # İşlemi durdur

            # Safety settings JSON parse et
            safety_json_text = self._gemini_safety_edit.toPlainText().strip()
            parsed_safety_settings = None
            if safety_json_text:
                try:
                    parsed_safety_settings = json.loads(safety_json_text)
                    # JSON formatı geçerli ama API'nin beklediği yapıya uymayabilir.
                    # Genai library'sinin bunu handle etmesini umuyoruz veya daha detaylı validation eklemeliyiz.
                    # Şu an sadece JSON geçerliliğini kontrol ediyoruz.
                except json.JSONDecodeError as e:
                    QMessageBox.warning(dlg, "JSON Hatası", f"Safety ayarları geçerli bir JSON formatında olmalı: {e}")
                    return # İşlemi durdur
                except Exception as e: # Diğer olası hatalar (örn: tip dönüşümü)
                     QMessageBox.warning(dlg, "Safety Settings Hatası", f"Safety ayarları işlenirken hata: {e}")
                     return # İşlemi durdur


            # Tüm değerleri self'e kaydet
            self.gemini_model_name = new_model_name
            self.gemini_temperature = new_temperature
            self.gemini_top_k = new_top_k
            self.gemini_top_p = new_top_p
            self.gemini_system_instruction = new_system_instruction
            self.gemini_safety_settings = parsed_safety_settings # Parsed JSON objesi veya None

            # GeminiHelper instance'ını yeni ayarlarla yeniden başlat/güncelle
            if GEMINI_MODULE_AVAILABLE:
                try:
                    # API Anahtarı ve Model ile yeni instance oluştur
                    self.gemini_helper_instance = GeminiHelper(api_key=self.gemini_api_key, model=self.gemini_model_name)
                    # Diğer parametreleri set_parameters ile uygula
                    self.gemini_helper_instance.set_parameters(
                        temperature=self.gemini_temperature,
                        top_k=self.gemini_top_k,
                        top_p=self.gemini_top_p,
                        safety_settings=self.gemini_safety_settings, # Bu dict/list olmalı
                        system_instruction=self.gemini_system_instruction if self.gemini_system_instruction else None
                    )
                    self.log(f"GeminiHelper '{self.gemini_model_name}' ile güncellendi/oluşturuldu.")
                    self.log(f"Ayarlar: Temp={self.gemini_temperature}, Top-K={self.gemini_top_k}, Top-P={self.gemini_top_p}, SysInstr set: {bool(self.gemini_system_instruction)}, Safety set: {bool(self.gemini_safety_settings)}")
                    dlg.accept() # Dialog'u kapat

                except ValueError as ve: # GeminiHelper __init__ içinden gelen API anahtarı hatası vb.
                    self.log(f"Gemini başlatılamadı: {ve}")
                    QtWidgets.QMessageBox.critical(dlg, "Gemini Başlatma Hatası", f"Gemini başlatılamadı. API anahtarınızı veya model adını kontrol edin:\n{ve}")
                    self.gemini_helper_instance = None # Hata durumunda instance'ı temizle
                    self.gemini_api_key = None # Geçersiz anahtarı temizle ki tekrar girilsin
                except Exception as e:
                    self.log(f"GeminiHelper başlatılırken beklenmedik bir hata oluştu: {e}")
                    self.log(traceback.format_exc())
                    QtWidgets.QMessageBox.critical(dlg, "Gemini Hatası", f"GeminiHelper başlatılırken beklenmedik bir hata oluştu:\n{e}")
                    self.gemini_helper_instance = None # Hata durumunda instance'ı temizle
            else:
                QtWidgets.QMessageBox.critical(dlg, "Modül Hatası", "GeminiHelper modülü yüklenemedi.")
                # Modül yoksa ve kullanıcı kaydetmeye çalışırsa, yine de GeminiHelper instance'ı None kalmalı.

        btns.accepted.connect(on_accept)
        btns.rejected.connect(dlg.reject)
        dlg.exec_() # Dialog'u modal olarak göster

    def __init__(self, **kwargs):
        super(DeskGUI, self).__init__()
        self.apply_dark_red_theme()
        self.setWindowTitle("Robot Control GUI v2")
        self.resize(1500, 850)  # Ekranına göre ayarla

        # --- Temel Konfigürasyon ---
        self.robot_ip = kwargs.get('robot_ip', '192.168.137.52')
        self.video_port = kwargs.get('video_port', 8000)
        self.command_port = kwargs.get('command_port', 8090)
        self.gui_listen_port = kwargs.get('gui_listen_port', 8091)
        # Yeni varsayılan Ollama portu 11435
        self.ollama_url = kwargs.get('ollama_url', 'http://localhost:11435/api')
        self.ollama_model = kwargs.get('ollama_model', 'SentryBOT:4b')
        self.encodings_file = kwargs.get('encodings_file', str(config.ENCODINGS_FILE))
        self.bluetooth_server = kwargs.get('bluetooth_server', '192.168.1.100')
        self.debug_mode = kwargs.get('debug', False)
        self.tts_service = kwargs.get('tts_service', 'piper')
        self.show_error_message_signal.connect(self._show_critical_message_box)

        # --- LLM & Gemini ---
        self.current_llm_service = 'ollama'
        self.gemini_api_key = kwargs.get('gemini_api_key', os.getenv('GEMINI_API_KEY'))
        self.gemini_model_name = kwargs.get('gemini_model', 'gemini-1.5-pro-latest')
        self.gemini_helper_instance = None
        self.gemini_temperature = 1.0
        self.gemini_top_k = 32
        self.gemini_top_p = 1.0
        self.gemini_system_instruction = ''
        self.gemini_safety_settings = None

        # --- Wake Word / STT Durumları ---
        self.speech_triggered_by_wake_word = False
        self.wake_word_enabled_by_checkbox = False

        # --- Robot Durum Değişkenleri ---
        self.robot_state = "Unknown"
        self.robot_eye_color = "Unknown"
        self.robot_personality = "Unknown"
        self.robot_connected_status = False

        # --- Diğer Durum Değişkenleri ---
        self.mic_active = False
        self.speaking_active = False
        self.speech_active = False
        self.using_bluetooth_audio = False
        self.mic_level = 0
        self.is_processing_request = False
        self.llm_response_pending_tts_completion = False  # <<< YENİ DEĞİŞKEN
        self.last_llm_response_text_for_tts = ""  # <<< YENİ DEĞİŞKEN
        self.request_timer = None
        self.request_timeout_seconds = 120
        self.last_detected_names = []
        self.last_speech_text = ""
        self.tts_speed = 1.0
        self.p_audio = None
        self.audio_animation_timer = None
        self.use_speech_for_llm = True
        self.fastapi_server_active = False
        self.xtts_api_url = kwargs.get('xtts_api_url', "http://localhost:5002/synthesize")
        self.xtts_speaker_wav = kwargs.get('xtts_speaker_wav', "C:/Users/emirh/xTTS/test2.wav")  # Mutlak yol örneği
        self.flip_horizontal = False
        self.flip_vertical = False
        self.last_face_info = None
        self.last_announced_persons = set()
        self.last_face_info_time = time.time()
        self.stream_connected = False
        self.processing_mode = 'none'
        self.tracking_enabled = False
        self.use_auto_language = True
        self.current_tts_lang = 'tr-TR'
        self.video_stream = None # <<< Video stream'i başta None olarak tanımla
        self.personalities_loaded = False
        self.face_tracker = None # Yüz takibi için ayrı tracker
        # DeskGUI.__init__ içine ekleyin
        self.face_target_locked = False # Şu anda bir yüze kilitli mi?
        self.audio_manager = AudioManager(self.bluetooth_server)
        self.audio_manager.tts_service = self.tts_service
        # self.audio_manager.start_wake_word_listener()
        self.face_tracker_consecutive_failures = 0 # Tracker'ın art arda kaç frame başarısız olduğu
        self.face_tracker_failure_threshold = 10  # Tracker'ın kaç başarısızlığa tolerans göstereceği (ayarlayabilirsiniz)
        self.last_face_detection_time = 0.0 # Detector'ın en son NE ZAMAN yüz gördüğü (kilit bırakma için)
        self.face_detection_timeout = 2.0 # Detector kaç saniye yüz görmezse kilit bırakılır (ayarlayabilirsiniz)

        # --- ANA LAYOUT ve WIDGET'LAR ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # --- Tema Switcher (Sağ üst köşe) ---
        self.theme_switcher_widget = QWidget()
        theme_layout = QHBoxLayout()
        theme_layout.setContentsMargins(0, 0, 0, 0)
        theme_layout.setSpacing(8)
        self.theme_buttons = {}
        theme_defs = [
            ("red", "#b71c1c"),
            ("dark", "#222"),
            ("light", "#fff")
        ]
        for theme_name, color in theme_defs:
            btn = QPushButton()
            btn.setFixedSize(28, 28)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(f"border-radius:14px; background:{color}; border:2px solid #888;")
            btn.clicked.connect(lambda checked, t=theme_name: self.set_theme_from_button(t))
            theme_layout.addWidget(btn)
            self.theme_buttons[theme_name] = btn
        self.theme_switcher_widget.setLayout(theme_layout)
        # Sağ üst köşeye yerleştirmek için ana layout'a eklenecek

        # --- Widget'ları Oluştur (Layout'a eklemeden ÖNCE) ---

        # Log Alanı (Erken oluştur)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # log() metodu artık kullanılabilir
        self.log(f"GUI Init: Starting...")
        self.log("AudioManager instance created.")

        # TTS Motoru (Control Panel'den ÖNCE)
        self.tts_engine = None
        self.tts_voices = {}
        self.current_tts_voice = None
        self.tts_engine_type = "piper"
        self.initialize_tts_engine() # TTS'i başlat (tts_voices'i doldurur)

        # Kontrol Paneli
        self.create_control_panel()
        self.control_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # Robot Durum Paneli
        self.robot_status_panel = QGroupBox("Robot Status")
        robot_status_layout = QFormLayout()
        self.robot_connection_label = QLabel("Disconnected")
        self.robot_connection_label.setStyleSheet("color: red; font-weight: bold;")
        self.robot_state_label = QLabel("Unknown")
        self.robot_eye_label = QLabel("Unknown")
        self.robot_personality_label = QLabel("Unknown")
        robot_status_layout.addRow("Connection:", self.robot_connection_label)
        robot_status_layout.addRow("State:", self.robot_state_label)
        robot_status_layout.addRow("Eye Color:", self.robot_eye_label)
        robot_status_layout.addRow("Personality:", self.robot_personality_label)
        self.robot_status_panel.setLayout(robot_status_layout)
        self.robot_status_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # G/Ç Paneli (Widget'ları ve GroupBox'ı oluştur) <<< BURADA OLUŞTURULUYOR
        self.io_input = QLineEdit()
        self.io_input.setPlaceholderText("Enter text command or speak...")
        self.io_output = QTextEdit()
        self.io_output.setReadOnly(True)
        font = QFont(); font.setPointSize(12); font.setWeight(QFont.Bold)
        self.io_output.setFont(font)
        self.io_output.setAcceptRichText(True)
        self.io_output.document().setDefaultStyleSheet("""
            body { font-size: 12pt; } b, strong { font-weight: bold; }
            .important { color: #aa0000; font-weight: bold; }
            .highlight { background-color: #ffffcc; }""")
        self.io_panel = QGroupBox("I/O") # <<< self.io_panel burada tanımlanıyor
        io_layout = QVBoxLayout()
        io_layout.addWidget(QLabel("Input:"))
        io_layout.addWidget(self.io_input)
        io_layout.addWidget(QLabel("Output:"))
        io_layout.addWidget(self.io_output)
        self.io_panel.setLayout(io_layout)
        self.io_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Video Görüntüleme Alanı
        self.video_label = QLabel("Video Stream Disconnected")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: gray; font-size: 16pt;")

        # --- Layout Oluşturma ve Yerleştirme ---
        self.main_h_layout = QHBoxLayout(self.central_widget)

        # Sol Dikey Layout
        left_v_layout = QVBoxLayout()
        left_v_layout.addWidget(self.robot_status_panel)  # Robot Status kısmını GUI Status'un üstüne taşıyoruz
        left_v_layout.addWidget(self.control_panel)
        left_v_layout.addStretch(1)  # Boşluğu aşağı itsin
        
        left_widget = QWidget()
        left_widget.setLayout(left_v_layout)

        # Sağ Dikey Layout
        right_v_layout = QVBoxLayout()
        right_v_layout.addWidget(self.video_label, 3)  # Video payı
        right_v_layout.addWidget(self.log_text, 1)     # Log payı

        right_widget = QWidget()
        right_widget.setLayout(right_v_layout)

        # I/O Panelini En Sağa Taşı
        io_layout = QVBoxLayout()
        io_layout.addWidget(self.io_panel)

        io_widget = QWidget()
        io_widget.setLayout(io_layout)

        # Ana Layout'a Ekleme
        self.main_h_layout.addWidget(left_widget, 1)
        self.main_h_layout.addWidget(right_widget, 3)
        self.main_h_layout.addWidget(io_widget, 1)  # I/O kısmını en sağa ekle

        # --- Sağ üst köşe için tema switcher'ı ekle ---
        self.add_theme_switcher_to_gui()

        # --- Thread Yönetimi ---
        self.audio_thread_manager = AudioThreadManager(max_workers=2)
        cpu_count = multiprocessing.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=cpu_count)
        self.log(f"Thread pool created: {cpu_count} cores")

        # --- Diğer Başlatmalar ---
        self.command_sender = CommandSender(self.robot_ip, self.command_port)
        self.face_detector = FaceDetector(self.encodings_file)
        self.motion_detector = MotionDetector()
        self.tracking = Tracking(self.command_sender)

        # Yeni Görüntü İşleme Modüllerini Başlat
        if VISION_MODULES_AVAILABLE:
            self.finger_tracker = FingerTracking(self.command_sender)
            self.age_emotion_detector = AgeEmotionDetector(self.command_sender)
            self.object_detector = ObjectDetector(self.command_sender)
            # ObjectTracker'ı burada başlatmıyoruz, tracking checkbox ile yönetilecek
            self.object_tracker = ObjectTracker(self.command_sender) # Instance oluştur, ama başlatma
        else:
            self.finger_tracker = None
            self.age_emotion_detector = None
            self.object_detector = None
            self.object_tracker = None
        # self.audio_manager = AudioManager(self.bluetooth_server)
        # Face worker optimizasyonu
        # face_workers = max(1, min(2, cpu_count - 1)) # Eski sınırlayıcı kod
        face_workers = 4 # <<< İstediğiniz worker sayısını buraya yazın (örn: 4)
        # CPU sayısını aşmamaya dikkat edin veya dinamik bir değer kullanın:
        # face_workers = max(1, min(4, cpu_count - 1)) # Örnek: En fazla 4, en az 1
        self.face_detector.executor = ThreadPoolExecutor(max_workers=face_workers)
        self.log(f"Face recognition workers: {face_workers}")
        
        # --- Zamanlayıcılar ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.audio_animation_timer = QTimer(self)
        self.audio_animation_timer.timeout.connect(self.update_audio_animations)
        self.audio_animation_timer.start(100)

        # --- Olay Abonelikleri ve Sinyal Bağlantıları ---
        # Pubsub
        pub.subscribe(self.update_log, 'log')
        pub.subscribe(self.handle_llm_response_received, 'llm_response') # <<< DEĞİŞTİRİLDİ/YENİ
        pub.subscribe(self.handle_speech_input, 'speech')
                # Yüz tanıma sonucu için bağlama
        pub.subscribe(self.handle_face_detection, 'face_detected')
        pub.subscribe(self.on_tts_really_complete_after_llm, 'tts:complete') # <<< YENİ ABONELİK

        # Hareket algılama sonucu için bağlama
        pub.subscribe(self.handle_motion_detection, 'motion_detected')

        pub.subscribe(self.handle_gesture_command, 'gesture_command') # Parmak izleme komutları
        pub.subscribe(self.handle_emotion_detected, 'emotion_detected') # Duygu tespiti
        pub.subscribe(self.handle_object_detected, 'object_detected') # Nesne tespiti

        pub.subscribe(self.handle_wake_word_detection_event, self.audio_manager.WAKE_WORD_DETECTED_TOPIC)
        pub.subscribe(self.handle_am_stt_stopped_check_ww, "am_stt_stopped_check_ww")
        pub.subscribe(self.update_wake_word_status_label, "wake_word_status_update") # AudioManager'dan durum günc.


        # Konuşma tanıma ve LLM yanıtı için bağlama
        self.speech_status_signal.connect(self.handle_speech_recognition_start)
        self.request_completed_signal.connect(self.handle_llm_response)

        # Sinyaller
        self.update_output_signal.connect(self.update_output_text)
        self.clear_output_signal.connect(self.clear_output_text)
        self.set_input_text_signal.connect(self.set_input_text)
        self.update_thinking_signal.connect(self.show_thinking_message)
        self.request_completed_signal.connect(self.on_request_completed)
        self.update_status_signal.connect(self.update_status_indicator)
        self.update_audio_status_signal.connect(self.update_audio_status)
        self.update_mic_level_signal.connect(self.update_mic_level)
        self.log_signal.connect(self.log)
        self.training_complete_signal.connect(self.handle_training_completion)
        self.speech_status_signal.connect(self.update_speech_status)

        if hasattr(self, 'update_gesture_label_signal'):
            self.update_gesture_label_signal.connect(self.update_gesture_display)
        if hasattr(self, 'update_emotion_label_signal'):
            self.update_emotion_label_signal.connect(self.update_emotion_display)
        if hasattr(self, 'update_object_label_signal'):
            self.update_object_label_signal.connect(self.update_object_display)
        # language_combo'nun create_control_panel içinde oluşturulduğunu varsayıyoruz.
        # Bu bağlantı create_control_panel çağrısından SONRA yapılmalı veya try-except içine alınmalı.
        # Şimdilik burada bırakalım, create_control_panel'de oluşturuluyorsa sorun olmaz.
        # Güvenlik için:
        try:
             if hasattr(self, 'language_combo'): # language_combo var mı diye kontrol et
                 self.language_combo.currentIndexChanged.connect(self.change_speech_language)
        except AttributeError as e:
             self.log(f"Warning: Could not connect language_combo signal yet: {e}")


        # Robot Listener Sinyalleri
        self.update_robot_status_ui.connect(self.update_robot_status_display)
        self.update_log_from_robot.connect(self.log)
       


        # Diğer Bağlantılar
        self.io_input.returnPressed.connect(self.send_to_llm)
        for widget in self.findChildren(QComboBox):
            widget.installEventFilter(self)
            widget.setFocusPolicy(Qt.StrongFocus)
        self.installEventFilter(self)

        self.robot_listener = RobotDataListener(listen_port=self.gui_listen_port)
        # !!! SİNYAL BAĞLANTILARINI BURAYA TAŞIYIN !!!
        self.robot_listener.robot_status_updated.connect(self.update_robot_status_ui)
        self.robot_listener.robot_log_received.connect(self.update_log_from_robot)
        self.robot_listener.robot_disconnected.connect(self.on_robot_disconnected)
        self.robot_listener.robot_connected.connect(self.on_robot_connected)
        # Kişilik listesi için sinyal bağlantısı (Listener'dan -> GUI Sinyaline)
        # Listener thread'ini başlat
        self.robot_listener.start()

        if hasattr(self, 'wake_word_checkbox') and self.wake_word_checkbox.isChecked():
            self.log("Başlangıçta WW checkbox işaretli, WW başlatılıyor.")
            self.audio_manager.start_wake_word_listener()
        else:
            self.log("Başlangıçta WW checkbox işaretli değil, WW kapalı başlıyor.")
            # Gerekirse durumu UI'a yansıt
            if hasattr(self, 'wake_word_status_label'):
                 self.wake_word_status_label.setText("WW: Kapalı (Başlangıç)")
            self.update_mic_button_and_indicator_based_on_state()

        # --- Son Başlatmalar ---
        # AudioManager başlatması (initialize çağrısı)
        self.audio_manager.initialize("direct", self.bluetooth_server) # <<< ARTIK HATA VERMEMELİ
        self.log("Audio Manager initialized in direct mode.")
        self.initialize_audio_events()
        # Bluetooth kontrolü kaldırıldı, gerekirse initialize veya apply_audio_settings içinde yapılır

        self.log("GUI initialized. Waiting for robot connection.")
        self.update_ui_connection_state(False) # Başlangıçta kontroller devre dışı

        # --- Tema seçili vurgusu için fonksiyon ---
        self.current_theme = 'red'  # Varsayılan
        self.update_theme_switcher_highlight(self.current_theme)

        # ... __init__ kodunuzun geri kalanı ...
        # GeminiHelper'ı API anahtarı varsa başlatmayı deneyin
        if self.gemini_api_key and GEMINI_MODULE_AVAILABLE:
            try:
                self.gemini_helper_instance = GeminiHelper(api_key=self.gemini_api_key, model=self.gemini_model_name)
                self.log(f"GeminiHelper '{self.gemini_model_name}' modeli ile başlangıçta yüklendi.")
                # Kayıtlı parametreleri uygula (eğer __init__ içinde daha önce set edildiyse)
                self.gemini_helper_instance.set_parameters(
                    temperature=self.gemini_temperature,
                    top_k=self.gemini_top_k,
                    top_p=self.gemini_top_p,
                    safety_settings=self.gemini_safety_settings,
                    system_instruction=self.gemini_system_instruction
                )
            except Exception as e:
                self.log(f"Başlangıçta GeminiHelper yüklenirken hata: {e}")
                self.gemini_helper_instance = None
        elif GEMINI_MODULE_AVAILABLE and not self.gemini_api_key:
            self.log("Gemini API anahtarı başlangıçta ayarlanmamış. Ayarlar menüsünden girilebilir.")

    # Modify send_animation signature and logic
    # Add color, color2 parameters with default None
    def send_animation(self, animation_name, color=None, color2=None):
        """Belirtilen animasyonu robota gönderir. Can accept programmatic colors."""
        self.log(f"--- Send Animation Call --- Name: {animation_name}, Color1: {color}, Color2: {color2}") # Çağrıyı logla

        if not self.command_sender or not self.command_sender.connected:
            self.log(f"Cannot send animation '{animation_name}': Not connected to robot.")
            return False

        try:
            # Define animation types (ensure these lists are accurate)
            # ESP32'deki animasyon isimleriyle eşleşmeli (BÜYÜK HARF)
            neopixel_no_color_animations = ["WAVE", "RAINBOW", "RAINBOW_CYCLE", "FIRE", "RANDOM_BLINK", "GRADIENT", "STACKED_BARS"]
            servo_animations = ["HEAD_NOD", "LOOK_UP", "WAVE_HAND", "CENTER"] # Servo animasyonları BÜYÜK HARF olmalı
            two_color_neopixel = ["ALTERNATING"] # Add others if needed

            # Prepare parameters
            params = {
                "animation": animation_name.upper(), # Robot side expects uppercase
                "repeat": 1 # Default repeat, can be overridden if needed
            }
            animation_type = "Unknown"

            # --- Determine Animation Type and Parameters ---
            # Servo animasyonları için küçük/büyük harf duyarsız kontrol
            if animation_name.upper() in servo_animations:
                animation_type = "SERVO"
                # Servo animations typically don't need color or repeat > 1
                params.pop("color", None)
                params.pop("color2", None)
                params["repeat"] = 1 # Force repeat to 1 for servos
                self.log(f"Animation type determined: SERVO")

            elif animation_name.upper() in neopixel_no_color_animations:
                animation_type = "NeoPixel (No Color)"
                params.pop("color", None)
                params.pop("color2", None)
                # Repeat might be relevant here, keep default or allow override later
                self.log(f"Animation type determined: NeoPixel (No Color)")

            else: # It's a NeoPixel animation that potentially needs color(s)
                animation_type = "NeoPixel (Color)"
                self.log(f"Animation type determined: NeoPixel (Color)")

                # --- COLOR 1 LOGIC ---
                if color: # 1. Use color passed as argument if available
                    params["color"] = color.upper() # Renkleri de büyük harfe çevir
                    self.log(f"Using programmatic color1: {params['color']}")
                elif hasattr(self, 'color_combo') and self.color_combo.isEnabled(): # 2. Try GUI combo if enabled
                    try:
                        params["color"] = self.color_combo.currentText().upper() # Renkleri de büyük harfe çevir
                        self.log(f"Using GUI color1: {params['color']}")
                    except Exception as e:
                        self.log(f"Could not get color1 from combo for {animation_name}: {e}")
                        params["color"] = "WHITE" # Fallback
                else: # 3. Fallback if no argument and no enabled combo
                    params["color"] = "WHITE"
                    self.log(f"Using fallback color1: WHITE")

                # --- COLOR 2 LOGIC (only for specific animations) ---
                if animation_name.upper() in two_color_neopixel:
                    self.log(f"Animation {animation_name} requires second color.")
                    if color2: # 1. Use color2 passed as argument
                        params["color2"] = color2.upper() # Renkleri de büyük harfe çevir
                        self.log(f"Using programmatic color2: {params['color2']}")
                    elif hasattr(self, 'color2_combo') and self.color2_combo.isEnabled(): # 2. Try GUI combo if enabled
                        try:
                            params["color2"] = self.color2_combo.currentText().upper() # Renkleri de büyük harfe çevir
                            self.log(f"Using GUI color2: {params['color2']}")
                        except Exception as e:
                            self.log(f"Could not get color2 from combo for {animation_name}: {e}")
                            params.pop("color2", None) # Remove if error
                    else: # 3. Fallback if no argument and no enabled combo
                         params.pop("color2", None)
                         self.log(f"No color2 provided or combo disabled for {animation_name}")
                else:
                    # Ensure color2 is not sent for single-color animations
                    params.pop("color2", None)
                    # self.log(f"Animation {animation_name} does not require second color.") # Çok fazla log olmaması için yorumlandı

            # --- Logging and Sending ---
            self.log(f"Sending {animation_type} animation command: '{params['animation']}' with final params: {params}")
            response = self.command_sender.send_command("send_animation", params)

            response_status = response.get('status', 'unknown') if response else 'no_response'
            response_message = response.get('message', '') if response else ''
            self.log(f"Animation command '{params['animation']}' sent. Response: {response_status} - {response_message}")
            self.log(f"--- Send Animation End ---") # Bitiş logu

            return response_status == 'ok'
        except Exception as e:
            self.log(f"!!! Error sending animation '{animation_name}': {e}")
            import traceback
            self.log(traceback.format_exc())
            self.log(f"--- Send Animation End (Error) ---") # Hata durumunda bitiş logu
            return False
        
    def toggle_connection(self):
        """Bağlantı durumunu değiştir (bağlan/bağlantıyı kes)."""
        if self.stream_connected:
            self.disconnect_from_robot()
            self.log("Robot bağlantısı kesildi.")
        else:
            self.connect_to_robot()

# toggle_microphone metodunu bu şekilde güncelleyin:
    # Mikrofon butonu
    def toggle_microphone(self):
        """Mikrofon butonunun davranışını yönetir."""
        if self.audio_manager.speech_active: # STT aktifse
            self.log("Mic Button: STT çalışıyor -> Durduruluyor (WW kontrol edilecek).")
            # STT'yi durdur, AudioManager WW'yi (gerekirse) yeniden başlatacak
            self.audio_manager.stop_speech_recognition(restart_wake_word_if_enabled=True)
        elif self.audio_manager.is_wake_word_listening: # Sadece WW aktifse
            self.log("Mic Button: Sadece WW çalışıyor -> Durduruluyor.")
            self.audio_manager.stop_wake_word_listener()
            if hasattr(self, 'wake_word_checkbox'): # Checkbox'ı da senkronize et
                self.wake_word_checkbox.setChecked(False)
        else: # Hiçbir şey aktif değilse (ne STT ne WW)
            self.log("Mic Button: Her şey kapalı -> STT Başlatılıyor.")
            self.audio_manager.start_speech_recognition()
        # UI güncellemeleri speech_status_signal ve wake_word_status_update üzerinden gelecek

    # Modify handle_face_detection
    def handle_face_detection(self, detected_names):
        """Yüz tanıma olayında animasyon gönderir."""
        if not detected_names:
            # Clear last detected if no faces are seen now
            if hasattr(self, 'last_detected_face_names') and self.last_detected_face_names:
                 self.last_detected_face_names = set()
                 # self.log("No faces detected, cleared last detected.") # Çok fazla log olmaması için yorumlandı
            return

        current_names_set = set(detected_names)
        # Sadece isim seti değiştiğinde loglama ve işlem yap
        if not hasattr(self, 'last_detected_face_names') or self.last_detected_face_names != current_names_set:
            self.last_detected_face_names = current_names_set
            self.log(f"--- Face Detection Event --- Detected faces changed: {list(current_names_set)}") # Ayırıcı log

            # --- Process only the highest priority person found in this frame ---
            priority_found = False
            highest_priority_person = None
            highest_priority_animation = None
            highest_priority_color = "GREEN" # Default color for priority

            # FaceDetector'dan priority verisini al (varsayılan yapıya göre)
            if hasattr(self.face_detector, 'priority_animations') and self.face_detector.priority_animations:
                priority_data = self.face_detector.priority_animations
                self.log(f"Priority animation data loaded: {priority_data}") # Verinin yüklendiğini logla

                for name in detected_names:
                    if name in priority_data:
                         person_info = priority_data[name]
                         self.log(f"Checking priority for detected person: {name}. Info: {person_info}") # Kişiyi ve bilgisini logla

                         # !!! DEĞİŞİKLİK BAŞLANGICI: person_info tipini kontrol et !!!
                         if isinstance(person_info, dict):
                             # Eğer sözlük ise, .get() kullan
                             animation_name = person_info.get("animation")
                             color_name = person_info.get("color", "GREEN") # Varsayılan renk GREEN
                         elif isinstance(person_info, str):
                             # Eğer metin ise, doğrudan animasyon adı olarak kullan
                             animation_name = person_info
                             color_name = "GREEN" # Varsayılan renk GREEN
                             self.log(f"Note: Priority info for {name} is a string. Using default color GREEN.")
                         else:
                             # Beklenmeyen bir tip ise atla
                             self.log(f"Warning: Unexpected priority info type for {name}: {type(person_info)}")
                             animation_name = None
                             color_name = "GREEN"
                         # !!! DEĞİŞİKLİK SONU !!!

                         # TODO: Could add logic here to find the *highest* priority if multiple are present
                         if animation_name and not priority_found: # Animasyon adı geçerliyse devam et
                             highest_priority_person = name
                             highest_priority_animation = animation_name # Değiştirildi
                             highest_priority_color = color_name # Değiştirildi
                             priority_found = True
                             self.log(f"Priority person found: {highest_priority_person}, Animation: {highest_priority_animation}, Color: {highest_priority_color}") # Bulunan öncelikli kişiyi logla
                             break # Stop after finding the first priority person
            else:
                self.log("No priority animation data found in FaceDetector.")

            if priority_found and highest_priority_animation:
                self.log(f"Attempting to send priority animation '{highest_priority_animation}' with color '{highest_priority_color}' for: {highest_priority_person}")
                # Pass animation name AND color to send_animation
                success = self.send_animation(highest_priority_animation, color=highest_priority_color) # Renk parametresini gönder
                if not success:
                     self.log(f"!!! Failed to send priority animation '{highest_priority_animation}' for {highest_priority_person}")
            elif detected_names: # If faces detected but none are priority
                 # self.log(f"Detected non-priority faces: {list(detected_names)}, no specific animation sent.") # Çok fazla log olmaması için yorumlandı
                 pass
        # else: # İsim seti değişmediyse loglama (debug için açılabilir)
            # self.log(f"Face detection names unchanged: {list(current_names_set)}")

    def handle_speech_recognition_start(self, active, message):
        """Konuşma tanıma başladığında/bittiğinde animasyon gönderir."""
        if active:
            # look_up bir servo animasyonu olmalı
            self.log("Speech recognition started, sending 'look_up' animation.")
            self.send_animation("look_up")
        else:
            self.log("Speech recognition stopped.")
            # İsteğe bağlı: Durduğunda farklı bir animasyon
            # self.send_animation("look_down")

    # Modify handle_motion_detection
    def handle_motion_detection(self, motion_detected):
        """Sends motion event notification to the robot."""
        if not self.command_sender or not self.command_sender.connected:
            # self.log("Cannot send motion event: Not connected.") # Reduce logging noise
            return

        # Only send if state changes to avoid flooding
        if not hasattr(self, 'last_motion_state') or self.last_motion_state != motion_detected:
            self.last_motion_state = motion_detected
            try:
                self.log(f"Sending motion event to robot: detected={motion_detected}")
                # Send an event notification instead of a specific animation command
                response = self.command_sender.send_command('motion_event', {'detected': motion_detected})
                # Optional: Log response
                response_status = response.get('status', 'unknown') if response else 'no_response'
                self.log(f"Motion event sent. Response: {response_status}")
            except Exception as e:
                self.log(f"Error sending motion event: {e}")
        
        # handle_llm_response metodunu önceki yanıttaki gibi parametresiz bırakın
    
    def handle_llm_response(self): # response_text parametresi kaldırıldı
        """LLM yanıtı geldiğinde animasyon gönderir."""
        try:
            if self.stream_connected:
                 # head_nod bir servo animasyonu, send_animation doğru işlemeli
                 self.send_animation("head_nod")
            else:
                 self.log("Cannot send head_nod animation: Not connected.")
        except Exception as e:
            self.log(f"Error sending 'head_nod' animation on LLM response: {e}")

    @QtCore.pyqtSlot(float)
    def update_mic_level(self, level):
        """Mikrofon ses seviyesi göstergesini günceller"""
        try:
            # Ses seviyesini 0-100 aralığına normalize et
            normalized_level = min(100, max(0, int(level * 100)))
            
            # Progress bar'ı güncelle
            if hasattr(self, 'audio_level_bar'):
                self.audio_level_bar.setValue(normalized_level)
                
            # Mikrofon seviyesini kaydet
            self.mic_level = normalized_level
            
            # Mikrofon aktifse, indikatörün rengini güncelle
            if hasattr(self, 'mic_indicator'):
                if self.mic_active:
                    # Ses seviyesine göre renk değiştir (yeşilden kırmızıya)
                    if normalized_level > 75:
                        self.mic_indicator.setStyleSheet("color: #ff0000; font-size: 18pt;")  # Kırmızı (yüksek)
                    elif normalized_level > 30:
                        self.mic_indicator.setStyleSheet("color: #ffa500; font-size: 18pt;")  # Turuncu (orta)
                    else:
                        self.mic_indicator.setStyleSheet("color: #00aa00; font-size: 18pt;")  # Yeşil (düşük)
                else:
                    self.mic_indicator.setStyleSheet("color: gray; font-size: 18pt;")  # İnaktif
                    
        except Exception as e:
            # Hata durumunda sessizce devam et, ancak hata mesajını kaydet
            self.log(f"Mikrofon seviyesi güncellenirken hata: {e}")

    def populate_tts_languages(self):
        """TTS dil seçeneği için kullanılabilir dilleri doldur"""
        self.tts_language_combo.clear()
        
        if not self.tts_voices:
            self.tts_language_combo.addItem("Dil seçenekleri bulunamadı")
            return
        
        # Dil adlarını düzenli bir şekilde göstermek için sözlük
        # Bazı diller için Türkçe adlar, diğerleri için genel format
        language_names = {
            'tr-TR': 'Türkçe',
            'en-US': 'İngilizce (ABD)',
            'en-GB': 'İngilizce (İngiltere)',
            'de-DE': 'Almanca',
            'fr-FR': 'Fransızca',
            'es-ES': 'İspanyolca',
            'it-IT': 'İtalyanca',
            'pt-PT': 'Portekizce',
            'nl-NL': 'Hollandaca',
            'ru-RU': 'Rusça',
            'pl-PL': 'Lehçe',
            'hu-HU': 'Macarca',
            'ar-SA': 'Arapça',
        }
        
        # Dil listesi düzenleniyor ve sıralanıyor
        sorted_languages = sorted(self.tts_voices.keys())
        
        # Her dil için seçeneği ekle
        for lang_code in sorted_languages:
            voice_count = len(self.tts_voices[lang_code])
            
            # Dil kodlarını tutarlı hale getir
            normalized_code = lang_code
            if len(lang_code) == 2:
                # 2 karakterli kodları standartlaştır: ar -> ar-AR
                normalized_code = f"{lang_code}-{lang_code.upper()}"
            
            # Dil adını belirle: Özel isim varsa kullan, yoksa kodu kullan
            pretty_name = language_names.get(normalized_code, normalized_code)
            
            # Gösterim formatı: "Dil Adı (ses sayısı)"
            display_text = f"{pretty_name} ({voice_count} ses)"
            
            # Seçeneği ekle
            self.tts_language_combo.addItem(display_text, lang_code)
            
            # Mevcut dil ayarına göre seçimi yap
            if self.current_tts_lang == lang_code:
                self.tts_language_combo.setCurrentIndex(self.tts_language_combo.count() - 1)

    def detect_language(self, text):
        """
        Metni analiz ederek dilin ne olduğunu tespit eder ve 
        uygun TTS dilini döndürür
        """
        try:
            if not text or len(text) < 5:  # Çok kısa metinlerde dil tespiti yanıltıcı olabilir
                self.log("Metin dil tespiti için çok kısa")
                return self.current_tts_lang  # Varsayılan dili kullan
            
            # langdetect ile dil tespiti yap
            detected_lang = detect(text)
            self.log(f"Tespit edilen dil kodu: {detected_lang}")
            
            # Piper dil formatına dönüştür: 2 karakter -> 2 karakter-2 KARAKTER 
            piper_format = f"{detected_lang.lower()}-{detected_lang.upper()}"
            self.log(f"Aranacak Piper dil formatı: {piper_format}")
            
            # Piper formatında arama yap
            if piper_format in self.tts_voices and self.tts_voices[piper_format]:
                self.log(f"Dil bulundu: '{piper_format}' - {len(self.tts_voices[piper_format])} ses mevcut")
                return piper_format
                
            # Bulunamadıysa self.tts_voices içinde arama yap (öneklere bak)
            for voice_lang in self.tts_voices.keys():
                # İki harfli dil kodu ile başlayan tüm dilleri kontrol et
                # Örneğin "en" için "en-US", "en-GB" gibi
                if voice_lang.lower().startswith(detected_lang.lower() + '-'):
                    self.log(f"Alternatif dil bulundu: {voice_lang} - {len(self.tts_voices[voice_lang])} ses mevcut")
                    return voice_lang
            
            # Özel dil haritalaması (olağan dışı kodlar için)
            special_mappings = {
                'zh': 'zh-ZH',  # Çince
                'ar': 'ar-AR',  # Arapça
                'cs': 'cs-CS',  # Çekçe
                'cy': 'cy-CY',  # Galce
                'da': 'da-DA',  # Danca
                'el': 'el-EL',  # Yunanca
                'fa': 'fa-FA',  # Farsça
                'fi': 'fi-FI',  # Fince
                'hu': 'hu-HU',  # Macarca
                'is': 'is-IS',  # İzlandaca
                'it': 'it-IT',  # İtalyanca
                'ka': 'ka-KA',  # Gürcüce
                'kk': 'kk-KK',  # Kazakça
                'lb': 'lb-LB',  # Lüksemburgca
                'lv': 'lv-LV',  # Letonca
                'ne': 'ne-NE',  # Nepalce
                'nl': 'nl-NL',  # Hollandaca
                'no': 'no-NO',  # Norveççe
                'pl': 'pl-PL',  # Lehçe
                'pt': 'pt-PT',  # Portekizce
                'ro': 'ro-RO',  # Romence
                'ru': 'ru-RU',  # Rusça
                'sk': 'sk-SK',  # Slovakça
                'sl': 'sl-SL',  # Slovence
                'sr': 'sr-SR',  # Sırpça
                'sv': 'sv-SV',  # İsveççe
                'sw': 'sw-SW',  # Svahili
                'tr': 'tr-TR',  # Türkçe
                'uk': 'uk-UK',  # Ukraynaca
                'vi': 'vi-VI'   # Vietnamca
            }
            
            # Özel eşleştirme kontrol et
            if detected_lang in special_mappings:
                mapped_lang = special_mappings[detected_lang]
                if mapped_lang in self.tts_voices:
                    self.log(f"Özel eşleştirme kullanıldı: {detected_lang} -> {mapped_lang}")
                    return mapped_lang
            
            # Son olarak, İngilizce için "en-EN" veya "en-US" kontrolü
            if detected_lang == 'en':
                # Önce "en-EN" (Piper standardı) dene
                if 'en-EN' in self.tts_voices:
                    self.log("İngilizce tespit edildi, en-EN kullanılıyor")
                    return 'en-EN'
                # Alternatif olarak en-US dene
                elif 'en-US' in self.tts_voices:
                    self.log("İngilizce tespit edildi, en-US kullanılıyor")
                    return 'en-US'
            
            # Detaylı hata mesajı logla
            self.log(f"UYARI: '{detected_lang}' dili için ses bulunamadı!")
            self.log(f"Mevcut diller: {', '.join(self.tts_voices.keys())}")
            
            # Hiçbir eşleşme bulunamazsa varsayılan dil kullan
            return self.current_tts_lang
                
        except LangDetectException as e:
            self.log(f"Dil tespiti hatası: {e}")
            return self.current_tts_lang
        except Exception as e:
            self.log(f"Beklenmeyen dil tespiti hatası: {e}")
            return self.current_tts_lang

    @QtCore.pyqtSlot(list)
    def update_personality_combo_from_list(self, personalities):
        """Thread güvenli olarak personality combo box'ı günceller"""
        if not hasattr(self, 'personality_combo'):
            return

        current_selection = self.personality_combo.currentText() # Mevcut seçimi sakla

        self.personality_combo.clear()
        if personalities and isinstance(personalities, list) and len(personalities) > 0: # Liste kontrolü
            self.personality_combo.addItems(personalities)
            self.personality_combo.setEnabled(True)
             # Eskiden seçili olanı tekrar seçmeye çalış
            if current_selection in personalities:
                self.personality_combo.setCurrentText(current_selection)
            elif self.personality_combo.count() > 0:
                 self.personality_combo.setCurrentIndex(0) # Veya ilk elemanı seç

            self.log(f"Personality list updated from robot: {personalities}")
        else:
            self.personality_combo.addItem("No personalities")
            self.personality_combo.setEnabled(False)
            self.log("Received empty or invalid personality list from robot.")

    def select_random_voice_for_language(self, lang_code):
            """
            Belirtilen dil için mevcut seslerden rastgele birini seçer
            """
            if not lang_code in self.tts_voices or not self.tts_voices[lang_code]:
                self.log(f"'{lang_code}' dili için ses bulunamadı")
                return None
            
            # O dildeki mevcut sesleri al
            voices = self.tts_voices[lang_code]
            
            # Rastgele bir ses seç
            import random
            random_voice = random.choice(voices)
            
            self.log(f"'{lang_code}' dili için rastgele ses seçildi: {random_voice['name']}")
            return random_voice['id']
        
    def speak_text_with_auto_language(self, text):
        """Metni otomatik dil algılama ile seslendirir - thread güvenli"""
        # Metni ve dil tespiti işlemlerini arka planda yap
        def _auto_lang_worker(text):
            try:
                # Metni temizle
                clean_text = re.sub(r"\[cmd:[a-zA-Z0-9_]+\]", "", text).strip()
                clean_text = self.clean_text_for_tts(clean_text)
                
                # Dili algıla
                detected_lang = self.detect_language(clean_text)
                
                # UI threadi'nde bilgi göster
                QtCore.QMetaObject.invokeMethod(self, "_log_auto_lang", 
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, detected_lang))
                
                # Seçilen dil için ses seç
                voice_id = self.get_voice_for_language(detected_lang)
                
                # Model dosyası mevcut mu?
                if not os.path.exists(voice_id):
                    # Dil için alternatif ses ara
                    alt_voice = self.select_random_voice_for_language(detected_lang)
                    if alt_voice:
                        voice_id = alt_voice
                
                # Verileri saklayarak TTS işlemini başlat
                original_voice = self.current_tts_voice
                original_lang = self.current_tts_lang
                
                # Geçici olarak dil ve ses ayarlarını değiştir
                self.current_tts_lang = detected_lang
                self.current_tts_voice = voice_id
                
                # Seslendirme işlemini başlat (bu zaten ayrı bir thread'de çalışacak)
                self.speak_text_locally(clean_text)
                
                # UI'da dili göster
                for i in range(self.tts_language_combo.count()):
                    if self.tts_language_combo.itemData(i) == detected_lang:
                        index = i
                        # UI threadi'nde Combobox değerini değiştir
                        QtCore.QMetaObject.invokeMethod(self.tts_language_combo, "setCurrentIndex", 
                            QtCore.Qt.QueuedConnection,
                            QtCore.Q_ARG(int, index))
                        break
                
            except Exception as e:
                # Hata durumunda normal yöntemle seslendir
                QtCore.QMetaObject.invokeMethod(self, "_log_auto_lang_error", 
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, str(e)))
                self.speak_text_locally(text)
        
        # İşlemi ayrı thread'e taşı
        threading.Thread(target=_auto_lang_worker, args=(text,), daemon=True).start()
        return True
    
    @QtCore.pyqtSlot(str)
    def _log_auto_lang(self, detected_lang):
        """Dil tespiti loglaması - UI thread'inde çalışır"""
        self.log(f"Otomatik dil tespiti: {detected_lang} dilinde konuşuluyor")
        
    @QtCore.pyqtSlot(str)
    def _log_auto_lang_error(self, error_msg):
        """Dil tespiti hatası loglaması - UI thread'inde çalışır"""
        self.log(f"Otomatik dil tespiti hatası: {error_msg}")
    def on_tts_language_changed(self, index):
        """TTS dili değiştiğinde sesleri güncelle"""
        if index < 0 or not self.tts_engine:
            return
        
        # Seçilen dil kodunu al
        lang_code = self.tts_language_combo.itemData(index)
        if not lang_code or lang_code not in self.tts_voices:
            return
        
        # Ses combobox'ını seçilen dil için güncelle
        self.populate_voice_combo_for_language(lang_code)
        
        # Google TTS için dil kodunu kaydet
        self.current_tts_lang = lang_code
        
        # İlk sesi seç ve ayarla
        if self.voice_combo.count() > 0:
            self.voice_combo.setCurrentIndex(0)
            voice_id = self.voice_combo.itemData(0)
            if voice_id:
                self.current_tts_voice = voice_id
                self.log(f"TTS dili değiştirildi: {lang_code}, ses kodu: {self.current_tts_voice}")
        
        # Otomatik test özelliğini kaldırdık
        # QtCore.QTimer.singleShot(500, lambda: self.speak_text_locally("Merhaba, bu bir test mesajıdır."))
    
    def populate_voice_combo_for_language(self, lang_code):
        """Belirli bir dil için ses combobox'ını doldur"""
        self.voice_combo.clear()
        
        if not self.tts_voices or lang_code not in self.tts_voices:
            self.voice_combo.addItem("Ses bulunamadı")
            return
        
        # Sadece belirtilen dildeki sesleri göster
        voices = self.tts_voices[lang_code]
        for voice in voices:
            # None kontrolü eklendi
            gender = voice.get('gender', '')
            gender_str = str(gender) if gender is not None else ''
            
            # "Ses Adı (Cinsiyet)" formatında göster
            gender_text = "Erkek" if gender_str.lower() == 'male' else "Kadın" if gender_str.lower() == 'female' else "?"
            display_name = f"{voice['name']} ({gender_text})"
            self.voice_combo.addItem(display_name, voice['id'])

    # --- YENİ SLOTLAR ---
    @QtCore.pyqtSlot(str, str)
    def _show_warning_message_slot(self, title, message):
        """Shows a warning message box (thread-safe)."""
        QtWidgets.QMessageBox.warning(self, title, message)

    @QtCore.pyqtSlot(str, str)
    def _show_critical_message_slot(self, title, message):
        """Shows a critical message box (thread-safe)."""
        QtWidgets.QMessageBox.critical(self, title, message)

    @QtCore.pyqtSlot(str, str)
    def _show_info_message_slot(self, title, message):
        """Shows an information message box (thread-safe)."""
        QtWidgets.QMessageBox.information(self, title, message)
    # --- SLOTLAR SONU ---

    def set_robot_eye_color(self):
        """Sends the command to set the robot's eye color."""
        if not self.command_sender or not self.command_sender.connected:
            self.log("Robot bağlı değil. Göz rengi ayarlanamıyor.")
            return
            
        color = self.eye_color_input.text().strip().lower()
        if not color:
            self.log("Lütfen bir göz rengi girin.")
            return
        
        self.log(f"Göz rengi ayarlama komutu gönderiliyor: {color}")
        try:
            response = self.command_sender.send_command("set_eye", {"color": color})
            self.log(f"Komut yanıtı: {response.get('status', 'unknown')} - {response.get('message', 'no message')}")
        except Exception as e:
            self.log(f"Göz rengi ayarlama hatası: {e}")

        # Set_state metodu için düzeltme
    def set_robot_state(self):
        """Sends the command to set the robot's state."""
        if not self.stream_connected:
            self.log("Cannot set state: Not connected.")
            return
        state = self.state_combo.currentText()
        if not state:
            self.log("Invalid state selected.")
            return
    
        self.log(f"Sending command to set state to: {state}")
        try:
             def _set_state_thread():
                 try:
                     response = self.command_sender.send_command('set_state', {'state': state})
                     if not response or response.get('status') != 'ok':
                         error_msg = response.get('message', 'Unknown error') if response else "No response"
                         self.log(f"Failed to send set_state command: {error_msg}")
                         QtCore.QMetaObject.invokeMethod(
                             self, "_show_warning_message_slot", QtCore.Qt.QueuedConnection,
                             QtCore.Q_ARG(str,"Error"),
                             QtCore.Q_ARG(str, f"Failed to set state: ...") # Hata mesajını düzelt
                         )
                 except Exception as e:
                     self.log(f"Error in set_state thread: {e}")
                     QtCore.QMetaObject.invokeMethod(
                         self, "_show_critical_message_slot", QtCore.Qt.QueuedConnection,
                         QtCore.Q_ARG(str,"Error"),
                         QtCore.Q_ARG(str, f"Exception while setting state: {e}")
                     )
             t = threading.Thread(target=_set_state_thread, daemon=True)
             t.start()
        except Exception as e:
            self.log(f"Error sending set_state command: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Error sending command:\n{e}")
    # --- SLOTS for Robot Listener Signals ---

    def disconnect_from_robot(self):
        """Disconnect from the robot by stopping video stream and closing command socket."""
        if hasattr(self, 'video_stream') and self.video_stream:
            self.log("Stopping video stream...")
            try:
                self.video_stream.stop()
                self.video_stream = None
            except Exception as e:
                self.log(f"Error stopping video stream: {e}")
        
        if hasattr(self, 'timer') and self.timer.isActive():
            self.log("Stopping timer...")
            self.timer.stop()
        
        if hasattr(self, 'command_sender') and self.command_sender:
            self.log("Closing command socket...")
            try:
                self.command_sender.close()
                # Don't set to None, just mark as disconnected
            except Exception as e:
                self.log(f"Error closing command socket: {e}")
        
        self.stream_connected = False
        self.log("Disconnected from robot.")
        
        # Update UI
        if hasattr(self, 'connect_button'):
            self.connect_button.setText("Connect")
        
        self.update_ui_connection_state(False)

    # update_robot_status_display METODUNU GÜNCELLE
    @QtCore.pyqtSlot(dict)
    def update_robot_status_display(self, status_data):
        """Robottan gelen durum verilerini UI'da günceller."""
        if not status_data:
            return
        
        try:
            # Robot durumunu gösterge etiketlerine yansıt
            state = status_data.get('state', 'UNKNOWN')
            eye_color = status_data.get('eye_color', 'unknown')
            personality = status_data.get('current_personality', 'unknown')
            
            # Durum göstergesini güncelle
            self.robot_state_label.setText(f"State: {state}")
            self.robot_eye_label.setText(f"Eye Color: {eye_color}")
            self.robot_personality_label.setText(f"Personality: {personality}")
            
            # ÖNEMLİ: Göz rengi giriş kutusunu güncelleme
            # self.eye_color_input.setText(eye_color)  # Bu satırı kaldır veya yorum yap
            
            # Durum göstergesinin rengini duruma göre ayarla
            if state == "ALERT":
                self.robot_state_label.setStyleSheet("color: red; font-weight: bold")
            elif state == "IDLE":
                self.robot_state_label.setStyleSheet("color: green; font-weight: bold")
            elif state == "RESTING":
                self.robot_state_label.setStyleSheet("color: blue; font-weight: bold")
            elif state == "SLEEPING":
                self.robot_state_label.setStyleSheet("color: gray; font-weight: bold")
            else:
                self.robot_state_label.setStyleSheet("")
                
            # Bağlantı durumu göstergesini güncelle
            self.robot_connection_label.setText("Connected")
            self.robot_connection_label.setStyleSheet("color: green; font-weight: bold")
            
            # Debug amaçlı logla
            self.log(f"Robot durumu güncellendi: Durum={state}, Göz={eye_color}, Kişilik={personality}")
            
        except Exception as e:
            self.log(f"UI güncelleme hatası: {e}")
    
    @QtCore.pyqtSlot(str)
    def on_robot_connected(self, address):
        """Called when the robot connects to our listener."""
        self.log(f"Robot connected from: {address}")
        self.robot_connected_status = True
        self.robot_connection_label.setText(f"Connected ({address})")
        self.robot_connection_label.setStyleSheet("color: green; font-weight: bold;")
        
        # Robot durumunu isteyelim
        QtCore.QTimer.singleShot(500, self.request_robot_status)
        
        # Bağlantı düğmesini güncelle
        if hasattr(self, 'connect_button'):
            self.connect_button.setText("Disconnect")
            
        # UI kontrollerini etkinleştir
        self.update_ui_connection_state(True)

    def request_robot_status(self):
        """Sends a command to request the robot's current status."""
        if not hasattr(self, 'command_sender') or not self.command_sender.connected:
            self.log("Cannot request status: Not connected to robot.")
            return
            
        self.log("Requesting robot status...")
        try:
            response = self.command_sender.send_command('get_status')
            if not response or response.get('status') != 'ok':
                error_msg = response.get('message', 'Unknown error') if response else "No response"
                self.log(f"Failed to request robot status: {error_msg}")
        except Exception as e:
            self.log(f"Error requesting robot status: {e}")

    @QtCore.pyqtSlot()
    def on_robot_disconnected(self):
        """Called when the robot disconnects from our listener."""
        self.log("Robot data connection lost.")
        self.robot_connected_status = False
        self.robot_connection_label.setText("Disconnected")
        self.robot_connection_label.setStyleSheet("color: red; font-weight: bold;")
        # Reset status fields
        self.update_robot_status_display({}) # Send empty dict to reset labels

    def closeEvent(self, event):
        """Release resources on application close."""
        self.log("Closing GUI...")
        # Stop listener thread first
        if hasattr(self, 'robot_listener') and self.robot_listener:
            self.log("Stopping Robot Listener thread...")
            self.robot_listener.stop_listening()
            if self.robot_listener.isRunning():
                try:
                    self.robot_listener.wait(1000)  # 1 saniye bekle
                except Exception as e:
                    self.log(f"Error waiting for robot listener to finish: {e}")
    
        # Disconnect from robot
        self.disconnect_from_robot()  # Artık tanımlı
    
        # Stop audio
        if hasattr(self, 'audio_manager'):
            try:
                if hasattr(self.audio_manager, 'stop_speech_recognition'):
                    self.audio_manager.stop_speech_recognition()
            except Exception as e:
                self.log(f"Error stopping audio manager: {e}")
    
        # Shutdown thread pools
        self.log("Shutting down thread pools...")
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        if hasattr(self, 'face_detector') and hasattr(self.face_detector, 'executor'):
            self.face_detector.executor.shutdown(wait=False)
        if hasattr(self, 'audio_thread_manager'):
            self.audio_thread_manager.shutdown()
    
        self.log("Cleanup complete. Exiting.")
        event.accept()

    def initialize_tts_engine(self):
        """Piper TTS motorunu başlat"""
        try:
            # PyDub kontrolü
            try:
                from pydub import AudioSegment
                self.pydub_available = True
                self.pydub_audioseq = AudioSegment
                self.log("PyDub bulundu, ses hızlandırma etkin")
            except ImportError:
                self.pydub_available = False
                self.log("PyDub bulunamadı, ses hızlandırma devre dışı. pip install pydub ile kurabilirsiniz")
            
            # Pygame mixer'ı başlat
            pygame.mixer.init()
            
            self.tempfile_module = tempfile
            self.os_module = os
            self.pygame_module = pygame
            
            # Piper çalıştırılabilir dosyasını bul
            self.piper_executable = self.find_piper_executable()
            
            # TTS çalışıyor mu kontrol et
            self.tts_engine = True
            self.tts_temp_dir = tempfile.gettempdir()
            self.tts_speed = 1.0
            
            # Varsayılan TTS motorunu ayarla - burası eksikti
            self.tts_engine_type = "piper"  # Varsayılan motor olarak gtts kullan
            
            # Piper ses dosyalarını bul
            self.tts_voices = self.find_piper_voices()
            if not self.tts_voices:
                self.log("UYARI: Piper ses dosyaları bulunamadı. Lütfen piper klasöründe *.onyx dosyalarının olduğundan emin olun.")
                # Ses modelleri bulunamadıysa boş bir çözüm ayarla
                self.tts_voices = {
                    'tr-TR': [{'id': 'piper/tr-TR-model.onnx', 'name': 'Türkçe', 'gender': 'unknown'}],
                    'en-US': [{'id': 'piper/en-US-model.onnx', 'name': 'İngilizce (ABD)', 'gender': 'unknown'}]
                }
            else:
                self.log(f"Piper TTS motoru başarıyla başlatıldı, {len(self.tts_voices)} dil kullanılabilir")
            
            # İlk sesi varsayılan olarak ayarla
            if 'tr-TR' in self.tts_voices and self.tts_voices['tr-TR']:
                self.current_tts_voice = self.tts_voices['tr-TR'][0]['id']
                self.current_tts_lang = 'tr-TR'
            elif len(self.tts_voices) > 0:
                # İlk mevcut dili kullan
                first_lang = list(self.tts_voices.keys())[0]
                self.current_tts_voice = self.tts_voices[first_lang][0]['id']
                self.current_tts_lang = first_lang
            else:
                self.log("UYARI: Hiç ses modeli bulunamadı!")
                return False
            
            return True
            
        except Exception as e:
            self.log(f"TTS motoru başlatılırken hata: {e}")
            self.log(traceback.format_exc())
            self.tts_engine = None
            return False
      
    def clean_text_for_tts(self, text):
        """Metni TTS için hazırlar, emojileri ve desteklenmeyen karakterleri temizler"""
        import re
        import unicodedata
        
        # Null kontrolü
        if not text:
            return ""
        
        # Önce unicode normalleştirme
        text = unicodedata.normalize('NFKD', text)
        
        # Emoji'leri ve diğer özel sembolleri tamamen temizle
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002500-\U00002BEF"  # Misc Symbols
            "\U00002702-\U000027B0"  # Dingbats
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001F004-\U0001F251"
            "]+", flags=re.UNICODE
        )
        clean_text = emoji_pattern.sub(' ', text)
        
        # Sadece Türkçe harfleri, ASCII ve temel noktalama işaretlerini tut
        allowed_chars = (
            'abcçdefgğhıijklmnoöpqrsştuüvwxyzABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVWXYZ'  # Türkçe harfler
            '0123456789'  # Rakamlar
            ' ,.!?;:()[]{}_-+="\'%/'  # Temel noktalama işaretleri
        )
        filtrelenmis_metin = ''.join(c for c in clean_text if c in allowed_chars)
        
        # Birden fazla boşlukları tek boşluğa indir
        filtrelenmis_metin = re.sub(r'\s+', ' ', filtrelenmis_metin).strip()
        
        self.log(f"Metin TTS için temizlendi: {len(text)} karakter -> {len(filtrelenmis_metin)} karakter")
        
        # Minimum uzunluk kontrolü
        if len(filtrelenmis_metin) < 2:
            filtrelenmis_metin = "Mesaj anlaşılamadı."
            
        return filtrelenmis_metin
    
    def convert_to_piper_lang_format(self, lang_code):
            """
            Herhangi bir dil kodunu Piper'ın kullandığı formata çevirir
            Örneğin: 'en' -> 'en-EN', 'en-US' -> 'en-EN'
            """
            # Eğer format zaten xx-XX şeklindeyse doğrudan kullan
            if len(lang_code) >= 5 and lang_code[2] == '-':
                base_lang = lang_code[:2].lower()
                return f"{base_lang}-{base_lang.upper()}"
            else:
                # İki harfli basit kod ise (en, tr, fr gibi) Piper formatına çevir
                base_lang = lang_code[:2].lower()
                piper_format = f"{base_lang}-{base_lang.upper()}"
                
                # Eğer bu formatta ses bulunmuyorsa, alternatif formatlar dene
                if piper_format not in self.tts_voices:
                    # Dil kodlarını sözlükte ara
                    for available_lang in self.tts_voices.keys():
                        # Eğer dil aynı ise (en-EN, en-US, en-GB gibi varyasyonları da kapsayacak şekilde)
                        if available_lang.lower().startswith(base_lang.lower() + '-'):
                            return available_lang
                
                return piper_format    
    
    def find_piper_executable(self):
        """İşletim sistemine göre uygun Piper çalıştırılabilir dosyasını bulur"""
        
        # Sabit olarak kullanıcının belirttiği konumu dene (öncelikli)
        user_piper_dir = "C:\\Users\\emirh\\piper"
        user_piper_exe = os.path.join(user_piper_dir, "piper.exe")
        
        if os.path.exists(user_piper_exe):
            self.log(f"Piper özel konumdan yüklendi: {user_piper_exe}")
            return user_piper_exe
        
        # Kullanıcı dizinindeki genel konumu dene
        home_piper_dir = os.path.join(os.path.expanduser("~"), "piper")
        home_piper_exe = os.path.join(home_piper_dir, "piper.exe" if sys.platform == 'win32' else "piper")
        
        if os.path.exists(home_piper_exe):
            self.log(f"Piper kullanıcı dizininden yüklendi: {home_piper_exe}")
            return home_piper_exe
        
        # Varsayılan uygulama konumuna dön
        piper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "piper")
        piper_exe = os.path.join(piper_dir, "piper.exe" if sys.platform == 'win32' else "piper")
        
        if os.path.exists(piper_exe):
            self.log(f"Piper uygulama dizininden yüklendi: {piper_exe}")
            return piper_exe
        
        # Hiçbir konum başarılı değilse, hata bildir ama bir değer döndür
        self.log(f"UYARI: Piper çalıştırılabilir dosyası bulunamadı! Aranan konumlar: {user_piper_exe}, {home_piper_exe}, {piper_exe}")
        return user_piper_exe  # Bile bulunamasa, ilk tercih edilen konumu döndür
    

    # In deskguiapp.py, inside DeskGUI class:
    def speak_text_locally(self, text):
        """Metni TTS motoruyla seslendir - ana işlemi arka planda yapar"""
        try:
            if not text or len(text) == 0:
                self.log("Boş metin seslendirilemez")
                self._on_tts_error("Boş metin")
                return False
            
            clean_text = re.sub(r"\[cmd:[a-zA-Z0-9_]+\]", "", text).strip()
            clean_text = self.clean_text_for_tts(clean_text)
            
            if not clean_text:
                self.log("Temizleme sonrası seslendirilecek metin kalmadı.")
                self._on_tts_error("Temiz metin boş")
                return False
            
            if not hasattr(self, 'tts_engine_type') or not self.tts_engine_type:
                # Varsayılan bir motor ayarla, örneğin piper
                self.log("Uyarı: TTS motor türü ayarlanmamış, varsayılan olarak 'piper' kullanılıyor.")
                self.tts_engine_type = "piper" 
            
            # self.speaking_active ve self.update_audio_status_signal çağrıları
            # her bir _speak_with_... metodunun kendi içinde yapılmalı (başlangıçta).
            # pub.sendMessage('tts:speaking', message=clean_text) de orada yapılmalı.

            if self.tts_engine_type == "piper":
                return self._speak_with_piper(clean_text)
            elif self.tts_engine_type == "gtts":
                return self._speak_with_gtts(clean_text)
            elif self.tts_engine_type == "espeak":
                return self._speak_with_espeak(clean_text)
            elif self.tts_engine_type == "pyttsx3":
                return self._speak_with_pyttsx3(clean_text)
            elif self.tts_engine_type == "xtts": # <<< GÜNCELLENDİ
                return self._speak_with_xtts(clean_text)
            else:
                # Bu log mesajı hatanın kaynağını gösteriyor
                self.log(f"Bilinmeyen veya desteklenmeyen TTS motoru ({self.tts_engine_type}) için işlem başlatılmıyor.")
                self._on_tts_error(f"Bilinmeyen TTS: {self.tts_engine_type}")
                return False
            
        except Exception as e:
            self.log(f"TTS işlemi başlatılırken genel hata: {e}")
            self.log(traceback.format_exc())
            self._on_tts_error(f"Genel TTS hatası: {str(e)}")
            return False

    def _speak_with_piper(self, text):
        """Piper TTS kullanarak konuşma - thread güvenli"""
        # İşlemi arka planda çalıştıracak bir fonksiyon tanımlayalım
        def _piper_worker_thread(text, callback_done, callback_error):
            try:
                # Piper çalıştırılabilir dosyası var mı kontrol et
                if not self.piper_executable or not os.path.exists(self.piper_executable):
                    raise FileNotFoundError(f"Piper çalıştırılabilir dosyası bulunamadı: {self.piper_executable}")
                
                # Model dosyası kontrolü
                if not os.path.exists(self.current_tts_voice):
                    raise FileNotFoundError(f"Piper model dosyası bulunamadı: {self.current_tts_voice}")
                    
                # Ses dosyası için geçici dosya oluştur
                temp_dir = tempfile.gettempdir()
                temp_file = os.path.join(temp_dir, f"tts_temp_{int(time.time())}.wav")
                
                # Metin dosyasını oluştur
                text_file = os.path.join(temp_dir, f"tts_text_{int(time.time())}.txt")
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                # Piper dizinini al
                piper_dir = os.path.dirname(self.piper_executable)
                piper_exe = os.path.basename(self.piper_executable)
                model_path = self.current_tts_voice
                
                # PowerShell komutunu oluştur
                ps_cmd = f'powershell -Command "cd \'{piper_dir}\'; cat \'{text_file}\' | .\\{piper_exe} -m \'{model_path}\' -f \'{temp_file}\'"'
                
                # Komutu arka planda çalıştır
                result = os.system(ps_cmd)
                
                # Kod 0 değilse hata var demektir
                if result != 0:
                    # Alternatif komut satırı komutu dene
                    cmd_cmd = f'cd /d "{piper_dir}" && type "{text_file}" | "{piper_exe}" -m "{model_path}" -f "{temp_file}"'
                    result = os.system(cmd_cmd)
                
                # WAV dosyası oluşturulmuş mu kontrol et
                if not os.path.exists(temp_file) or os.path.getsize(temp_file) < 100:
                    raise Exception("Piper ses dosyası oluşturulamadı")
                
                # Ses dosyasını çal - ana thread'e sinyal gönder
                # GUI'yi dondurmadan PyQt sinyali kullan
                QtCore.QMetaObject.invokeMethod(self, "_play_audio_file", 
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, temp_file))
                
                # İşlem başarılı sinyali gönder
                callback_done()
                
                # Geçici dosyaları temizle (oynatma bittikten sonra) - bu işlem bir süre bekletilmeli
                time.sleep(2)
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    if os.path.exists(text_file):
                        os.remove(text_file)
                except:
                    pass
                
            except Exception as e:
                # Hata durumunda callback ile bildir
                callback_error(str(e))
        
        # Thread-safe callback fonksiyonları
        def on_done():
            QtCore.QMetaObject.invokeMethod(self, "_on_tts_completed", 
                QtCore.Qt.QueuedConnection)
        
        def on_error(error_msg):
            QtCore.QMetaObject.invokeMethod(self, "_on_tts_error", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, error_msg))
        
        # UI'ı güncelle - konuşma başladı
        self.speaking_active = True
        self.update_audio_status_signal.emit("speaking", f"Konuşma sentezleniyor...")
        
        # İşlemi ayrı bir thread'de başlat
        threading.Thread(
            target=_piper_worker_thread, 
            args=(text, on_done, on_error), 
            daemon=True
        ).start()
        
        # Hemen True dön, işlem arka planda devam edecek
        return True
    

# In desk_gui_app.py, inside DeskGUI class:
# Mevcut _speak_with_xtts fonksiyonunuzun başındaki değişken tanımlamaları aynı kalsın.
# Sadece _xtts_worker_thread fonksiyonunu aşağıdak
    def _speak_with_xtts(self, text):
        """XTTS API sunucusu ile konuşma sentezler ve oynatır."""
        api_url = getattr(self, 'xtts_api_url', None)
        if not api_url:
            self._on_tts_error("XTTS API URL'si (self.xtts_api_url) yapılandırılmamış.")
            return False

        speaker_wav_for_api = getattr(self, 'xtts_speaker_wav', "C:/Users/emirh/xTTS/test2.wav") # Varsayılan
        xtts_cwd = os.path.dirname(speaker_wav_for_api) # Genellikle speaker_wav ile aynı dizinde olur

        if not os.path.isabs(speaker_wav_for_api):
             # Eğer xtts_cwd ayarlanmışsa ve speaker_wav_for_api göreceli ise, birleştir
            if hasattr(self, 'xtts_cwd') and self.xtts_cwd:
                 speaker_wav_for_api = os.path.join(self.xtts_cwd, speaker_wav_for_api)
            else: # Değilse, mevcut çalışma dizinine göreceli olduğunu varsay
                 speaker_wav_for_api = os.path.abspath(speaker_wav_for_api)

        if not os.path.exists(speaker_wav_for_api):
            self._on_tts_error(f"XTTS referans ses dosyası API için bulunamadı: {speaker_wav_for_api}")
            return False

        current_lang_from_ui = self.current_tts_lang
        language_to_send = "tr" # Varsayılan olarak Türkçe gönder

        if current_lang_from_ui and isinstance(current_lang_from_ui, str):
            supported_xtts_simple_codes = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']
            simple_lang_code = current_lang_from_ui.split('-')[0].lower()
            if simple_lang_code in supported_xtts_simple_codes:
                language_to_send = simple_lang_code
                self.log(f"XTTS için dil kodu '{current_lang_from_ui}' -> '{language_to_send}' olarak ayarlandı.")
            else:
                self.log(f"UYARI: UI'dan gelen dil '{current_lang_from_ui}' (basit hali '{simple_lang_code}') XTTS tarafından desteklenmiyor. Varsayılan '{language_to_send}' kullanılacak.")
        else:
            self.log(f"UYARI: Geçerli bir dil kodu UI'dan alınamadı. Varsayılan '{language_to_send}' kullanılacak.")


        def _xtts_worker_thread(text_to_speak, api_url_worker, speaker_wav_path_worker, language_worker, callback_done, callback_error):
            self.log(f"XTTS API Worker: Başladı. URL: {api_url_worker}")
            try:
                if not text_to_speak.strip():
                    self.log("XTTS API Worker: Boş metin.")
                    callback_error("XTTS için seslendirilecek metin yok.")
                    return

                payload = {
                    "text": text_to_speak.strip(),
                    "speaker_wav": speaker_wav_path_worker,
                    "language": language_worker # Her zaman bir dil gönderiyoruz
                }
                self.log(f"XTTS API Worker: İstek gönderiliyor. Payload: text='{payload['text'][:30]}...', speaker='{os.path.basename(payload['speaker_wav'])}', lang='{payload['language']}'")

                headers = {'Content-Type': 'application/json'}
                # Timeout süresini artırabiliriz, XTTS sentezlemesi uzun sürebilir
                response = requests.post(api_url_worker, json=payload, headers=headers, timeout=300) # 5 dakika timeout

                self.log(f"XTTS API Worker: Yanıt alındı. Durum Kodu: {response.status_code}")

                if response.status_code == 200:
                    if 'audio/wav' in response.headers.get('Content-Type', '').lower():
                        audio_data = response.content
                        self.log(f"XTTS API Worker: Ses verisi ({len(audio_data)} bytes) başarıyla alındı.")

                        if len(audio_data) < 100: # Çok küçük ses dosyası genellikle hatadır
                            self.log("XTTS API Worker: Alınan ses verisi çok küçük, muhtemelen bir hata oluştu.")
                            callback_error("XTTS API'den geçersiz (çok küçük) ses verisi alındı.")
                            return

                        # Pygame mixer'ı burada, ses çalmadan hemen önce başlat/kontrol et
                        if not pygame.mixer.get_init():
                            # XTTS genellikle 24kHz mono ses üretir.
                            pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024) # buffer boyutunu küçülttüm
                            self.log("XTTS API Worker: Pygame mixer (24kHz) başlatıldı.")

                        if pygame.mixer.music.get_busy():
                            self.log("XTTS API Worker: Önceki müzik durduruluyor...")
                            pygame.mixer.music.stop() # Sadece stop yeterli olabilir
                            pygame.mixer.music.unload() # Unload etmek daha güvenli
                            time.sleep(0.2) # Kısa bir bekleme

                        self.log(f"XTTS API Worker: Ses çalınıyor...")
                        try:
                            audio_stream = io.BytesIO(audio_data)
                            pygame.mixer.music.load(audio_stream)
                            pygame.mixer.music.play()

                            while pygame.mixer.music.get_busy():
                                QtCore.QCoreApplication.processEvents() # GUI'nin donmasını engelle
                                time.sleep(0.01) # CPU kullanımını azaltmak için kısa bekleme

                            pygame.mixer.music.unload() # Çalma bittikten sonra kaynağı serbest bırak
                            self.log("XTTS API Worker: Tüm ses çalındı.")
                            callback_done()

                        except Exception as play_exc:
                            self.log(f"XTTS API Worker: Ses çalınırken hata: {play_exc}")
                            self.log(traceback.format_exc())
                            callback_error(f"XTTS ses çalma hatası: {str(play_exc)}")
                    else:
                        err_msg = f"XTTS API'den beklenmeyen Content-Type: {response.headers.get('Content-Type')}. Yanıt: {response.text[:200]}"
                        self.log(err_msg)
                        callback_error(err_msg)
                else:
                    try:
                        error_details = response.json() # API'den JSON formatında hata detayı bekliyoruz
                        detail_msg = error_details.get("detail", response.text[:500]) # 1.py'deki ErrorResponse modeline uygun
                        err_msg = f"XTTS API Hatası (Kod: {response.status_code}): {detail_msg}"
                    except ValueError: # JSON parse edilemezse
                        err_msg = f"XTTS API Hatası (Kod: {response.status_code}): {response.text[:500]}"
                    self.log(err_msg)
                    callback_error(err_msg)

            except requests.exceptions.Timeout:
                self.log(f"XTTS API Worker: İstek zaman aşımına uğradı ({api_url_worker}).")
                callback_error(f"XTTS API isteği ({os.path.basename(api_url_worker)}) zaman aşımına uğradı.")
            except requests.exceptions.ConnectionError:
                self.log(f"XTTS API Worker: Bağlantı hatası ({api_url_worker}). Sunucu çalışıyor mu?")
                callback_error(f"XTTS API sunucusuna bağlanılamadı ({os.path.basename(api_url_worker)}).")
            except Exception as e:
                self.log(f"XTTS API Worker: Genel hata: {e}")
                self.log(traceback.format_exc())
                callback_error(f"XTTS API worker hatası: {str(e)}")
            self.log("XTTS API Worker: Bitti.")

        # --- Geri arama fonksiyonları ve thread başlatma kısmı aynı kalabilir ---
        def on_done_xtts():
            # Bu _on_tts_completed QT slotuna bağlanır
            QtCore.QMetaObject.invokeMethod(self, "_on_tts_completed", QtCore.Qt.QueuedConnection)

        def on_error_xtts(error_msg):
            # Bu _on_tts_error QT slotuna bağlanır
            QtCore.QMetaObject.invokeMethod(self, "_on_tts_error", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, error_msg))

        self.speaking_active = True
        self.update_audio_status_signal.emit("speaking", "XTTS API Sentezleniyor...")
        pub.sendMessage('tts:speaking', message=text) # AudioManager veya diğer dinleyiciler için

        threading.Thread(
            target=_xtts_worker_thread,
            args=(text, api_url, speaker_wav_for_api, language_to_send, on_done_xtts, on_error_xtts),
            daemon=True
        ).start()

        return True

    @QtCore.pyqtSlot(str)
    def _play_audio_file(self, audio_file):
        """UI thread'i içinde ses dosyasını bloklamadan çal"""
        try:
            # ÖNCE mevcut timer'ı durdur ve temizle
            if hasattr(self, 'play_timer') and self.play_timer:
                if self.play_timer.isActive():
                    self.play_timer.stop()
                # Sinyal bağlantısını kesmeyi dene (güvenlik için)
                try:
                    self.play_timer.timeout.disconnect()
                except TypeError: # Bağlı olmayabilir
                    pass
                # Eski timer referansını temizle
                self.play_timer = None

            # Pygame mixer başlatma
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

            # Varsa önceki müziği durdur ve kaldır
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                time.sleep(0.1) # Kısa bekleme

            # Ses dosyasını yükle ve çal
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            # Timer kullanarak müzik bitimini kontrol et
            def check_if_done():
                # Timer hala var mı ve aktif mi kontrol et
                if not hasattr(self, 'play_timer') or not self.play_timer or not self.play_timer.isActive():
                    return # Timer durduruldu veya silindi

                if not pygame.mixer.music.get_busy():
                    # Müzik bitti, timer'ı durdur
                    self.play_timer.stop()
                    # TTS tamamlandı sinyali gönder (ana thread'de çalışır)
                    self._on_tts_completed()
                    # Timer referansını temizle
                    self.play_timer = None
                # else:
                    # UI yenileme (genellikle gereksiz)
                    # QtWidgets.QApplication.processEvents()

            # Yeni timer oluştur
            self.play_timer = QtCore.QTimer(self) # self ile ilişkilendir
            self.play_timer.timeout.connect(check_if_done)
            self.play_timer.start(100)  # 100ms'de bir kontrol et

            # UI güncelleme
            self.update_audio_status_signal.emit("speaking", f"Ses çalınıyor...")

        except Exception as e:
            self.log(f"Ses dosyası çalınırken hata: {e}")
            # Hata durumunda timer'ı durdur
            if hasattr(self, 'play_timer') and self.play_timer and self.play_timer.isActive():
                self.play_timer.stop()
                self.play_timer = None
            # Hata sinyali gönder
            self._on_tts_error(str(e))
    
    # TTS tamamlandı
    @QtCore.pyqtSlot()
    def _on_tts_completed(self):
        """TTS tamamlandığında çağrılır"""
        # Timer hala aktifse durdur (check_if_done çalışmamış olabilir)
        if hasattr(self, 'play_timer') and self.play_timer and self.play_timer.isActive():
            self.play_timer.stop()
            self.play_timer = None

        self.speaking_active = False
        self.update_audio_status_signal.emit("idle", "TTS tamamlandı")
        pub.sendMessage('tts:complete')
        # Robot'a bildirim (isteğe bağlı)
        # if self.command_sender and self.command_sender.connected:
        #     self.command_sender.send_command('audio_event', {'type': 'tts_end'})
            
    # TTS hatası
    @QtCore.pyqtSlot(str)
    def _on_tts_error(self, error_msg):
        """TTS hatası alındığında çağrılır"""
        self.speaking_active = False
        self.update_audio_status_signal.emit("error", f"TTS hatası: {error_msg}")
        self.log(f"TTS hatası: {error_msg}")
        pub.sendMessage('tts:error', error_msg=error_msg)
    
    def _speak_with_gtts(self, text):
        """Google TTS (gTTS) kullanarak konuşma - thread güvenli versiyon"""
        # Arka planda çalışacak fonksiyon
        def _gtts_worker_thread(text, callback_done, callback_error):
            try:
                # Ses dosyası için geçici dosya oluştur
                temp_dir = tempfile.gettempdir()
                temp_file = os.path.join(temp_dir, f"gtts_temp_{int(time.time())}.mp3")
                
                # Dil kodunu belirle
                lang_code = self.current_tts_lang if hasattr(self, 'current_tts_lang') else 'tr'
                if '-' in lang_code:
                    lang_code = lang_code.split('-')[0]  # "tr-TR" -> "tr"
                
                # gTTS ile ses oluştur - bu uzun sürebilir
                tts = gtts.gTTS(text, lang=lang_code)
                tts.save(temp_file)
                
                # Ana thread'e ses dosyasını çalmak için sinyal gönder
                QtCore.QMetaObject.invokeMethod(self, "_play_audio_file", 
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, temp_file))
                
                # Başarılı sinyal
                callback_done()
                
                # Ses dosyası için bir temizleme beklemesi
                time.sleep(2)
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
                    
            except Exception as e:
                # Hata durumunda
                callback_error(str(e))
        
        # Thread-safe callback'ler
        def on_done():
            QtCore.QMetaObject.invokeMethod(self, "_on_tts_completed", 
                QtCore.Qt.QueuedConnection)
        
        def on_error(error_msg):
            QtCore.QMetaObject.invokeMethod(self, "_on_tts_error", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, error_msg))
                
        # UI güncellemesi
        self.speaking_active = True
        self.update_audio_status_signal.emit("speaking", f"Google TTS hazırlanıyor...")
        
        # Ayrı thread'de işlemi başlat
        threading.Thread(
            target=_gtts_worker_thread,
            args=(text, on_done, on_error),
            daemon=True
        ).start()
        
        return True
     
    def _speak_with_espeak(self, text):
        """eSpeak kullanarak konuşma - thread güvenli versiyon"""
        def _espeak_worker_thread(text, callback_done, callback_error):
            try:
                # espeak komutunu oluştur
                cmd = ["espeak"]
                
                # Dil seçeneği ekle
                lang_code = self.current_tts_lang if hasattr(self, 'current_tts_lang') else 'tr'
                cmd.extend(["-v", lang_code])
                
                # Hız parametresi ekle (espeak'te hız 80-500 arasında, 175 varsayılan)
                if hasattr(self, 'tts_speed'):
                    speed_value = int(175 * self.tts_speed)
                    cmd.extend(["-s", str(speed_value)])
                    
                # Çıktıyı bir ses dosyasına yönlendir
                temp_dir = tempfile.gettempdir()
                temp_file = os.path.join(temp_dir, f"espeak_temp_{int(time.time())}.wav")
                cmd.extend(["-w", temp_file])
                    
                # Metni ekle
                cmd.append(text)
                
                # Süreci başlat ve tamamlanmasını bekle
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                _, stderr = process.communicate()
                
                if process.returncode != 0:
                    raise Exception(f"eSpeak hata kodu: {process.returncode}, hata: {stderr.decode('utf-8')}")
                    
                # Ses dosyasını çalmak için ana thread'e sinyal gönder
                QtCore.QMetaObject.invokeMethod(self, "_play_audio_file", 
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, temp_file))
                    
                # Başarılı sinyal
                callback_done()
                
                # Temizlik için bekle
                time.sleep(2)
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
                    
            except Exception as e:
                callback_error(f"eSpeak hatası: {str(e)}")
        
        # Thread-safe callback'ler
        def on_done():
            QtCore.QMetaObject.invokeMethod(self, "_on_tts_completed", 
                QtCore.Qt.QueuedConnection)
        
        def on_error(error_msg):
            QtCore.QMetaObject.invokeMethod(self, "_on_tts_error", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, error_msg))
        
        # UI güncelleme
        self.speaking_active = True
        self.update_audio_status_signal.emit("speaking", f"eSpeak hazırlanıyor...")
        
        # Ayrı thread'de başlat
        threading.Thread(
            target=_espeak_worker_thread,
            args=(text, on_done, on_error),
            daemon=True
        ).start()
        
        return True
    
    def _speak_with_pyttsx3(self, text):
        """pyttsx3 kullanarak konuşma - thread güvenli versiyon"""
        def _pyttsx3_worker_thread(text, callback_done, callback_error):
            try:
                # Ses dosyası için geçici dosya oluştur
                temp_dir = tempfile.gettempdir()
                temp_file = os.path.join(temp_dir, f"pyttsx3_temp_{int(time.time())}.wav")
                
                # pyttsx3 motorunu başlat
                engine = pyttsx3.init()
                
                # Hız ayarını uygula
                if hasattr(self, 'tts_speed'):
                    engine.setProperty('rate', int(engine.getProperty('rate') * self.tts_speed))
                
                # Ses dosyasına kaydet
                engine.save_to_file(text, temp_file)
                engine.runAndWait()
                
                # Dosya kontrolü
                if not os.path.exists(temp_file) or os.path.getsize(temp_file) < 100:
                    raise Exception("Ses dosyası oluşturulamadı")
                    
                # Ana thread'e ses çalma sinyali gönder
                QtCore.QMetaObject.invokeMethod(self, "_play_audio_file", 
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, temp_file))
                    
                # Başarılı sinyal
                callback_done()
                
                # Temizlik için bekle
                time.sleep(2)
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
                    
            except Exception as e:
                callback_error(f"pyttsx3 hatası: {str(e)}")
        
        # Thread-safe callback'ler
        def on_done():
            QtCore.QMetaObject.invokeMethod(self, "_on_tts_completed", 
                QtCore.Qt.QueuedConnection)
        
        def on_error(error_msg):
            QtCore.QMetaObject.invokeMethod(self, "_on_tts_error", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, error_msg))
        
        # UI güncelleme
        self.speaking_active = True
        self.update_audio_status_signal.emit("speaking", f"Sistem TTS hazırlanıyor...")
        
        # Ayrı thread'de başlat
        threading.Thread(
            target=_pyttsx3_worker_thread,
            args=(text, on_done, on_error),
            daemon=True
        ).start()
        
        return True
    
    def on_speed_changed(self, value):
        """TTS hızı değiştiğinde çağrılır"""
        # Değeri 0.7 ile 1.5 arasında bir ondalık sayıya dönüştür
        self.tts_speed = value / 100.0
        self.speed_value_label.setText(f"{self.tts_speed:.1f}x")
        self.log(f"TTS hızı {self.tts_speed:.1f}x olarak ayarlandı")
    

    
    def set_tts_voice(self, voice_id):
        """Set the TTS voice by ID"""
        if not self.tts_engine:
            return False
        
        try:
            self.tts_engine.setProperty('voice', voice_id)
            self.current_tts_voice = voice_id
            
            # Find voice info to log
            voice_name = voice_id
            for voices in self.tts_voices.values():
                for voice in voices:
                    if voice['id'] == voice_id:
                        voice_name = voice['name']
                        break
            
            self.log(f"Changed TTS voice to {voice_name}")
            return True
        except Exception as e:
            self.log(f"Error setting TTS voice: {e}")
            return False
    
    def get_voice_for_language(self, lang_code):
        """Get the best voice for a given language code"""
        if not lang_code or not self.tts_voices:
            return self.current_tts_voice
        
        # Try direct match
        if lang_code in self.tts_voices and self.tts_voices[lang_code]:
            return self.tts_voices[lang_code][0]['id']
        
        # Try language prefix match (e.g., 'en-US' -> 'en')
        lang_prefix = lang_code.split('-')[0]
        for code, voices in self.tts_voices.items():
            if code.startswith(lang_prefix) and voices:
                return voices[0]['id']
        
        # Return current voice if no match
        return self.current_tts_voice
    
    def handle_robot_tts_request(self, text, lang_code=None):
        """Handle TTS request from the robot"""
        if lang_code:
            # Temporarily switch to appropriate voice for this language
            original_voice = self.current_tts_voice
            lang_voice = self.get_voice_for_language(lang_code)
            self.set_tts_voice(lang_voice)
            
            # Speak text
            result = self.speak_text_locally(text)
            
            # Restore original voice
            self.set_tts_voice(original_voice)
            return result
        else:
            # Use current voice
            return self.speak_text_locally(text)
    
    # Yeni metod olarak ekleyin
    def toggle_auto_language(self, checked):
            """Otomatik dil tespiti özelliğini açıp kapatır"""
            self.use_auto_language = checked
            self.log(f"Otomatik dil tespiti: {'Aktif' if checked else 'Devre Dışı'}")


        # --- YENİ veya GÜNCELLENMİŞ SLOTLAR ---

    @QtCore.pyqtSlot(str)
    def update_wake_word_status_label(self, status_text):
        if hasattr(self, 'wake_word_status_label'):
            self.wake_word_status_label.setText(f"WW: {status_text}")
            # Renklendirme
            if "Dinliyor" in status_text or "Başlatılıyor" in status_text:
                self.wake_word_status_label.setStyleSheet("color: blue;")
            elif "Algılandı" in status_text:
                self.wake_word_status_label.setStyleSheet("color: green; font-weight: bold;")
            elif "Durdu" in status_text or "Durduruldu" in status_text or "Sonlandı" in status_text:
                self.wake_word_status_label.setStyleSheet("color: gray;")
            elif "Hata" in status_text:
                self.wake_word_status_label.setStyleSheet("color: red;")
            self.update_mic_button_and_indicator_based_on_state() # Merkezi fonksiyonu çağır
        # Mic butonunun metnini de burada senkronize edebiliriz, çünkü WW durumu değişti.
        # Bu, update_speech_status ile çakışabilir, dikkatli olmak lazım.
        # Belki ayrı bir UI güncelleme fonksiyonu daha iyi olur.
        # Şimdilik update_speech_status'un bunu hallettiğini varsayalım.
        # self.update_mic_button_and_indicator() # Böyle bir merkezi fonksiyon olabilir



    # AudioManager'dan WAKE_WORD_DETECTED_TOPIC mesajı geldiğinde
    @QtCore.pyqtSlot() # Eğer pubsub farklı thread'den emit ediyorsa, slot yapmak iyi olabilir
    def handle_wake_word_detection_event(self):
        self.log(">>> DeskGUI: WAKE WORD ALGILANDI (PubSub ile) - STT başlatılıyor...")
        self.speech_triggered_by_wake_word = True

        # Wake word dinleyici AudioManager içinde durdurulmuş olmalı.
        # STT'yi başlat
        success, msg = self.audio_manager.start_speech_recognition()
        self.speech_status_signal.emit(success, msg) # UI'ı güncelle

        if success:
            self.log("DeskGUI: Wake word sonrası STT başarıyla başlatıldı.")
            # 10 saniye sonra komut gelmezse STT'yi durdur ve WW'ye geri dön
            QtCore.QTimer.singleShot(10000, self.stop_stt_if_no_input_after_ww_timeout)
        else:
            self.log(f"HATA: DeskGUI: Wake word sonrası STT başlatılamadı: {msg}")
            self.speech_triggered_by_wake_word = False
            # STT başlayamadıysa ve WW checkbox işaretliyse, WW'yi tekrar başlatmayı dene
            if self.wake_word_enabled_by_checkbox:
                self.log("DeskGUI: STT başlatılamadığı için WW dinleyici yeniden başlatılıyor (checkbox aktif).")
                QtCore.QTimer.singleShot(500, self.audio_manager.start_wake_word_listener)

    def toggle_wake_word_checkbox(self, checked):
        # Eğer checkbox'ın mevcut durumu zaten 'checked' ile aynıysa,
        # bu programatik bir güncelleme olabilir, tekrar işlem yapma.
        if self.wake_word_enabled_by_checkbox == checked:
            # self.log(f"WW Checkbox: Durum zaten {'Aktif' if checked else 'Devre Dışı'}, işlem yok.")
            return

        self.log(f"WW Checkbox: Kullanıcı tarafından {'Aktif' if checked else 'Devre Dışı'} edildi.")
        self.wake_word_enabled_by_checkbox = checked

        if checked:
            self.log("WW Checkbox: WW başlatılıyor (STT kapalıysa).")
            if not self.audio_manager.speech_active:
                # WW başlatılmadan önce, STT'nin kapalı olduğundan emin olalım
                # (AudioManager bunu zaten kontrol ediyor ama çift kontrol)
                if self.audio_manager.speech_active:
                    self.log("UYARI: WW aktif edilirken STT hala açık görünüyor. STT durduruluyor.")
                    self.audio_manager.stop_speech_recognition(restart_wake_word_if_enabled=False)
                    # STT durduktan sonra WW'yi başlatmak için kısa bir gecikme gerekebilir.
                    # QtCore.QTimer.singleShot(100, self.audio_manager.start_wake_word_listener)
                    # Ancak AudioManager.start_wake_word_listener zaten STT aktifse başlamaz.
                self.audio_manager.start_wake_word_listener()
            else:
                self.log("WW Checkbox: STT aktif olduğu için WW şimdilik başlatılmıyor. STT kapanınca başlayacak.")
        else:
            self.log("WW Checkbox: WW durduruluyor.")
            self.audio_manager.stop_wake_word_listener()

        # Checkbox durumu değiştiğinde mikrofon butonunu ve göstergesini de güncelle
        # Bu, update_speech_status ile senkronize olmalı. Belki merkezi bir UI güncelleme fonksiyonu daha iyi olur.
        # Şimdilik update_speech_status'u çağırarak genel UI'ı senkronize edelim.
        # Ancak bu, update_speech_status'un toggle_mic_button'ı tekrar tetiklemesine neden olabilir.
        # Bu yüzden dikkatli olmalı veya ayrı bir UI güncelleme fonksiyonu yazılmalı.
        # self.update_speech_status(self.audio_manager.speech_active, "WW checkbox değişti") # Bu riskli olabilir
        # Bunun yerine, sadece mic_indicator ve toggle_mic_button'ı doğrudan güncelleyebiliriz:
        self.update_mic_button_and_indicator_based_on_state()

    # AudioManager'dan "am_stt_stopped_check_ww" mesajı geldiğinde
    def handle_am_stt_stopped_check_ww(self):
        self.log("DeskGUI: AudioManager'dan STT durdu, WW kontrol edilecek mesajı alındı.")
        if self.wake_word_enabled_by_checkbox and not self.audio_manager.is_wake_word_listening:
            self.log("DeskGUI: WW checkbox işaretli ve WW çalışmıyor, WW yeniden başlatılıyor.")
            self.audio_manager.start_wake_word_listener()
        else:
            self.log("DeskGUI: WW checkbox işaretli değil veya WW zaten çalışıyor, ek işlem yok.")


# deskguiapp.py - DeskGUI sınıfı içinde
    def stop_tts(self):
        """TTS konuşmasını anında durdurur."""
        try:
            if self.tts_engine_type == "piper":
                if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                # Piper'ın kendi _on_tts_completed çağrısını beklemesi gerekebilir veya
                # burada da speaking_active = False ve durum güncellemesi yapılabilir.
                # Şimdilik sadece müziği durduruyoruz.
                self._on_tts_completed() # Manuel olarak bitti sayalım
            elif self.tts_engine_type == "pyttsx3" and hasattr(self, 'tts_engine') and self.tts_engine:
                self.tts_engine.stop()
                self._on_tts_completed() # Manuel olarak bitti sayalım
            # Diğer TTS motorları için de (gTTS, espeak) pygame.mixer.music.stop() yeterli olacaktır.
            # ve _on_tts_completed çağrılmalı.
            else: # Diğer motorlar genellikle pygame ile çalar
                if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                self._on_tts_completed() # Manuel olarak bitti sayalım

            self.log("TTS konuşması durduruldu (kullanıcı tarafından).")
        except Exception as e:
            self.log(f"TTS durdurulurken hata: {e}")
            self.log(traceback.format_exc())
        # ... (create_control_panel çağrısından sonra LLM servis durumunu kontrol et)
        if hasattr(self, 'ollama_radio'): # Widget'lar oluşturulduktan sonra
            self.on_llm_service_changed(self.current_llm_service, True) # Başlangıç durumunu ayarla

    def create_control_panel(self):
        self.control_panel = QtWidgets.QWidget()
        # Use a VBox layout for the whole control panel
        main_control_layout = QtWidgets.QVBoxLayout()
        main_control_layout.setSpacing(10)
        main_control_layout.setContentsMargins(8, 8, 8, 8)

        # --- GUI Status Indicator (Moved inside control panel for better grouping) ---
        status_group = QtWidgets.QGroupBox("GUI Status")
        status_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Idle")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        main_control_layout.addWidget(status_group) # Add GUI status first
        self.stop_tts_button = QPushButton("TTS Durdur")
        self.stop_tts_button.setToolTip("Konuşmayı hemen kes")
        self.stop_tts_button.clicked.connect(self.stop_tts)
        main_control_layout.addWidget(self.stop_tts_button)
        # Servo animasyonları için ComboBox ve Buton ekle
        self.servo_anim_combo = QComboBox()
        self.servo_anim_combo.addItems(["BOUNCE", "CELEBRATE", "HEAD_LEFT", "HEAD_NOD", "HEAD_NOD_ABS", "HEAD_RIGHT", "HEAD_SHAKE", "HEAD_SHAKE_ABS", "LOOK_DOWN", "LOOK_UP", "RAISED","SCAN", "SIT", "SLEEP"])
        self.servo_anim_combo.setToolTip("Manuel servo animasyonu seç")
        self.send_servo_anim_button = QPushButton("Servo Animasyonu Gönder")
        self.send_servo_anim_button.clicked.connect(self.send_manual_servo_animation)
        main_control_layout.addWidget(self.servo_anim_combo)
        main_control_layout.addWidget(self.send_servo_anim_button)
        # --- Connection Group ---
        connection_group = QtWidgets.QGroupBox("Robot Connection")
        connection_layout = QtWidgets.QGridLayout()
        self.ip_input = QtWidgets.QLineEdit(self.robot_ip)
        self.ip_input.setPlaceholderText("Robot IP")
        self.connect_button = QtWidgets.QPushButton("Connect to Robot")
        self.connect_button.clicked.connect(self.toggle_connection)
        # Add Port Inputs
        self.cmd_port_input = QtWidgets.QLineEdit(str(self.command_port))
        self.vid_port_input = QtWidgets.QLineEdit(str(self.video_port))
        self.gui_port_input = QtWidgets.QLineEdit(str(self.gui_listen_port))
        connection_layout.addWidget(QtWidgets.QLabel("Robot IP:"), 0, 0)
        connection_layout.addWidget(self.ip_input, 0, 1)
        connection_layout.addWidget(QtWidgets.QLabel("Cmd Port:"), 1, 0)
        connection_layout.addWidget(self.cmd_port_input, 1, 1)
        connection_layout.addWidget(QtWidgets.QLabel("Vid Port:"), 2, 0)
        connection_layout.addWidget(self.vid_port_input, 2, 1)
        connection_layout.addWidget(QtWidgets.QLabel("GUI Listen Port:"), 3, 0)
        connection_layout.addWidget(self.gui_port_input, 3, 1)
        connection_layout.addWidget(self.connect_button, 4, 0, 1, 2) # Span 2 columns
        connection_group.setLayout(connection_layout)
        main_control_layout.addWidget(connection_group)

        # --- Tab Widget for other controls ---
        self.tab_widget = QtWidgets.QTabWidget()

        # Tab 1: Robot Control
        robot_control_tab = QtWidgets.QWidget()
        rc_layout = QtWidgets.QVBoxLayout(robot_control_tab)
        rc_layout.setSpacing(8) # Spacing within the tab


        # Eye Color Control
        eye_group = QtWidgets.QGroupBox("Eye Color")
        eg_layout = QtWidgets.QHBoxLayout()
        self.eye_color_input = QtWidgets.QLineEdit("blue") # Default
        self.eye_color_input.setPlaceholderText("e.g., red, green, #FF00FF")
        self.set_eye_button = QtWidgets.QPushButton("Set Eye")
        self.set_eye_button.clicked.connect(self.set_robot_eye_color)
        self.set_eye_button.setEnabled(False) # Disable until connected
        eg_layout.addWidget(QtWidgets.QLabel("Color:"))
        eg_layout.addWidget(self.eye_color_input)
        eg_layout.addWidget(self.set_eye_button)
        eye_group.setLayout(eg_layout)
        rc_layout.addWidget(eye_group)

        # create_control_panel içinde "Robot Kontrol" kısmına ekle
        
        # Manuel Animasyon Paneli
        animation_group = QGroupBox("Manuel Animasyon")
        animation_layout = QGridLayout()
        animation_group.setLayout(animation_layout)
        
        # Animasyon seçicileri
        animation_layout.addWidget(QLabel("Animasyon:"), 0, 0)
        self.animation_combo = QComboBox()
        animation_layout.addWidget(self.animation_combo, 0, 1, 1, 2)
        
        # Tüm animasyonları ekle
        self.animations = [
            "RAINBOW", "RAINBOW_CYCLE", "SPINNER", "BREATHE", 
            "METEOR", "FIRE", "COMET", "WAVE", "PULSE", 
            "TWINKLE", "COLOR_WIPE", "RANDOM_BLINK", 
            "THEATER_CHASE", "SNOW", "ALTERNATING", 
            "GRADIENT", "BOUNCING_BALL", "RUNNING_LIGHTS", 
            "STACKED_BARS"
        ]
                
        for anim in self.animations:
            self.animation_combo.addItem(anim)
        
        # Renk seçiciler
        animation_layout.addWidget(QLabel("Renk 1:"), 1, 0)
        self.color_combo = QComboBox()
        animation_layout.addWidget(self.color_combo, 1, 1)
        
        # Arduino renkleri ekle
        arduino_colors = [
            "RED", "GREEN", "BLUE", "YELLOW", "PURPLE", 
            "CYAN", "WHITE", "ORANGE", "PINK", "GOLD", 
            "TEAL", "MAGENTA", "LIME", "SKY_BLUE", "NAVY", 
            "MAROON", "AQUA", "VIOLET", "CORAL", "TURQUOISE"
        ]
        
        for color in arduino_colors:
            self.color_combo.addItem(color)
        
        animation_layout.addWidget(QLabel("Renk 2:"), 1, 2)
        self.color2_combo = QComboBox()
        animation_layout.addWidget(self.color2_combo, 1, 3)
        
        for color in arduino_colors:
            self.color2_combo.addItem(color)
        
        # Tekrar sayısı
        animation_layout.addWidget(QLabel("Tekrar:"), 2, 0)
        self.repeat_spinner = QSpinBox()
        self.repeat_spinner.setMinimum(1)
        self.repeat_spinner.setMaximum(10)
        self.repeat_spinner.setValue(1)
        animation_layout.addWidget(self.repeat_spinner, 2, 1)
        
        # Gönder butonu
        self.send_animation_btn = QPushButton("Animasyonu Gönder")
        self.send_animation_btn.clicked.connect(self.send_manual_animation)
        animation_layout.addWidget(self.send_animation_btn, 2, 2, 1, 2)
        
        # Animasyon açıklaması
        self.animation_desc = QLabel("Açıklama: Lütfen bir animasyon seçin")
        animation_layout.addWidget(self.animation_desc, 3, 0, 1, 4)
        
        # Animation tips dictionary
        self.animation_tips = {
        "RAINBOW": "Tüm renkler aynı anda değişir (Renk seçimi gerektirmez)",
        "RAINBOW_CYCLE": "Renkler tek tek değişir (Renk seçimi gerektirmez)",
        "SPINNER": "İlerleme çubuğu şeklinde dönen animasyon",
        "BREATHE": "Rengin parlaklığı nefes alır gibi değişir",
        "METEOR": "Meteor yağmuru animasyonu", 
        "FIRE": "Ateş alevi animasyonu (Renk seçimi gerektirmez)",
        "COMET": "Kuyruklu yıldız animasyonu",
        "WAVE": "Dalga animasyonu (Renk seçimi gerektirmez)",
        "PULSE": "Nabız şeklinde yanıp sönme",
        "TWINKLE": "Yıldız parıltısı efekti",
        "COLOR_WIPE": "Renk silme efekti",
        "RANDOM_BLINK": "Rastgele yanıp sönme (Renk seçimi gerektirmez)",
        "THEATER_CHASE": "Tiyatro takip efekti",
        "SNOW": "Kar yağışı efekti",
        "ALTERNATING": "İki renk arasında geçiş (İkinci renk gerekli)",
        "GRADIENT": "Renk geçişi animasyonu (Renk seçimi gerektirmez)",
        "BOUNCING_BALL": "Zıplayan top efekti",
        "RUNNING_LIGHTS": "Koşan ışıklar efekti",
        "STACKED_BARS": "Yığılmış çubuklar efekti (Renk seçimi gerektirmez)"
        }
        
        self.animation_combo.currentTextChanged.connect(self.update_animation_desc)
        
        # Paneli yerleştir
        rc_layout.addWidget(animation_group)
       # State Control
        state_group = QtWidgets.QGroupBox("Robot State")
        sg_layout = QtWidgets.QHBoxLayout()
        self.state_combo = QtWidgets.QComboBox()
        # Add states - use constants if available, otherwise hardcode strings carefully
        # Ensure these strings EXACTLY match the ones expected/used by the robot's Personality class
        states = ["IDLE", "ALERT", "SLEEPING", "RESTING"] # Make sure these match robot Config/constants
        self.state_combo.addItems(states)
        self.state_combo.setEnabled(False) # Disable until connected
        self.set_state_button = QtWidgets.QPushButton("Set State")
        self.set_state_button.clicked.connect(self.set_robot_state)
        self.set_state_button.setEnabled(False)
        sg_layout.addWidget(QtWidgets.QLabel("State:"))
        sg_layout.addWidget(self.state_combo)
        sg_layout.addWidget(self.set_state_button)
        state_group.setLayout(sg_layout)
        rc_layout.addWidget(state_group)

        # Robot Movement Controls
        movement_group = QtWidgets.QGroupBox("Robot Movement")
        movement_layout = QtWidgets.QGridLayout()
        self.btn_up = QtWidgets.QPushButton("▲"); self.btn_down = QtWidgets.QPushButton("▼")
        self.btn_left = QtWidgets.QPushButton("◄"); self.btn_right = QtWidgets.QPushButton("►")
        self.btn_center = QtWidgets.QPushButton("■")
        # Connect signals
        self.btn_up.clicked.connect(lambda: self.send_servo_command('tilt', 10))
        self.btn_down.clicked.connect(lambda: self.send_servo_command('tilt', -10))
        self.btn_left.clicked.connect(lambda: self.send_servo_command('pan', 10))
        self.btn_right.clicked.connect(lambda: self.send_servo_command('pan', -10))
        self.btn_center.clicked.connect(self.center_servos)
        # Disable buttons initially
        self.btn_up.setEnabled(False); self.btn_down.setEnabled(False); self.btn_left.setEnabled(False); self.btn_right.setEnabled(False); self.btn_center.setEnabled(False)
        # Add to layout
        movement_layout.addWidget(self.btn_up, 0, 1); movement_layout.addWidget(self.btn_left, 1, 0)
        movement_layout.addWidget(self.btn_center, 1, 1); movement_layout.addWidget(self.btn_right, 1, 2)
        movement_layout.addWidget(self.btn_down, 2, 1)
        movement_group.setLayout(movement_layout)
        rc_layout.addWidget(movement_group)

        rc_layout.addStretch(1) # Push controls towards the top
        self.tab_widget.addTab(robot_control_tab, "Robot Control")

        # Tab 2: Vision Settings
        vision_tab = QtWidgets.QWidget()
        vision_layout = QtWidgets.QVBoxLayout(vision_tab)
        vision_layout.setSpacing(8)
        # Vision Processing Mode Group
        vision_group = QtWidgets.QGroupBox("Vision Processing Mode")
        vision_layout_group = QtWidgets.QVBoxLayout()
        self.face_recognition_radio = QtWidgets.QRadioButton("Face Recognition")
        self.face_find_radio = QtWidgets.QRadioButton("Face Finding (Location Only)")
        self.motion_detection_radio = QtWidgets.QRadioButton("Motion Detection")
        self.no_processing_radio = QtWidgets.QRadioButton("Processing Off")
        self.no_processing_radio.setChecked(True)
        self.tracking_checkbox = QtWidgets.QCheckBox("Enable Tracking")
        # Yeni radio button'ları ekleyin
        self.finger_tracking_radio = QtWidgets.QRadioButton("Finger Tracking")
        self.age_emotion_radio = QtWidgets.QRadioButton("Age/Emotion Detection")
        self.object_detection_radio = QtWidgets.QRadioButton("Object Detection")
        # Add widgets to group layout
        vision_layout_group.addWidget(self.face_recognition_radio)
        vision_layout_group.addWidget(self.face_find_radio)
        vision_layout_group.addWidget(self.motion_detection_radio)
        vision_layout_group.addWidget(self.finger_tracking_radio)
        vision_layout_group.addWidget(self.age_emotion_radio)
        vision_layout_group.addWidget(self.object_detection_radio)
        vision_layout_group.addWidget(self.no_processing_radio)
        vision_layout_group.addWidget(self.tracking_checkbox)
        vision_group.setLayout(vision_layout_group)
        vision_layout.addWidget(vision_group)

        # Camera View Group
        camera_group = QtWidgets.QGroupBox("Camera View")
        camera_layout = QtWidgets.QHBoxLayout()
        self.flip_h_checkbox = QtWidgets.QCheckBox("Flip Horizontal")
        self.flip_v_checkbox = QtWidgets.QCheckBox("Flip Vertical")
        self.flip_h_checkbox.toggled.connect(self.toggle_flip_horizontal)
        self.flip_v_checkbox.toggled.connect(self.toggle_flip_vertical)
        camera_layout.addWidget(self.flip_h_checkbox)
        camera_layout.addWidget(self.flip_v_checkbox)
        camera_group.setLayout(camera_layout)
        vision_layout.addWidget(camera_group)

        # Face Training Group
        train_group = QtWidgets.QGroupBox("Face Training")
        train_layout = QtWidgets.QHBoxLayout()
        self.train_button = QtWidgets.QPushButton("Train Model")
        self.add_face_button = QtWidgets.QPushButton("Add Face from Video")
        self.train_button.clicked.connect(self.train_model)
        self.add_face_button.clicked.connect(self.add_face_from_frame)
        # Disable initially? Only enable if video is active?
        self.add_face_button.setEnabled(False)
        train_layout.addWidget(self.add_face_button)
        train_layout.addWidget(self.train_button)
        train_group.setLayout(train_layout)
        vision_layout.addWidget(train_group)

        vision_layout.addStretch(1) # Push controls up
        self.tab_widget.addTab(vision_tab, "Vision")

        # Tab 3: Audio Settings
        audio_tab = QtWidgets.QWidget()
        audio_layout = QtWidgets.QGridLayout(audio_tab)
        audio_layout.setSpacing(8)

        # TTS Engine
        tts_engine_label = QtWidgets.QLabel("TTS Engine:")
        self.tts_engine_combo = QtWidgets.QComboBox()
        self.tts_engine_combo.addItems([
            "Piper TTS (Yerel)",
            "gTTS (Google)",
            "pyttsx3 (Sistem)",
            "espeak (Yerel)",
            "xTTS (Yerel)"

        ])
        self.tts_engine_combo.currentIndexChanged.connect(self.on_tts_engine_changed)
        audio_layout.addWidget(tts_engine_label, 0, 0)
        audio_layout.addWidget(self.tts_engine_combo, 0, 1)

        # TTS Language
        tts_lang_label = QtWidgets.QLabel("TTS Language:")
        self.tts_language_combo = QtWidgets.QComboBox()
        self.populate_tts_languages() # Populate
        self.tts_language_combo.currentIndexChanged.connect(self.on_tts_language_changed)
        audio_layout.addWidget(tts_lang_label, 1, 0)
        audio_layout.addWidget(self.tts_language_combo, 1, 1)

        # TTS Voice
        voice_label = QtWidgets.QLabel("TTS Voice:")
        self.voice_combo = QtWidgets.QComboBox()
        self.populate_voice_combo() # Populate
        self.voice_combo.currentIndexChanged.connect(self.on_voice_changed)
        audio_layout.addWidget(voice_label, 2, 0)
        audio_layout.addWidget(self.voice_combo, 2, 1)

        # TTS Speed
        speed_label = QtWidgets.QLabel("TTS Speed:")
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setRange(70, 200); self.speed_slider.setValue(int(self.tts_speed * 100)); self.speed_slider.valueChanged.connect(self.on_speed_changed)
        self.speed_value_label = QtWidgets.QLabel(f"{self.tts_speed:.1f}x")
        speed_layout = QtWidgets.QHBoxLayout(); speed_layout.addWidget(self.speed_slider); speed_layout.addWidget(self.speed_value_label)
        audio_layout.addWidget(speed_label, 3, 0)
        audio_layout.addLayout(speed_layout, 3, 1)

        # Recognition Language
        recog_label = QtWidgets.QLabel("Recognition Lang:")
        self.language_combo = QtWidgets.QComboBox()
        # Add more languages as needed
        recog_langs = ["English (US) - en-US", "Turkish - tr-TR", "German - de-DE", "Spanish - es-ES", "French - fr-FR"] # Örnek liste
        self.language_combo.addItems(recog_langs)
        # Set initial selection based on self.audio_manager.language
        default_recog_lang_index = -1
        try: # Dil kodunu ayıklarken hata olabileceği için try-except
            current_lang_code = self.audio_manager.language
            for i, item in enumerate(recog_langs):
                if f" - {current_lang_code}" in item:
                    default_recog_lang_index = i
                    break
        except Exception as e:
            self.log(f"Error setting initial recog lang: {e}")

        if default_recog_lang_index >= 0:
             self.language_combo.setCurrentIndex(default_recog_lang_index)
        elif self.language_combo.count() > 0: # Eşleşme yoksa ilkini seç
            self.language_combo.setCurrentIndex(0)

        self.language_combo.currentIndexChanged.connect(self.change_speech_language) # Connect signal
        audio_layout.addWidget(recog_label, 4, 0)
        audio_layout.addWidget(self.language_combo, 4, 1)

        # Options Checkboxes
        self.ask_llm_checkbox = QtWidgets.QCheckBox("Send Speech to LLM"); self.ask_llm_checkbox.setChecked(True); self.ask_llm_checkbox.toggled.connect(self.toggle_speech_to_llm)
        self.auto_language_checkbox = QtWidgets.QCheckBox("Auto-Detect TTS Language"); self.auto_language_checkbox.setChecked(True); self.auto_language_checkbox.toggled.connect(self.toggle_auto_language)
        audio_layout.addWidget(self.ask_llm_checkbox, 5, 0, 1, 2)
        audio_layout.addWidget(self.auto_language_checkbox, 6, 0, 1, 2)


        # Layout'a ekleme (Satır numaralarını kendi kodunuza göre ayarlayın)
        current_row = 5 # Recog Lang'ın olduğu varsayılan satır + 1
        audio_layout.addWidget(self.ask_llm_checkbox, current_row, 0, 1, 2)
        current_row += 1
        audio_layout.addWidget(self.auto_language_checkbox, current_row, 0, 1, 2)
        current_row += 1
    
      


        # Buttons
        self.test_tts_button = QtWidgets.QPushButton("Test TTS")
        self.test_tts_button.clicked.connect(self.test_tts)

        # Yeni mikrofon butonunu ekleyin
        self.toggle_mic_button = QtWidgets.QPushButton("Mikrofonu Aç")
        self.toggle_mic_button.clicked.connect(self.toggle_microphone)
    

        # Button layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.test_tts_button)
        button_layout.addWidget(self.toggle_mic_button)
        audio_layout.addLayout(button_layout, current_row, 0, 1, 2)
        current_row += 1

        # Audio Status Panel (Added inside the tab)
        audio_status_group = QtWidgets.QGroupBox("Audio Status")
        audio_status_layout = QtWidgets.QGridLayout()
        self.mic_status_label = QtWidgets.QLabel("Mic:"); self.mic_indicator = QtWidgets.QLabel("●"); self.mic_indicator.setStyleSheet("color: gray; font-size: 18pt;")
        self.recognition_status_label = QtWidgets.QLabel("Recog:"); self.recognition_indicator = QtWidgets.QLabel("Inactive"); self.recognition_indicator.setStyleSheet("color: gray;")
        self.tts_status_label = QtWidgets.QLabel("TTS:"); self.tts_indicator = QtWidgets.QLabel("●"); self.tts_indicator.setStyleSheet("color: gray; font-size: 18pt;")
        self.audio_mode_label = QtWidgets.QLabel("Mode:"); self.audio_mode_indicator = QtWidgets.QLabel("Direct"); self.audio_mode_indicator.setStyleSheet("color: green;")
        self.audio_level_label = QtWidgets.QLabel("Level:"); self.audio_level_bar = QtWidgets.QProgressBar(); self.audio_level_bar.setMaximum(100); self.audio_level_bar.setValue(0); self.audio_level_bar.setTextVisible(False)
        # Apply initial style for progress bar
        self.audio_level_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid gray; border-radius: 3px;
                background-color: #f0f0f0; height: 10px; text-align: center;
            } QProgressBar::chunk { background-color: #4CAF50; width: 1px; }""")
        # Add to status layout
        audio_status_layout.addWidget(self.mic_status_label, 0, 0); audio_status_layout.addWidget(self.mic_indicator, 0, 1)
        audio_status_layout.addWidget(self.recognition_status_label, 1, 0); audio_status_layout.addWidget(self.recognition_indicator, 1, 1)
        audio_status_layout.addWidget(self.tts_status_label, 2, 0); audio_status_layout.addWidget(self.tts_indicator, 2, 1)
        audio_status_layout.addWidget(self.audio_mode_label, 3, 0); audio_status_layout.addWidget(self.audio_mode_indicator, 3, 1)
        audio_status_layout.addWidget(self.audio_level_label, 4, 0); audio_status_layout.addWidget(self.audio_level_bar, 4, 1)
        audio_status_group.setLayout(audio_status_layout)
        # Add status group to main audio tab layout
        audio_layout.addWidget(audio_status_group, current_row, 0, 1, 2)
        current_row += 1

        # Wake Word Checkbox
        self.wake_word_checkbox = QtWidgets.QCheckBox("Enable Wake Word")
        self.wake_word_checkbox.setChecked(False) # Başlangıçta kapalı
        self.wake_word_checkbox.toggled.connect(self.toggle_wake_word_checkbox)
        audio_layout.addWidget(self.wake_word_checkbox, current_row, 0, 1, 2)
        current_row += 1


        # Wake Word Status Label (Audio Status Group içine veya ayrı bir yere)
        if hasattr(self, 'audio_status_layout'): # Eğer audio_status_layout varsa
            self.wake_word_status_label = QtWidgets.QLabel("WW: Kapalı")
            self.wake_word_status_label.setStyleSheet("color: gray;")
            # audio_status_layout bir QGridLayout ise:
            # Mevcut satır sayısını alıp bir altına ekleyebiliriz
            next_audio_status_row = self.audio_status_layout.rowCount()
            self.audio_status_layout.addWidget(QtWidgets.QLabel("WakeWord:"), next_audio_status_row, 0)
            self.audio_status_layout.addWidget(self.wake_word_status_label, next_audio_status_row, 1)
        # ...
        # Add a stretch to push content up if needed (might not be necessary with grid layout)
        # audio_layout.setRowStretch(9, 1)

        audio_tab.setLayout(audio_layout)
        self.tab_widget.addTab(audio_tab, "Audio")

        # --- Tab 4: LLM & Gemini Ayarları (YENİ DÜZENLEME) ---
        llm_gemini_tab = QtWidgets.QWidget()
        llm_gemini_layout = QtWidgets.QVBoxLayout(llm_gemini_tab)

        # LLM Servis Seçimi
        llm_service_group = QtWidgets.QGroupBox("LLM Servisi Seçin")
        llm_service_h_layout = QtWidgets.QHBoxLayout()
        self.ollama_radio = QtWidgets.QRadioButton("Ollama")
        self.ollama_radio.setChecked(self.current_llm_service == "ollama")
        self.ollama_radio.toggled.connect(lambda checked: self.on_llm_service_changed("ollama", checked))
        llm_service_h_layout.addWidget(self.ollama_radio)

        self.gemini_radio = QtWidgets.QRadioButton("Gemini")
        self.gemini_radio.setChecked(self.current_llm_service == "gemini")
        self.gemini_radio.toggled.connect(lambda checked: self.on_llm_service_changed("gemini", checked))
        llm_service_h_layout.addWidget(self.gemini_radio)
        llm_service_group.setLayout(llm_service_h_layout)
        llm_gemini_layout.addWidget(llm_service_group)

        # Ollama Model Seçimi (Sadece Ollama seçiliyse görünür)
        self.ollama_model_group = QtWidgets.QGroupBox("Ollama Model Seçimi")
        ollama_model_group_layout = QtWidgets.QVBoxLayout()
        self.model_button_group = QtWidgets.QButtonGroup(self) # Ollama modelleri için
        ollama_models = [
            "SentryBOT:4b",
            "phi:latest",
            "gemma2:2b",
            "phi3.5:latest",
            "llama3.1:8b",
            "deepseek-r1:1.5b",
            "llama3.2:1b",
            "llama3.2:3b",  # Yeni eklenen model
            "gemma3:4b"
        ]
        default_ollama_model_found = False
        for m_idx, m_name in enumerate(ollama_models):
            btn = QtWidgets.QRadioButton(m_name)
            # self.ollama_model __init__ içinde ayarlanıyor.
            if m_name == self.ollama_model:
                btn.setChecked(True)
                default_ollama_model_found = True
            # Seçilince modeli güncelle (lambda ile isim capture)
            btn.toggled.connect(lambda checked, name=m_name: self.on_ollama_model_selected(name, checked))
            ollama_model_group_layout.addWidget(btn)
            self.model_button_group.addButton(btn, m_idx) # ID ile ekleyebiliriz
        if not default_ollama_model_found and ollama_model_group_layout.count() > 0:
            ollama_model_group_layout.itemAt(0).widget().setChecked(True)
            self.ollama_model = ollama_models[0] # Varsayılanı güncelle
        self.ollama_model_group.setLayout(ollama_model_group_layout)
        llm_gemini_layout.addWidget(self.ollama_model_group)

        # Gemini Ayarları Butonu (Sadece Gemini seçiliyse görünür)
        self.gemini_settings_button_llm_tab = QtWidgets.QPushButton("Gemini API Ayarları")
        self.gemini_settings_button_llm_tab.clicked.connect(self.show_gemini_settings_menu)
        llm_gemini_layout.addWidget(self.gemini_settings_button_llm_tab)
        
        llm_gemini_layout.addStretch(1)
        self.tab_widget.addTab(llm_gemini_tab, "LLM & Gemini")
        
        # Başlangıçta doğru grubun görünürlüğünü ayarla
        self.ollama_model_group.setVisible(self.current_llm_service == "ollama")
        self.gemini_settings_button_llm_tab.setVisible(self.current_llm_service == "gemini")
        # --- LLM & Gemini Ayarları SONU ---
        
        # Eski "LLM" ve "Gemini" tablarını kaldır (eğer varsa ve farklıysa)
        tabs_to_remove_names = ["LLM", "Gemini"]
        for i in range(self.tab_widget.count() -1, -1, -1): # Tersten iterate et ki indexler kaymasın
            tab_text = self.tab_widget.tabText(i)
            if tab_text in tabs_to_remove_names and self.tab_widget.widget(i) != llm_gemini_tab:
                old_tab_widget = self.tab_widget.widget(i)
                self.tab_widget.removeTab(i)
                if old_tab_widget:
                    old_tab_widget.deleteLater()
                self.log(f"Eski '{tab_text}' tabı kaldırıldı.")


        main_control_layout.addWidget(self.tab_widget)
        self.control_panel.setLayout(main_control_layout)

        # --- Connect Vision Radio/Checkbox Signals ---
        self.face_recognition_radio.toggled.connect(self.update_processing_mode)
        self.face_find_radio.toggled.connect(self.update_processing_mode)
        self.motion_detection_radio.toggled.connect(self.update_processing_mode)
        self.no_processing_radio.toggled.connect(self.update_processing_mode)
        self.tracking_checkbox.toggled.connect(self.update_tracking)
        self.finger_tracking_radio.toggled.connect(self.update_processing_mode)
        self.age_emotion_radio.toggled.connect(self.update_processing_mode)
        self.object_detection_radio.toggled.connect(self.update_processing_mode)

        # --- Initial Enable/Disable state based on connection ---
        self.update_ui_connection_state(False) # Start as disconnected
        self.create_vision_status_labels() # Vision status label'larını oluştur
        return self.control_panel # Bu satır önemli
        # DeskGUI sınıfı içine bu metodları ekleyin
    
    @pyqtSlot(str, bool)
    def on_llm_service_changed(self, service_name, checked):
        """LLM Servisi radyo butonları değiştiğinde çağrılır."""
        # Sadece seçilen radyo butonunun olayıyla ilgilen
        if not checked:
            return

        self.current_llm_service = service_name
        self.log(f"LLM Servisi şuna değiştirildi: {self.current_llm_service}")

        is_ollama = (self.current_llm_service == "ollama")
        is_gemini = (self.current_llm_service == "gemini")

        # Ollama model grubunun ve Gemini ayar butonunun görünürlüğünü yönet
        if hasattr(self, 'ollama_model_group'):
             self.ollama_model_group.setVisible(is_ollama)
             # Ollama model butonlarını da etkinleştir/devre dışı bırak
             if hasattr(self, 'model_button_group'):
                  for btn in self.model_button_group.buttons():
                      btn.setEnabled(is_ollama)

        if hasattr(self, 'gemini_settings_button_llm_tab'):
             self.gemini_settings_button_llm_tab.setVisible(is_gemini)

        # Gemini seçildiğinde ek kontroller yap
        if is_gemini:
            if not GEMINI_MODULE_AVAILABLE:
                QMessageBox.critical(self, "Gemini Modülü Eksik",
                                     "Gemini modülü (google.generativeai) yüklenemedi. Lütfen 'pip install google-generativeai' ile kurun.")
                # Eğer modül yoksa, Ollama'ya geri dön ve işlemi sonlandır
                if hasattr(self, 'ollama_radio'):
                    self.ollama_radio.setChecked(True)
                return # Önemli: İşlemi burada durdur

            # API anahtarı veya helper instance yoksa kullanıcıyı bilgilendir
            if not self.gemini_api_key or not self.gemini_helper_instance:
                self.log("Gemini seçildi, ancak API anahtarı ayarlanmamış veya yardımcı başlatılmamış. Ayarlar menüsünü açmayı düşünebilirsiniz.")
                # Kullanıcıyı doğrudan ayarlara yönlendirme:
                # QTimer.singleShot(100, self.show_gemini_settings_menu) # Bu, kullanıcıyı rahatsız edebilir

            # GeminiHelper instance'ının parametrelerini ayarlar menüsünden gelen değerlerle güncelle (varsa)
            if self.gemini_helper_instance:
                 try:
                     self.gemini_helper_instance.set_parameters(
                         temperature=getattr(self, 'gemini_temperature', 1.0),
                         top_k=getattr(self, 'gemini_top_k', 32),
                         top_p=getattr(self, 'gemini_top_p', 1.0),
                         safety_settings=getattr(self, 'gemini_safety_settings', None),
                         system_instruction=getattr(self, 'gemini_system_instruction', None)
                     )
                     self.log("GeminiHelper parametreleri güncellendi.")
                 except Exception as e:
                     self.log(f"GeminiHelper parametreleri güncellenirken hata: {e}")
                     # Hata olursa ne yapmalı? Belki bir uyarı gösterip yine de Gemini kullanmaya devam et?
                     # Veya sadece logla. Şu an sadece logluyoruz.

        # Ollama seçildiğinde modeli güncelle
        elif is_ollama:
            if hasattr(self, 'model_button_group') and self.model_button_group:
                 selected_ollama_radio = self.model_button_group.checkedButton()
                 if selected_ollama_radio:
                     self.ollama_model = selected_ollama_radio.text()
                     self.log(f"Ollama'ya geçildi. Mevcut Ollama modeli: {self.ollama_model}")
    def on_ollama_model_selected(self, model_name: str, checked: bool):
        """Bir Ollama model radyo butonu seçildiğinde güncelleme yap."""
        if not checked:
            return
        prev = getattr(self, 'ollama_model', None)
        self.ollama_model = model_name
        if prev != model_name:
            self.log(f"[ModelSelect] Ollama modeli değişti: {prev} -> {model_name}")
        else:
            self.log(f"[ModelSelect] Ollama modeli yeniden seçildi: {model_name}")
        # Anında doğrulama için seçili buton adını göster
        if hasattr(self, 'model_button_group'):
            checked_btn = self.model_button_group.checkedButton()
            if checked_btn:
                self.log(f"[ModelSelect] CheckedButton text: {checked_btn.text()}")
                 # else: # Eğer hiçbir Ollama modeli seçili değilse (bu durum olmamalı ama önlem)
                 #     if self.model_button_group.buttons():
                 #          first_ollama_button = self.model_button_group.buttons()[0]
                 #          first_ollama_button.setChecked(True)
                 #          self.ollama_model = first_ollama_button.text()
                 #          self.log(f"Ollama'ya geçildi. Varsayılan Ollama modeli ayarlandı: {self.ollama_model}")


    def show_gemini_settings_menu(self):
        """Gemini ayarları için bir PyQt5 ayar menüsü açar."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Gemini API Ayarları")
        layout = QFormLayout(dlg)

        # API Anahtarı Girişi
        # API anahtarını her seferinde yeniden oluşturmak yerine self'te tutalım
        # Dialog alanlarını da self'te tutarak değerlerin kalmasını sağlayabiliriz
        if not hasattr(self, '_gemini_api_key_input'):
            self._gemini_api_key_input = QLineEdit(self)
            self._gemini_api_key_input.setPlaceholderText("Gemini API Anahtarınızı buraya girin")
            self._gemini_api_key_input.setEchoMode(QLineEdit.Password)
        # Mevcut değeri göster
        self._gemini_api_key_input.setText(getattr(self, 'gemini_api_key', ''))
        layout.addRow("API Anahtarı:", self._gemini_api_key_input)

        # Gemini Model Seçimi
        if not hasattr(self, '_gemini_model_combo'):
            self._gemini_model_combo = QComboBox()
            # API'den çekmek ideal ama sabit liste de kullanılabilir
            gemini_models_available = [
                "gemini-2.5-flash-preview-04-17",
                "gemini-2.5-pro-preview-05-06",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                # Daha fazla model eklenebilir
            ]
            self._gemini_model_combo.addItems(gemini_models_available)
        # Mevcut modeli seç
        current_gem_model = getattr(self, 'gemini_model_name', 'gemini-1.5-pro-latest')
        index = self._gemini_model_combo.findText(current_gem_model)
        if index != -1:
            self._gemini_model_combo.setCurrentIndex(index)
        else:
            # Eğer kayıtlı model listede yoksa, listeye ekleyip seçin veya logla
            self.log(f"Kayıtlı Gemini modeli '{current_gem_model}' listede bulunamadı.")
            # İlkini seçebiliriz veya listeye ekleyebiliriz
            self._gemini_model_combo.insertItem(0, current_gem_model)
            self._gemini_model_combo.setCurrentIndex(0)


        layout.addRow("Gemini Model:", self._gemini_model_combo)

        # Diğer parametreler (Temperature, Top-K, Top-P, System Instruction, Safety Settings)
        if not hasattr(self, '_gemini_temp_spin'): self._gemini_temp_spin = QDoubleSpinBox(); self._gemini_temp_spin.setRange(0.0, 2.0); self._gemini_temp_spin.setSingleStep(0.01); self._gemini_temp_spin.setValue(getattr(self, 'gemini_temperature', 1.0))
        layout.addRow("Temperature:", self._gemini_temp_spin)

        if not hasattr(self, '_gemini_topk_spin'): self._gemini_topk_spin = QSpinBox(); self._gemini_topk_spin.setRange(1, 100); self._gemini_topk_spin.setValue(getattr(self, 'gemini_top_k', 32))
        layout.addRow("Top-K:", self._gemini_topk_spin)

        if not hasattr(self, '_gemini_topp_spin'): self._gemini_topp_spin = QDoubleSpinBox(); self._gemini_topp_spin.setRange(0.0, 1.0); self._gemini_topp_spin.setSingleStep(0.01); self._gemini_topp_spin.setValue(getattr(self, 'gemini_top_p', 1.0))
        layout.addRow("Top-P:", self._gemini_topp_spin)

        if not hasattr(self, '_gemini_instr_edit'): self._gemini_instr_edit = QTextEdit(); self._gemini_instr_edit.setPlaceholderText("İsteğe bağlı sistem talimatı"); self._gemini_instr_edit.setPlainText(getattr(self, 'gemini_system_instruction', ''))
        layout.addRow("System Instruction:", self._gemini_instr_edit)

        if not hasattr(self, '_gemini_safety_edit'): self._gemini_safety_edit = QTextEdit(); self._gemini_safety_edit.setPlaceholderText("Örn: [{'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_ONLY_HIGH'}]");
        current_safety_settings = getattr(self, 'gemini_safety_settings', None)
        if current_safety_settings is None:
            self._gemini_safety_edit.setPlainText("")
        elif isinstance(current_safety_settings, (list, dict)):
             try:
                self._gemini_safety_edit.setPlainText(json.dumps(current_safety_settings, indent=2, ensure_ascii=False))
             except Exception:
                self._gemini_safety_edit.setPlainText(str(current_safety_settings))
        else:
            self._gemini_safety_edit.setPlainText(str(current_safety_settings))
        layout.addRow("Safety Settings (JSON):", self._gemini_safety_edit)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addRow(btns)

        def on_accept():
            new_api_key = self._gemini_api_key_input.text().strip()
            new_model_name = self._gemini_model_combo.currentText()
            new_temperature = self._gemini_temp_spin.value()
            new_top_k = self._gemini_topk_spin.value()
            new_top_p = self._gemini_topp_spin.value()
            new_system_instruction = self._gemini_instr_edit.toPlainText().strip()

            # API anahtarını hemen kaydet
            self.gemini_api_key = new_api_key

            # API Anahtarı kontrolü
            if not self.gemini_api_key:
                QMessageBox.warning(dlg, "Eksik API Anahtarı", "Lütfen Gemini API anahtarınızı girin.")
                return # İşlemi durdur

            safety_json_text = self._gemini_safety_edit.toPlainText().strip()
            parsed_safety_settings = None
            if safety_json_text:
                try:
                    parsed_safety_settings = json.loads(safety_json_text)
                    # Gemini'nin beklediği formatta olup olmadığını doğrulamak iyi bir pratik olur.
                    # Örn: [{'category': HarmCategory, 'threshold': HarmBlockThreshold}]
                    # Ancak bu, genai.types importlarını gerektirir ve karmaşıklığı artırır.
                    # Şimdilik sadece JSON geçerliliğini kontrol ediyoruz.
                    # Genai'deki uygun tiplere dönüştürmek gerekebilir, şu an sadece dict/list formatında saklıyoruz.
                except json.JSONDecodeError as e:
                    QMessageBox.warning(dlg, "JSON Hatası", f"Safety ayarları geçerli bir JSON formatında olmalı: {e}")
                    return # İşlemi durdur
                except Exception as e:
                    QMessageBox.warning(dlg, "Safety Settings Hatası", f"Safety ayarları işlenirken hata: {e}")
                    return # İşlemi durdur

            # Tüm değerleri self'e kaydet
            self.gemini_model_name = new_model_name
            self.gemini_temperature = new_temperature
            self.gemini_top_k = new_top_k
            self.gemini_top_p = new_top_p
            self.gemini_system_instruction = new_system_instruction
            self.gemini_safety_settings = parsed_safety_settings # Parsed JSON objesi veya None

            # GeminiHelper instance'ını yeni ayarlarla yeniden başlat/güncelle
            if GEMINI_MODULE_AVAILABLE:
                try:
                    self.gemini_helper_instance = GeminiHelper(api_key=self.gemini_api_key, model=self.gemini_model_name)
                    self.gemini_helper_instance.set_parameters(
                        temperature=self.gemini_temperature,
                        top_k=self.gemini_top_k,
                        top_p=self.gemini_top_p,
                        safety_settings=self.gemini_safety_settings, # Bu dict/list olmalı
                        system_instruction=self.gemini_system_instruction if self.gemini_system_instruction else None
                    )
                    self.log(f"GeminiHelper '{self.gemini_model_name}' ile güncellendi/oluşturuldu.")
                    self.log(f"Ayarlar: Temp={self.gemini_temperature}, Top-K={self.gemini_top_k}, Top-P={self.gemini_top_p}, SysInstr set: {bool(self.gemini_system_instruction)}, Safety set: {bool(self.gemini_safety_settings)}")
                    dlg.accept() # Dialog'u kapat

                except ValueError as ve: # GeminiHelper __init__ içinden gelen API anahtarı hatası vb.
                    self.log(f"Gemini başlatılamadı: {ve}")
                    QtWidgets.QMessageBox.critical(dlg, "Gemini Başlatma Hatası", f"Gemini başlatılamadı. API anahtarınızı veya model adını kontrol edin:\n{ve}")
                    self.gemini_helper_instance = None # Hata durumunda instance'ı temizle
                except Exception as e:
                    self.log(f"GeminiHelper başlatılırken beklenmedik bir hata oluştu: {e}")
                    self.log(traceback.format_exc())
                    QtWidgets.QMessageBox.critical(dlg, "Gemini Hatası", f"GeminiHelper başlatılırken beklenmedik bir hata oluştu:\n{e}")
                    self.gemini_helper_instance = None # Hata durumunda instance'ı temizle
            else:
                QtWidgets.QMessageBox.critical(dlg, "Modül Hatası", "GeminiHelper modülü yüklenemedi.")


        btns.accepted.connect(on_accept)
        btns.rejected.connect(dlg.reject)
        dlg.exec_() # Dialog'u modal olarak göster


    def create_vision_status_labels(self):
        """Vision tabına durum etiketleri ekler."""
        try:
            # Vision tabını bul
            vision_tab_index = -1
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == "Vision":
                    vision_tab_index = i
                    break

            if vision_tab_index != -1:
                vision_tab = self.tab_widget.widget(vision_tab_index)
                vision_layout = vision_tab.layout() # QVBoxLayout olduğunu varsayıyoruz

                if vision_layout:
                    status_group = QGroupBox("Detection Status")
                    status_layout = QFormLayout()

                    self.gesture_status_label = QLabel("Gesture: -")
                    self.emotion_status_label = QLabel("Emotion: -")
                    self.object_status_label = QLabel("Object: -")

                    status_layout.addRow(self.gesture_status_label)
                    status_layout.addRow(self.emotion_status_label)
                    status_layout.addRow(self.object_status_label)
                    status_group.setLayout(status_layout)

                    # Stretch'ten önce ekle
                    vision_layout.insertWidget(vision_layout.count() - 1, status_group)
                    self.log("Vision status labels added.")
                else:
                    self.log("Vision tab layout not found.")
            else:
                self.log("Vision tab not found.")
        except Exception as e:
            self.log(f"Error adding vision status labels: {e}")

    # Opsiyonel: Sinyalleri handle edecek slotlar
    @pyqtSlot(str)
    def update_gesture_display(self, command):
        if hasattr(self, 'gesture_status_label'):
            self.gesture_status_label.setText(f"Gesture: {command}")

    @pyqtSlot(str)
    def update_emotion_display(self, emotion):
        if hasattr(self, 'emotion_status_label'):
            self.emotion_status_label.setText(f"Emotion: {emotion}")

    @pyqtSlot(str)
    def update_object_display(self, object_label):
         if hasattr(self, 'object_status_label'):
            self.object_status_label.setText(f"Object: {object_label}")

    # PubSub mesajlarını handle edecek metotlar
    def handle_gesture_command(self, command):
        # Sinyal ile UI güncellemesi
        if hasattr(self, 'update_gesture_label_signal'):
            self.update_gesture_label_signal.emit(command)
        self.log(f"Gesture detected: {command}") # Loglama

    def handle_emotion_detected(self, emotion):
        if hasattr(self, 'update_emotion_label_signal'):
            self.update_emotion_label_signal.emit(emotion)
        self.log(f"Emotion detected: {emotion}")

    def handle_object_detected(self, object_info):
        label = object_info.get('label', 'Unknown')
        confidence = object_info.get('confidence', 0)
        if hasattr(self, 'update_object_label_signal'):
            self.update_object_label_signal.emit(f"{label} ({confidence:.2f})")
        self.log(f"Object detected: {label} (Conf: {confidence:.2f})")

    def update_animation_desc(self, animation_name):
        """Seçilen animasyonun açıklamasını güncelle ve gerekli kontrolleri etkinleştir/devre dışı bırak"""
        if animation_name in self.animation_tips:
            self.animation_desc.setText(f"Açıklama: {self.animation_tips[animation_name]}")
        else:
            self.animation_desc.setText("Açıklama: Bu animasyon için açıklama yok")
        
        # Animasyon türüne göre renk seçicileri etkinleştir/devre dışı bırak
        no_color_animations = ["WAVE", "RAINBOW", "RAINBOW_CYCLE", "FIRE", "RANDOM_BLINK", "GRADIENT", "STACKED_BARS"]
        two_color_animations = ["ALTERNATING"]
        
        if animation_name in no_color_animations:
            self.color_combo.setEnabled(False)
            self.color2_combo.setEnabled(False)
            self.animation_desc.setText(f"Açıklama: {self.animation_tips.get(animation_name, 'Renk seçimi gerektirmeyen animasyon')}")
        elif animation_name in two_color_animations:
            self.color_combo.setEnabled(True)
            self.color2_combo.setEnabled(True)
            self.animation_desc.setText(f"Açıklama: {self.animation_tips.get(animation_name, 'İki renk gerektiren animasyon')}")
        else:
            self.color_combo.setEnabled(True)
            self.color2_combo.setEnabled(False)
    
    def send_manual_animation(self):
        """Seçilen animasyonu robota gönder"""
        if not self.command_sender or not self.command_sender.connected:
            self.log("Robot bağlı değil. Animasyon gönderilemiyor.")
            return
        
        animation = self.animation_combo.currentText()
        
        # Parametre ihtiyacına göre animasyonları grupla
        no_color_animations = ["WAVE", "RAINBOW", "RAINBOW_CYCLE", "FIRE", "RANDOM_BLINK", "GRADIENT", "STACKED_BARS"]
        two_color_animations = ["ALTERNATING"]
        
        # Varsayılan parametreleri hazırla
        params = {
            "animation": animation,
            "repeat": self.repeat_spinner.value()
        }
        
        # Animasyon türüne göre renk parametresi ekle
        if animation not in no_color_animations:
            color = self.color_combo.currentText()
            params["color"] = color
            
            # İki renk gerektiren animasyonlar için ikinci rengi ekle
            if animation in two_color_animations:
                color2 = self.color2_combo.currentText()
                params["color2"] = color2
        
        # Renk gerektiren/gerektirmeyen animasyonlar için log mesajını özelleştir
        if animation in no_color_animations:
            self.log(f"Animasyon gönderiliyor: {animation}, Tekrar: {params['repeat']}")
        elif animation in two_color_animations:
            self.log(f"Animasyon gönderiliyor: {animation}, Renk1: {params.get('color')}, Renk2: {params.get('color2')}, Tekrar: {params['repeat']}")
        else:
            self.log(f"Animasyon gönderiliyor: {animation}, Renk: {params.get('color')}, Tekrar: {params['repeat']}")
        
        try:
            response = self.command_sender.send_command("send_animation", params)
            self.log(f"Komut yanıtı: {response.get('status', 'unknown')} - {response.get('message', 'no message')}")
        except Exception as e:
            self.log(f"Animasyon gönderme hatası: {e}")
    
    def send_manual_servo_animation(self):
        anim = self.servo_anim_combo.currentText()
        self.send_animation(anim)
        self.log(f"Manuel servo animasyonu gönderildi: {anim}")


    def update_ui_connection_state(self, connected):
        """Enable/Disable UI elements based on connection status."""
        # Buttons in Robot Control Tab
        self.set_eye_button.setEnabled(connected)
        self.state_combo.setEnabled(connected)
        self.set_state_button.setEnabled(connected)
        # Movement buttons
        self.btn_up.setEnabled(connected); self.btn_down.setEnabled(connected)
        self.btn_left.setEnabled(connected); self.btn_right.setEnabled(connected)
        self.btn_center.setEnabled(connected)
        # Vision Tab Controls
        self.add_face_button.setEnabled(connected)
        self.tracking_checkbox.setEnabled(connected and VISION_MODULES_AVAILABLE) # Modül varsa etkinleştir
        self.face_recognition_radio.setEnabled(connected)
        self.face_find_radio.setEnabled(connected)
        self.motion_detection_radio.setEnabled(connected)
        self.no_processing_radio.setEnabled(connected)
        # Yeni radio button'lar
        self.finger_tracking_radio.setEnabled(connected and FingerTracking is not None)
        self.age_emotion_radio.setEnabled(connected and AgeEmotionDetector is not None)
        self.object_detection_radio.setEnabled(connected and ObjectDetector is not None)

        # Bağlantı kesildiğinde aktif modülleri durdur
        if not connected:
            if hasattr(self, 'processing_mode') and self.processing_mode != 'none':
                self.log("Disconnecting: Stopping active vision modules.")
                # Tüm potansiyel modülleri durdurmayı dene
                if hasattr(self, 'face_detector') and self.face_detector: self.face_detector.stop()
                if hasattr(self, 'motion_detector') and self.motion_detector: self.motion_detector.stop()
                if hasattr(self, 'finger_tracker') and self.finger_tracker: self.finger_tracker.stop()
                if hasattr(self, 'age_emotion_detector') and self.age_emotion_detector: self.age_emotion_detector.stop()
                if hasattr(self, 'object_detector') and self.object_detector: self.object_detector.stop()
                if hasattr(self, 'object_tracker') and self.object_tracker: self.object_tracker.stop()

                self.processing_mode = 'none'
                self.no_processing_radio.setChecked(True) # UI'ı da güncelle
                self.tracking_checkbox.setChecked(False)
        
        # Reset personality combo on disconnect
        if not connected:
             # Reset robot status display values
             self.robot_state = "Unknown"
             self.robot_eye_color = "Unknown"
             self.robot_personality = "Unknown"
             # Optionally reset state combo?
             # self.state_combo.setCurrentIndex(0) # Or find the index for a default state like IDLE

    def toggle_flip_horizontal(self, checked):
        """Toggle horizontal flipping of the camera image."""
        # Değişken yoksa oluştur
        if not hasattr(self, 'flip_horizontal'):
            self.flip_horizontal = False
        
        self.flip_horizontal = checked
        self.log(f"Yatay çevirme: {'Etkin' if checked else 'Devre dışı'}")
        
    def toggle_flip_vertical(self, checked):
        """Toggle vertical flipping of the camera image."""
        # Değişken yoksa oluştur
        if not hasattr(self, 'flip_vertical'):
            self.flip_vertical = False
        
        self.flip_vertical = checked
        self.log(f"Dikey çevirme: {'Etkin' if checked else 'Devre dışı'}")
        
    def apply_audio_settings(self):
        """Apply audio settings based on the selected radio button."""
        # Basitleştirilmiş seçim - kullanıcının tercihine göre ayarlanabilir
        # Bluetooth kullanmak istemediğiniz için direct mod kullanıyoruz
        mode = "direct"
        server = self.bluetooth_server
        
        # Bluetooth mod bayrağını güncelle
        self.using_bluetooth_audio = False  # Her zaman False olacak (yerel TTS kullanılacak)
        self.log(f"Audio mode set to: Direct (Local TTS)")
        
        # Get the selected language code
        lang_text = self.language_combo.currentText()
        lang_code = lang_text.split(" - ")[-1]
        
        try:
            self.audio_manager.initialize(mode, server, lang_code)
            self.log(f"Audio settings updated: direct mode")
            self.log(f"Speech recognition language set to: {lang_code}")
            
            # Audio başlatıldıktan sonra sesli tanımayı başlat
            QtCore.QTimer.singleShot(1000, self.start_speech_recognition)
            
            # Update audio mode indicator in the status panel
            self.update_audio_status_signal.emit("direct_mode", "Direct Mode (Local TTS)")
            
        except Exception as e:
            self.log(f"Error initializing audio: {e}")
        
        # If switching to Bluetooth, automatically check speech status
        if mode == "bluetooth":
            threading.Thread(target=self._check_remote_speech_status, daemon=True).start()
        
        # Update audio mode indicator in the status panel
        if mode == "bluetooth":
            self.update_audio_status_signal.emit("bluetooth_mode", "Bluetooth Mode")
        else:
            self.update_audio_status_signal.emit("direct_mode", "Direct Mode")
    
    def change_speech_language(self):
        """Change the speech recognition language."""
        lang_text = self.language_combo.currentText()
        lang_code = lang_text.split(" - ")[-1]
        self.audio_manager.change_language(lang_code)
    
    # Aşağıdaki metodu düzenlememiz gerekiyor
    
    def find_piper_executable(self):
        """İşletim sistemine göre uygun Piper çalıştırılabilir dosyasını bulur"""
        
        # 1. Kullanıcının belirlediği sabit konumu kontrol et (öncelikli)
        user_piper_dir = "C:\\Users\\emirh\\piper"
        user_piper_exe = os.path.join(user_piper_dir, "piper.exe")
        
        self.log(f"Piper aranıyor: {user_piper_exe}")
        
        if os.path.exists(user_piper_exe):
            self.log(f"Piper başarıyla bulundu: {user_piper_exe}")
            return user_piper_exe
        else:
            self.log(f"Piper belirtilen konumda bulunamadı: {user_piper_exe}")
        
        # 2. Ana kullanıcı dizininde piper klasörünü kontrol et
        home_dir = os.path.expanduser("~")
        home_piper_dir = os.path.join(home_dir, "piper")
        home_piper_exe = os.path.join(home_piper_dir, "piper.exe")
        
        self.log(f"Piper kullanıcı dizininde aranıyor: {home_piper_exe}")
        
        if os.path.exists(home_piper_exe):
            self.log(f"Piper kullanıcı dizininde bulundu: {home_piper_exe}")
            return home_piper_exe
        
        # 3. Uygulama dizinindeki piper klasörünü kontrol et
        app_piper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "piper")
        app_piper_exe = os.path.join(app_piper_dir, "piper.exe")
        
        self.log(f"Piper uygulama dizininde aranıyor: {app_piper_exe}")
        
        if os.path.exists(app_piper_exe):
            self.log(f"Piper uygulama dizininde bulundu: {app_piper_exe}")
            return app_piper_exe
        
        # 4. Hiçbir konumda bulunamadı - açık bir hata mesajıyla kullanıcıyı bilgilendir
        self.log("HATA: Piper çalıştırılabilir dosyası hiçbir konumda bulunamadı!")
        self.log(f"  - Aranan konumlar:")
        self.log(f"  - {user_piper_exe}")
        self.log(f"  - {home_piper_exe}")
        self.log(f"  - {app_piper_exe}")
        self.log("Lütfen Piper uygulamasını doğru konuma yerleştirdiğinizden emin olun.")
        
        # Hata durumunda başarısız olmasını sağlayalım ki kullanıcı hatayı görsün
        return ""  # Boş string döndürmek, dosyanın bulunamadığını açıkça belirtir
    
    def find_piper_voices(self):
        """Piper klasörü altındaki dil klasörlerindeki ses modellerini bulur"""
        voices = {}
        
        # Aranacak tüm piper konumları
        piper_dirs = [
            "C:\\Users\\emirh\\piper",  # Kullanıcının belirttiği sabit konum
            os.path.join(os.path.expanduser("~"), "piper"),  # Kullanıcı dizini
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "piper")  # Uygulama dizini
        ]
        
        found_models = False
        searched_paths = []
        
        # Tüm konumları kontrol et
        for piper_dir in piper_dirs:
            if not os.path.exists(piper_dir):
                searched_paths.append(f"{piper_dir} (bulunamadı)")
                continue
            
            searched_paths.append(f"{piper_dir} (kontrol edildi)")
            self.log(f"Piper ses modelleri bu konumda aranıyor: {piper_dir}")
            
            # Dil klasörlerini bul
            try:
                lang_dirs = os.listdir(piper_dir)
            except Exception as e:
                self.log(f"Dizin okunamadı: {piper_dir} - Hata: {e}")
                continue
                
            for lang_dir in lang_dirs:
                lang_path = os.path.join(piper_dir, lang_dir)
                
                # Sadece klasörleri işle ve gizli klasörleri atla
                if not os.path.isdir(lang_path) or lang_dir.startswith('.'):
                    continue
                    
                # Dil kodunu belirle (klasör adı)
                lang_code = lang_dir.lower()
                
                # Model dosyalarını ara
                model_found_for_lang = False
                
                for root, dirs, files in os.walk(lang_path):
                    for file in files:
                        if file.endswith(".onnx") or file.endswith(".onyx"):
                            # Dosya yolunu al
                            full_path = os.path.join(root, file)
                            model_found_for_lang = True
                            
                            # Dizin yapısından bilgileri çıkar
                            rel_path = os.path.relpath(full_path, lang_path)
                            parts = rel_path.split(os.path.sep)
                            
                            # Model bilgilerini oluştur
                            model_info = {}
                            model_info['id'] = full_path
                            
                            # Dosya adından bilgileri çıkar (örn: xyz-low.onnx)
                            file_parts = os.path.splitext(file)[0].split('-')
                            model_name = file_parts[0]
                            quality = file_parts[1] if len(file_parts) > 1 else "medium"
                            
                            # Ses adını oluştur
                            voice_name = f"{lang_code}_{model_name}_{quality}"
                            model_info['name'] = voice_name
                            model_info['gender'] = 'unknown'  # Piper modellerinde gender bilgisi yok
                            
                            # Dil kodu için ISO formatı (örn: tr-TR)
                            iso_lang = lang_code
                            if len(lang_code) == 2:  # 2 karakterli ise (örn: tr)
                                iso_lang = f"{lang_code}-{lang_code.upper()}"
                            
                            # Dil grubuna ekle
                            if iso_lang not in voices:
                                voices[iso_lang] = []
                                    
                            voices[iso_lang].append(model_info)
                            
                            self.log(f"Ses modeli bulundu: {iso_lang} -> {voice_name} [{full_path}]")
                            found_models = True
                
                if not model_found_for_lang:
                    self.log(f"UYARI: {lang_path} klasöründe model dosyası bulunamadı (.onnx veya .onyx)")
            
            # Eğer bu konumda modeller bulunduysa diğer konumlara bakma
            if found_models:
                self.log(f"Ses modelleri bulundu, arama tamamlandı: {piper_dir}")
                break
        
        if not voices:
            self.log(f"UYARI: Hiçbir ses modeli bulunamadı!")
            self.log(f"Aranan konumlar: {', '.join(searched_paths)}")
            self.log(f"Lütfen şu uzantılı dosyaların olduğundan emin olun: .onnx veya .onyx")
        else:
            self.log(f"Toplam {len(voices)} dil için ses modelleri bulundu")
            for lang, voice_list in voices.items():
                self.log(f"  - {lang}: {len(voice_list)} model ({', '.join([v['name'] for v in voice_list])})")
        
        return voices
    
# start_speech_recognition metodunu bu şekilde güncelleyin:
    def start_speech_recognition(self):
        """Konuşma tanıma (STT) sistemini başlatır."""
        try:
            # Wake word dinleyicisi çalışıyorsa durdur (AudioManager'dan)
            if hasattr(self.audio_manager, 'wake_word_detector') and \
               self.audio_manager.wake_word_detector and \
               self.audio_manager.wake_word_detector.is_running:
                self.log("STT başlatılırken aktif Wake Word dinleyici durduruluyor.")
                self.audio_manager.wake_word_detector.stop_listening()
                # UI'daki checkbox'ı da senkronize et
                if hasattr(self, 'wake_word_checkbox'):
                    self.wake_word_checkbox.setChecked(False)


            self.log("DeskGUI: AudioManager aracılığıyla STT başlatılıyor...")
            # AudioManager'ın başlatma fonksiyonunu çağır ve sonucu al
            success, message = self.audio_manager.start_speech_recognition() # AudioManager artık (bool, str) dönecek

            # Durumu her zaman sinyal ile güncelle, AudioManager'dan gelen mesaja göre
            self.speech_status_signal.emit(success, message) # Bu update_speech_status'u tetikleyecek

            if success:
                self.log(f"DeskGUI: STT başarıyla başlatıldı: {message}")
                return True
            else:
                self.log(f"DeskGUI: STT başlatılamadı: {message}")
                return False
        except Exception as e:
             self.log(f"DeskGUI: STT başlatılırken kritik hata: {e}")
             self.log(traceback.format_exc())
             self.speech_status_signal.emit(False, f"STT Başlatma Hatası: {e}")
             return False
    
    # Wake word durum etiketini güncellemek için slot
    @QtCore.pyqtSlot(str)
    def update_wake_word_status_label(self, status_text):
        if hasattr(self, 'wake_word_status_label'): # Bu etiketi create_control_panel'de oluşturmanız gerekecek
            self.wake_word_status_label.setText(f"WW: {status_text}")
            # Renklendirme
            if "Dinliyor" in status_text or "Başlatılıyor" in status_text:
                self.wake_word_status_label.setStyleSheet("color: blue;")
                if not self.audio_manager.speech_active and hasattr(self, 'mic_indicator'): # STT kapalıysa mic'i mavi yap
                    self.mic_indicator.setStyleSheet("color: blue; font-size: 18pt;")
            elif "Algılandı" in status_text:
                self.wake_word_status_label.setStyleSheet("color: green; font-weight: bold;")
            elif "Durdu" in status_text or "Durduruldu" in status_text:
                self.wake_word_status_label.setStyleSheet("color: gray;")
                if not self.audio_manager.speech_active and hasattr(self, 'mic_indicator'): # STT de kapalıysa mic'i gri yap
                     self.mic_indicator.setStyleSheet("color: gray; font-size: 18pt;")
            elif "Hata" in status_text:
                self.wake_word_status_label.setStyleSheet("color: red;")

# stop_speech_recognition metodunu bu şekilde güncelleyin:
    def stop_speech_recognition(self):
        """
        Konuşma tanıma (STT) sistemini durdurur.
        AudioManager'dan gelen sonuca göre UI'ı günceller.
        """
        try:
            self.log("DeskGUI: AudioManager aracılığıyla STT durduruluyor...")
            success, message = self.audio_manager.stop_speech_recognition() # AudioManager artık (bool, str) dönecek

            self.speech_status_signal.emit(not success, message) # STT durduysa active=False olur

            if success:
                self.log(f"DeskGUI: STT başarıyla durduruldu: {message}")
                # Eğer wake word checkbox işaretliyse, WW dinlemeye geri dön
                if hasattr(self, 'wake_word_checkbox') and self.wake_word_checkbox.isChecked():
                    self.log("STT durduruldu, Wake Word dinleyiciye dönülüyor.")
                    QtCore.QTimer.singleShot(100, self.restart_wake_word_listener_safely) # Kısa bir gecikmeyle
            else:
                self.log(f"DeskGUI: STT durdurulamadı: {message}")

        except Exception as e:
            self.log(f"DeskGUI: STT durdurulurken hata: {e}")
            self.log(traceback.format_exc())
            self.speech_status_signal.emit(False, f"STT Durdurma Hatası: {e}") # Hata durumunda STT kapalı kabul edilir

        
    def speak_text(self):
        """Speak the text in the text input field using the configured TTS system."""
        text = self.speak_text_input.text().strip()
        if text:
            self.audio_manager.speak(text)
            self.speak_text_input.clear()
            
    def handle_speech_input(self, text):
        """Handle recognized speech input."""
        try:
            # TTS konuşuyorsa, speech girişini işleme
            if self.speaking_active:
                self.log("⚠️ TTS hala konuşuyor, konuşma girişi yok sayıldı: " + text[:30])
                # Konuşma devam ederken girişi yok say ve çık
                return
                
            if not text or not isinstance(text, str):
                # Boş veya geçersiz girişleri sessizce yok say (log'a yazma)
                return
                    
            self.log(f"Speech recognized: [{text}]")
            
            # Update audio status to show recognition occurred
            self.update_audio_status_signal.emit("recognized", text)
            
            # Ses tanıma API'sinden gelen gereksiz whitespace'leri temizle
            text = text.strip()
            if not text:
                self.log("Speech input was empty after stripping whitespace")
                return
            
            # İşlem zaten devam ediyorsa, yeni istekleri reddet
            if self.is_processing_request:
                self.log("⚠️ Ignoring speech input - already processing a request")
                return
                    
            # UI güncellemelerini ana thread'de yap (thread-safe)
            self.set_input_text_signal.emit(text)
            
            # Konuşma tanıma verisini kaydet
            self.last_speech_text = text
            
            # If Ask LLM checkbox is enabled, send the text to LLM
            if self.use_speech_for_llm and text:
                # Mark as processing to prevent multiple requests
                self.is_processing_request = True
                self.log(f"🔴 Speech to LLM active: Sending text to LLM: '{text}'")
                
                # Update status indicator
                self.update_status_signal.emit("Processing Request", "processing")
                
                # "Thinking..." mesajını thread-safe göster (sinyal kullanarak)
                self.update_thinking_signal.emit()
                
                # Log mesajı gönder
                if self.stream_connected:
                    try:
                        self.command_sender.send_command('log', {
                            'message': f"User input (voice): {text}",
                            'level': 'info'
                        })
                    except Exception as e:
                        self.log(f"Error sending command: {e}")
                
                # LLM işlemini başlat
                try:
                    # Use daemon=True to ensure thread doesn't block application exit
                    llm_thread = threading.Thread(
                        target=self._process_llm_request, 
                        args=(text,), 
                        daemon=True
                    )
                    llm_thread.start()
                    self.log("LLM request thread started successfully")
                except Exception as e:
                    self.log(f"Error starting LLM request thread: {e}")
                    # Directly process if thread creation fails
                    self._process_llm_request(text)
                    
                # Pause speech recognition only in direct mode
                if not self.using_bluetooth_audio:
                    # İşlem devam ederken konuşma tanımayı durdur (direkt modda)
                    self.audio_manager.stop_speech_recognition()
                    self.speech_active = False  # Speech active durumunu güncelle
                    self.mic_active = False     # Mikrofon durumunu güncelle
                    self.log("Speech recognition paused during LLM processing (direct mode)")
                else:
                    self.log("Using Bluetooth mode - speech recognition handled by laptop")
        except Exception as e:
            self.log(f"Error in handle_speech_input: {e}")
            self.is_processing_request = False  # Reset processing flag on error
            # Update status indicators on error
            self.update_status_signal.emit("Error", "error")
            self.update_audio_status_signal.emit("error", str(e))
            import traceback
            self.log(traceback.format_exc())
            self.speech_triggered_by_wake_word = False # Resetle
            if hasattr(self, 'wake_word_checkbox') and self.wake_word_checkbox.isChecked():
                self.restart_wake_word_listener_safely()
            
            # Hata durumunda konuşma tanımayı yeniden başlatmayı dene
            QtCore.QTimer.singleShot(2000, self.force_restart_speech_recognition)
        """Handle recognized speech input."""
        try:
            if not text or not isinstance(text, str):
                self.log(f"Warning: Invalid speech input received: {repr(text)}")
                return
                    
            self.log(f"Speech recognized: [{text}]")
            
            # Update audio status to show recognition occurred
            self.update_audio_status_signal.emit("recognized", text)
            
            # Ses tanıma API'sinden gelen gereksiz whitespace'leri temizle
            text = text.strip()
            if not text:
                self.log("Speech input was empty after stripping whitespace")
                return
            
            # İşlem zaten devam ediyorsa, yeni istekleri reddet
            if self.is_processing_request:
                self.log("⚠️ Ignoring speech input - already processing a request")
                return
                    
            # UI güncellemelerini ana thread'de yap (thread-safe)
            self.set_input_text_signal.emit(text)
            
            # Konuşma tanıma verisini kaydet
            self.last_speech_text = text
            
            # If Ask LLM checkbox is enabled, send the text to LLM
            if self.use_speech_for_llm and text:
                # Mark as processing to prevent multiple requests
                self.is_processing_request = True
                self.log(f"🔴 Speech to LLM active: Sending text to LLM: '{text}'")
                
                # Update status indicator
                self.update_status_signal.emit("Processing Request", "processing")
                
                # "Thinking..." mesajını thread-safe göster (sinyal kullanarak)
                self.update_thinking_signal.emit()
                
                # Log mesajı gönder
                if self.stream_connected:
                    try:
                        self.command_sender.send_command('log', {
                            'message': f"User input (voice): {text}",
                            'level': 'info'
                        })
                    except Exception as e:
                        self.log(f"Error sending command: {e}")
                
                # LLM işlemini başlat
                try:
                    # Use daemon=True to ensure thread doesn't block application exit
                    llm_thread = threading.Thread(
                        target=self._process_llm_request, 
                        args=(text,), 
                        daemon=True
                    )
                    llm_thread.start()
                    self.log("LLM request thread started successfully")
                except Exception as e:
                    self.log(f"Error starting LLM request thread: {e}")
                    # Directly process if thread creation fails
                    self._process_llm_request(text)
                    
                # Pause speech recognition only in direct mode
                if not self.using_bluetooth_audio:
                    # İşlem devam ederken konuşma tanımayı durdur (direkt modda)
                    self.audio_manager.stop_speech_recognition()
                    self.speech_active = False  # Speech active durumunu güncelle
                    self.mic_active = False     # Mikrofon durumunu güncelle
                    self.log("Speech recognition paused during LLM processing (direct mode)")
                else:
                    self.log("Using Bluetooth mode - speech recognition handled by laptop")
        except Exception as e:
            self.log(f"Error in handle_speech_input: {e}")
            self.is_processing_request = False  # Reset processing flag on error
            # Update status indicators on error
            self.update_status_signal.emit("Error", "error")
            self.update_audio_status_signal.emit("error", str(e))
            import traceback
            self.log(traceback.format_exc())
            
            # Hata durumunda konuşma tanımayı yeniden başlatmayı dene
            QtCore.QTimer.singleShot(2000, self.force_restart_speech_recognition)
    
    # In deskgui.py, within the DeskGUI class, modify the _process_llm_request method:

    def _process_llm_request(self, input_text):
        """Seçilen servisi kullanarak LLM isteğini işler ve yanıtı yönetir."""
        try:
            self.log(f"LLM isteği işleniyor: '{input_text[:100]}' (Servis: {self.current_llm_service})")

            output_text_final = ""
            llm_response_successful = False
            error_message_for_status = "" # Durum çubuğu için hata mesajı

            if self.current_llm_service == "ollama":
                # --- Ollama İşleme Mantığı ---
                # (Ollama logic remains unchanged as per the problem description focusing on Gemini)
                self.log("Ollama servisi kullanılıyor.")
                recog_lang_code = getattr(self.audio_manager, 'language', 'tr-TR')
                if not recog_lang_code: recog_lang_code = "tr-TR"
                src_lang_for_translate = recog_lang_code.split('-')[0] if recog_lang_code else "tr"
                llm_input_for_ollama = input_text

                if src_lang_for_translate != "en" and TRANSLATE_MODULE_AVAILABLE and TranslateHelper:
                    self.log(f"Ollama için giriş İngilizce'ye çevriliyor ({src_lang_for_translate} -> en)...")
                    try:
                        llm_input_for_ollama = TranslateHelper.translate(input_text, src_lang_for_translate, "en")
                        if not llm_input_for_ollama:
                            self.log("Çeviri boş döndü, orijinal metin kullanılıyor.")
                            llm_input_for_ollama = input_text
                        else:
                            self.log(f"Ollama için çevrilmiş giriş: {llm_input_for_ollama[:100]}...")
                    except Exception as e:
                        self.log(f"Ollama için giriş çevrilirken hata: {e}. Orijinal metin kullanılıyor.")
                        llm_input_for_ollama = input_text

                api_url = self.ollama_url.rstrip('/')
                # Windows'ta os.path.join URL içinde ters slash (\\) üreterek 404 oluşturuyordu; manuel ekle
                if not api_url.endswith('/api/generate'):
                    if api_url.endswith('/api'):
                        api_url += '/generate'
                    else:
                        api_url += '/api/generate'
                if '\\' in api_url:
                    self.log(f"Uyarı: Ollama API URL'sinde ters slash tespit edildi, istek başarısız olabilir: {api_url}")

                self.log(f"Ollama API URL'si: {api_url}")
                # UI model butonlarından seçim oku (llm_request fonksiyonundaki mantıkla uyumlu)
                selected_ollama_model = getattr(self, 'ollama_model', None)
                try:
                    if hasattr(self, 'model_button_group'):
                        for btn in self.model_button_group.buttons():
                            if btn.isChecked():
                                selected_ollama_model = btn.text().strip()
                                # Seçimi kalıcı tut
                                self.ollama_model = selected_ollama_model
                                break
                except Exception as e_btn:
                    self.log(f"Model butonları okunurken hata: {e_btn}")
                if not selected_ollama_model:
                    selected_ollama_model = 'SentryBOT:4b'
                self.log(f"Kullanılan Ollama modeli: {selected_ollama_model}")
                payload = {'model': selected_ollama_model, 'prompt': llm_input_for_ollama, 'stream': False}
                self.log("Ollama API'sine istek gönderiliyor...")
                response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=self.request_timeout_seconds)

                if response.status_code == 200:
                    self.log("Ollama API'sinden başarılı yanıt alındı.")
                    try:
                        data = response.json()
                        raw_ollama_output = data.get('response', '')
                        if not raw_ollama_output:
                            self.log("Uyarı: Ollama API'sinden boş yanıt.")
                            raw_ollama_output = "Ollama'dan yanıt alamadım. Lütfen tekrar deneyin."

                        if src_lang_for_translate != "en" and TRANSLATE_MODULE_AVAILABLE and TranslateHelper:
                            self.log(f"Ollama yanıtı '{src_lang_for_translate}' diline geri çevriliyor...")
                            try:
                                output_text_final = TranslateHelper.translate(raw_ollama_output, "en", src_lang_for_translate)
                                if not output_text_final:
                                    self.log("Ollama yanıt çevirisi boş döndü, ham İngilizce yanıt kullanılıyor.")
                                    output_text_final = raw_ollama_output
                                else:
                                    self.log(f"Çevrilmiş Ollama yanıtı: {output_text_final[:100]}...")
                            except Exception as e:
                                self.log(f"Ollama yanıtı çevrilirken hata: {e}. Ham İngilizce yanıt kullanılıyor.")
                                output_text_final = raw_ollama_output
                        else:
                            output_text_final = raw_ollama_output
                        llm_response_successful = True
                    except ValueError as json_err:
                        self.log(f"Ollama JSON ayrıştırma hatası: {json_err}")
                        output_text_final = f"Ollama yanıtı ayrıştırılırken hata: {json_err}"
                        error_message_for_status = "Ollama Yanıt Hatası"
                else:
                    error_message_for_status = f"Ollama API Hatası ({response.status_code})"
                    error_message = f"Ollama API Hatası: HTTP {response.status_code}"
                    try:
                         error_details = response.json()
                         detail_msg = error_details.get("error", response.text[:200])
                         error_message += f" - {detail_msg}"
                    except:
                         error_message += f" - {response.text[:200]}"
                    self.log(error_message)
                    # Model bulunamadıysa mevcut modelleri listelemeyi dene
                    if response.status_code == 404 and "model" in error_message.lower():
                        try:
                            base_api = self.ollama_url.rstrip('/')
                            # /api veya /api/generate varyantlarından kök /api elde et
                            if base_api.endswith('/api/generate'):
                                base_api = base_api[:-len('/generate')]
                            elif base_api.endswith('/generate'):
                                base_api = base_api[:-len('/generate')]
                            if not base_api.endswith('/api'):
                                base_api += '/api'
                            tags_url = base_api + '/tags'
                            self.log(f"Mevcut modelleri çekiyor: {tags_url}")
                            tags_resp = requests.get(tags_url, timeout=10)
                            if tags_resp.status_code == 200:
                                try:
                                    tags_json = tags_resp.json()
                                    models = [m.get('name') for m in tags_json.get('models', []) if m.get('name')]
                                    if models:
                                        self.log("Ollama mevcut modeller: " + ', '.join(models))
                                        if selected_ollama_model not in models:
                                            self.log(f"Seçilen model listede yok. GUI'de bulunan adla Ollama'daki ad tam eşleşmiyor olabilir.")
                                    else:
                                        self.log("Model listesi boş döndü.")
                                except Exception as e_tags_parse:
                                    self.log(f"Model listesi ayrıştırılamadı: {e_tags_parse}")
                            else:
                                self.log(f"Model listesi alınamadı: HTTP {tags_resp.status_code}")
                        except Exception as e_tags:
                            self.log(f"Modeller sorgulanırken hata: {e_tags}")
                    output_text_final = error_message

            elif self.current_llm_service == "gemini":
                # --- Gemini İşleme Mantığı ---
                self.log("Gemini servisi kullanılıyor.")
                gemini_response_obj = None # API'den dönen orijinal yanıtı tutmak için

                if not self.gemini_helper_instance:
                    error_message_for_status = "Gemini İstemci Hatası"
                    error_message = "Gemini istemcisi başlatılmadı. Lütfen API anahtarını ve ayarları yapılandırın."
                    self.log(error_message)
                    output_text_final = error_message
                    QMetaObject.invokeMethod(self, "_show_warning_message_slot",
                                             QtCore.Qt.QueuedConnection,
                                             Q_ARG(str, "Gemini Hatası"),
                                             Q_ARG(str, error_message))
                    QMetaObject.invokeMethod(self, "show_gemini_settings_menu", QtCore.Qt.QueuedConnection)
                else:
                    self.log(f"Gemini modeli '{self.gemini_helper_instance.model}' için prompt gönderiliyor: '{input_text[:100]}...'")
                    try:
                        gemini_response_obj = self.gemini_helper_instance.send_prompt(prompt=input_text) # API'den dönen objeyi al

                        # Gemini yanıtı string olarak dönebilir (bazı hata durumları veya direkt metin)
                        # veya GenerativeModel response objesi olarak (başarı, filtre vb.)
                        if isinstance(gemini_response_obj, str): 
                            output_text_final = gemini_response_obj
                            # !!! BURASI ÖNEMLİ DÜZELTME !!!
                            # Eğer GeminiHelper'dan gelen string boş değilse, bunu başarılı bir metin yanıtı kabul et.
                            if output_text_final.strip(): # Boşlukları temizledikten sonra dolu mu?
                                self.log(f"GeminiHelper'dan non-empty string yanıt alındı (başarılı kabul ediliyor): {output_text_final[:200]}...")
                                llm_response_successful = True # <<--- BAŞARILI DURUMDA TRUE YAPILDI
                            else: # Boş string ise metin çıkarılamadı veya başka bir durum
                                self.log("GeminiHelper'dan boş veya sadece boşluk içeren string yanıt alındı.")
                                output_text_final = "Gemini'den boş metin yanıtı alındı." # Kullanıcıya gösterilecek mesaj
                                error_message_for_status = "Gemini Yanıtı Boş" # Durum çubuğu için
                                # llm_response_successful False kalmalı

                        elif hasattr(gemini_response_obj, 'text') and gemini_response_obj.text: # Başarılı yanıt objesi, text içeriği dolu
                            output_text_final = gemini_response_obj.text
                            self.log(f"Gemini yanıt objesinden dolu metin alındı: {output_text_final[:100]}...")
                            llm_response_successful = True # <<--- BAŞARILI DURUMDA TRUE YAPILDI

                        elif hasattr(gemini_response_obj, 'prompt_feedback') and \
                             hasattr(gemini_response_obj.prompt_feedback, 'safety_ratings') and \
                             gemini_response_obj.prompt_feedback.safety_ratings: # Güvenlik filtresi
                            # ... (Güvenlik filtresi işleme mantığı aynı kalır) ...
                            safety_ratings = gemini_response_obj.prompt_feedback.safety_ratings
                            safety_issues_list = []
                            for rating in safety_ratings:
                                category_name = str(rating.category).split('.')[-1]
                                probability_name = str(rating.probability).split('.')[-1]
                                if probability_name not in ["NEGLIGIBLE", "LOW"]:
                                    safety_issues_list.append(f"{category_name}: {probability_name}")

                            if safety_issues_list:
                                safety_issues = ", ".join(safety_issues_list)
                                error_message_for_status = "Gemini Filtrelendi"
                                error_message = f"Gemini yanıtı filtrelendi. Güvenlik sorunları: {safety_issues}"
                                self.log(error_message)
                                output_text_final = f"Üzgünüm, bu istekle ilgili güvenlik yönergelerime takıldım. ({safety_issues})"
                                QMetaObject.invokeMethod(self, "_show_warning_message_slot",
                                                         QtCore.Qt.QueuedConnection,
                                                         Q_ARG(str, "Gemini Filtrelendi"),
                                                         Q_ARG(str, f"Güvenlik sebebiyle yanıt filtrelendi:\n{safety_issues}"))
                            else: # Safety rating var ama sorunlu değil ve text yoksa
                                error_message_for_status = "Gemini Boş Yanıt"
                                error_message = "Gemini API'sinden .text özelliği olmayan bir yanıt alındı (safety rating sorunsuz)."
                                self.log(error_message)
                                output_text_final = "Gemini'den geçerli bir metin yanıtı alamadım (güvenlik sorunu yok)."

                        else: # Beklenmeyen yanıt tipi veya boş obje
                            error_message_for_status = "Gemini Beklenmedik Yanıt"
                            error_message = f"Gemini API'sinden boş veya beklenmedik yanıt objesi alındı: {type(gemini_response_obj)}"
                            self.log(error_message)
                            output_text_final = "Gemini'den beklenmedik bir yanıt aldım. Lütfen tekrar deneyin."
                            QMetaObject.invokeMethod(self, "_show_warning_message_slot",
                                                     QtCore.Qt.QueuedConnection,
                                                     Q_ARG(str, "Gemini API Hatası"),
                                                     Q_ARG(str, f"Beklenmeyen yanıt türü: {type(gemini_response_obj)}"))

                    except Exception as gemini_err:
                        # ... (Hata işleme mantığı aynı kalır) ...
                        self.log(f"Gemini API çağrısı veya yanıt işleme sırasında hata: {gemini_err}")
                        self.log(traceback.format_exc())
                        error_message_for_status = "Gemini API Hatası"
                        error_message = f"Gemini API hatası: {str(gemini_err)}"
                        output_text_final = error_message
                        QMetaObject.invokeMethod(self, "_show_warning_message_slot",
                                                 QtCore.Qt.QueuedConnection,
                                                 Q_ARG(str, "Gemini API Hatası"),
                                                 Q_ARG(str, f"Gemini API çağrısı sırasında hata: {gemini_err}.\nLütfen ayarlarınızı ve API anahtarınızı kontrol edin."))
                        if "API_KEY" in str(gemini_err).upper() or "PERMISSION" in str(gemini_err).upper():
                            if hasattr(self, 'ollama_radio'):
                                QMetaObject.invokeMethod(self.ollama_radio, "setChecked",
                                                         QtCore.Qt.QueuedConnection,
                                                         QtCore.Q_ARG(bool, True))
            else:
                error_message_for_status = "Bilinmeyen LLM"
                error_message = f"Bilinmeyen LLM servisi seçili: {self.current_llm_service}"
                self.log(error_message)
                output_text_final = error_message
                QMetaObject.invokeMethod(self, "_show_critical_message_slot",
                                         QtCore.Qt.QueuedConnection,
                                         Q_ARG(str, "LLM Servis Hatası"),
                                         Q_ARG(str, f"Bilinmeyen LLM servisi: {self.current_llm_service}"))

        except requests.exceptions.Timeout:
            error_message_for_status = "İstek Zaman Aşımı"
            timeout_msg = f"Hata: {self.current_llm_service.capitalize()} API isteği zaman aşımına uğradı."
            self.log(timeout_msg)
            output_text_final = timeout_msg
        except requests.exceptions.ConnectionError:
            error_message_for_status = "Bağlantı Hatası"
            conn_err_msg = f"Hata: {self.current_llm_service.capitalize()} API'sine bağlanılamadı. Servisin çalıştığından emin olun."
            self.log(conn_err_msg)
            output_text_final = conn_err_msg
        except requests.exceptions.RequestException as req_err:
            error_message_for_status = "İstek Hatası"
            req_err_msg = f"{self.current_llm_service.capitalize()} API isteği gönderilirken hata: {req_err}"
            self.log(req_err_msg)
            output_text_final = req_err_msg
        except Exception as e:
            error_message_for_status = "İşlem Hatası"
            gen_err_msg = f"{self.current_llm_service.capitalize()} isteği sırasında genel hata: {e}"
            self.log(gen_err_msg)
            self.log(traceback.format_exc())
            output_text_final = gen_err_msg
            QMetaObject.invokeMethod(self, "update_audio_status_signal", Qt.QueuedConnection,
                                     Q_ARG(str, "error"), Q_ARG(str, f"Hata: {str(e)}"))
        finally:
            if self.request_timer:
                self.request_timer.stop()
                self.request_timer = None
                self.log("Request timeout timer durduruldu.")

            QMetaObject.invokeMethod(self, "update_output_signal",
                                     Qt.QueuedConnection,
                                     Q_ARG(str, output_text_final),
                                     Q_ARG(bool, True))

            status_type_final = "success" if llm_response_successful else "error"
            status_text_final = "İşlem Tamamlandı" if llm_response_successful else (error_message_for_status if error_message_for_status else f"{self.current_llm_service.capitalize()} Hatası")

            QMetaObject.invokeMethod(self, "update_status_signal",
                                     Qt.QueuedConnection,
                                     Q_ARG(str, status_text_final),
                                     Q_ARG(str, status_type_final))

            if llm_response_successful and output_text_final:
                try:
                    pub.sendMessage('llm_response', response_text=output_text_final)
                    self.log("LLM yanıtı için pubsub mesajı gönderildi, TTS abone tarafından yönetilecek.")
                except Exception as e:
                    self.log(f"llm_response olayı gönderilirken hata: {e}")
                QMetaObject.invokeMethod(self, "update_audio_status_signal",
                                         Qt.QueuedConnection,
                                         Q_ARG(str, "idle"),
                                         Q_ARG(str, "Yanıt hazır"))
                self.llm_response_pending_tts_completion = True
            else:
                self.log("LLM yanıtı başarısız veya boş, TTS atlanıyor.") # This log should now only appear for actual failures
                self.llm_response_pending_tts_completion = False
                QMetaObject.invokeMethod(self, "restart_stt_if_needed_after_llm", Qt.QueuedConnection)

            self.log(f"LLM ({self.current_llm_service}) istek işleme (worker thread) tamamlandı.")

    @QtCore.pyqtSlot()
    def on_request_completed(self): # Bu sinyal artık _process_llm_request.finally'den emit edilmiyor
        """
        Bu slot, _process_llm_request bittiğinde çağrılıyordu.
        Artık LLM yanıtının işlenmesi ve TTS'in tamamlanması ayrı yönetiliyor.
        Bu fonksiyon ya kaldırılabilir ya da sadece çok hızlı UI güncellemeleri için tutulabilir.
        """
        self.log("🟢 on_request_completed sinyali alındı (muhtemelen kullanım dışı).")
        # Animasyon gibi şeyler burada kalabilir, eğer LLM yanıtından bağımsızsa
        # try:
        #     if self.stream_connected:
        #          self.send_animation("head_nod")
        # except Exception as e:
        #     self.log(f"on_request_completed içinde animasyon hatası: {e}")

    def on_tts_really_complete_after_llm(self):
        """
        Herhangi bir TTS konuşması bittiğinde çağrılır.
        Eğer bir LLM yanıtının TTS'i bekleniyorsa, STT'yi yeniden başlatır.
        """
        if self.llm_response_pending_tts_completion:
            self.log("LLM yanıtının TTS'i tamamlandı. STT yeniden başlatılacak.")
            self.llm_response_pending_tts_completion = False # Bayrağı sıfırla
            self.last_llm_response_text_for_tts = "" # Saklanan metni temizle
            
            # STT'yi yeniden başlat
            self.restart_stt_if_needed_after_llm()
        # else:
            # self.log("Normal TTS tamamlandı, LLM ile ilgili değil.") # Çok fazla log olmaması için
            pass

    # STT bittiğinde (LLM ve TTS sonrası)
    @QtCore.pyqtSlot()
    def restart_stt_if_needed_after_llm(self):
        self.log("DeskGUI: restart_stt_if_needed_after_llm çağrıldı.")
        self.is_processing_request = False # LLM işlemi bitti
        self.update_status_signal.emit("Idle", "success")

        if self.speech_triggered_by_wake_word:
            self.log("DeskGUI: STT wake word ile tetiklenmişti, işlem bitti, WW kontrol edilecek.")
            self.speech_triggered_by_wake_word = False # Bayrağı sıfırla
            if self.wake_word_enabled_by_checkbox:
                self.log("DeskGUI: WW checkbox aktif, WW yeniden başlatılıyor.")
                self.audio_manager.start_wake_word_listener()
            else:
                self.log("DeskGUI: WW checkbox kapalı, STT/WW kapalı kalacak.")
                self.speech_status_signal.emit(False, "STT Kapalı (WW sonrası işlem bitti)")

        elif not self.using_bluetooth_audio: # WW ile tetiklenmediyse ve direct mode ise
            self.log("DeskGUI: Normal LLM/TTS sonrası, STT/WW durumu kontrol ediliyor (direct mode).")
            if self.wake_word_enabled_by_checkbox: # Kullanıcı WW istiyorsa
                self.log("DeskGUI: WW checkbox aktif, WW başlatılıyor.")
                self.audio_manager.start_wake_word_listener()
            else: # Kullanıcı WW istemiyorsa, STT'yi yeniden başlatmayı deneyebiliriz (manuel mod için)
                  # Veya kapalı bırakabiliriz. Şimdilik kapalı bırakalım, butonla açılsın.
                self.log("DeskGUI: WW checkbox kapalı, STT/WW kapalı kalacak (manuel başlatma bekleniyor).")
                self.speech_status_signal.emit(False, "STT Kapalı (İşlem Bitti)")
        # Bluetooth modunda STT yönetimi laptopta olduğu için ek bir şey yapmaya gerek yok.


    # start_speech_recognition ve stop_speech_recognition DeskGUI'den kaldırılabilir,
    # doğrudan toggle_microphone veya handle_wake_word_detection_event üzerinden AudioManager çağrılacak.
    # Ancak UI güncellemeleri için speech_status_signal'e bağlı kalacaklar.
    def force_restart_speech_recognition(self):
        """Konuşma tanımayı zorunlu olarak yeniden başlat"""
        self.log("Konuşma tanıma zorunlu olarak yeniden başlatılıyor...")
        
        # AudioManager'ı tam olarak sıfırlayalım
        if hasattr(self, 'audio_manager'):
            try:
                # Önce varolan tüm kaynakları kapatalım
                self.audio_manager.stop_speech_recognition()
                time.sleep(0.3)  # Kaynak temizleme için kısa bir bekleme
                
                # AudioManager durumunu sıfırlayalım
                self.audio_manager.speech_active = False
                
                # Speech input nesnesini tamamen yeniden oluşturmak için
                if hasattr(self.audio_manager, 'speech_input'):
                    if hasattr(self.audio_manager.speech_input, 'stop_listening'):
                        try:
                            self.audio_manager.speech_input.stop_listening(wait_for_stop=True)
                        except Exception as e:
                            self.log(f"stop_listening hatası: {e}")
                    
            except Exception as e:
                self.log(f"Audio manager sıfırlanırken hata: {e}")
        
        # Durumu sıfırla
        self.speech_active = False
        self.mic_active = False
        
        # Olay dinleyicilerini sıfırla
        try:
            # speech event'inden önce ayrılalım, sonra tekrar abone olalım
            pub.unsubscribe(self.handle_speech_input, 'speech')
            time.sleep(0.2)
            pub.subscribe(self.handle_speech_input, 'speech')
            self.log("Speech event aboneliği yenilendi")
        except Exception as e:
            self.log(f"Event aboneliği yenilenirken hata: {e}")
        
        # Konuşma tanımayı tamamen yeniden başlat
        success = self.start_speech_recognition()
        
        if success:
            self.log("Konuşma tanıma başarıyla yeniden başlatıldı")
        else:
            self.log("Konuşma tanıma yeniden başlatılamadı!")
            # 5 saniye sonra tekrar deneyelim
            QtCore.QTimer.singleShot(5000, self.force_restart_speech_recognition)
        
        return success
            
    def connect_to_robot(self):
        """Connect to the robot's video stream and command server."""
        # Update robot IP and ports from the input fields
        self.robot_ip = self.ip_input.text().strip()
        try:
            self.command_port = int(self.cmd_port_input.text().strip())
            self.video_port = int(self.vid_port_input.text().strip())
            self.gui_listen_port = int(self.gui_port_input.text().strip()) # Eğer listener portu da ayarlanabilirse
        except ValueError:
            self.log("Error: Invalid port number entered. Please check ports.")
            QtWidgets.QMessageBox.warning(self, "Invalid Port", "Please enter valid numeric port numbers.")
            return
    
        if not self.robot_ip:
            self.log("Please enter a valid robot IP address")
            QtWidgets.QMessageBox.warning(self, "Missing IP", "Please enter the robot's IP address.")
            return
    
        self.log(f"Connecting to robot at {self.robot_ip}...")
        self.update_status_signal.emit("Connecting...", "processing") # Durumu güncelle
    
        # Update the command sender with new IP/Port
        self.command_sender = CommandSender(self.robot_ip, self.command_port)
    
        # Check if we can connect to the command port first
        if not self.command_sender.connect():
            error_msg = f"Could not connect to robot command port at {self.robot_ip}:{self.command_port}"
            self.log(error_msg)
            QtWidgets.QMessageBox.critical(self, "Connection Failed", error_msg)
            self.update_status_signal.emit("Command Connection Failed", "error") # Durumu güncelle
            return
    
        self.log("Command connection successful, initializing video stream...")
    
        # Create video stream
        try:
            stream_url = f"http://{self.robot_ip}:{self.video_port}/stream.mjpg"
            self.log(f"Attempting to connect to video stream: {stream_url}") # Log stream URL
            self.video_stream = RemoteVideoStream(
                self.robot_ip,
                self.video_port
            ).start()
    
            # Küçük bir bekleme süresi ekleyerek stream'in başlamasını bekle
            time.sleep(1.0) # 1 saniye bekle
    
            # İlk frame'i okumayı dene (opsiyonel test)
            first_frame = self.video_stream.read()
            if first_frame is None:
                 # Stream başladı ama frame gelmiyor olabilir
                 self.log("Warning: Video stream started but no frame received yet. Check stream URL and robot camera.")
                 # Kullanıcıyı bilgilendir ama bağlantıyı koparma, belki düzelir
                 # QtWidgets.QMessageBox.warning(self, "Video Warning", "Video stream connected, but no image received yet.")
    
            # Update button text and status
            self.connect_button.setText("Disconnect")
            self.stream_connected = True
            self.video_label.setText("") # "Video Stream Disconnected" yazısını kaldır
            self.video_label.setStyleSheet("background-color: black;") # Siyah arka plan
            self.log(f"Connected to robot at {self.robot_ip}")
            self.update_status_signal.emit("Connected", "success") # Durumu güncelle
    
            # !!! YENİ SATIR: Timer'ı burada başlat !!!
            self.timer.start(30)  # 30 ms = ~33 FPS için güncelleme sıklığı
    
            # Robot bağlandıktan sonra UI elemanlarını etkinleştir
            self.update_ui_connection_state(True)
            # Robot durumunu iste (gecikmeli)
            QtCore.QTimer.singleShot(500, self.request_robot_status)
            # Kişilik istemesi artık yok
            # QtCore.QTimer.singleShot(1000, self.request_personalities_from_robot)
    
        except Exception as e:
            error_msg = f"Error connecting to video stream: {str(e)}"
            self.log(error_msg)
            self.log(traceback.format_exc()) # Detaylı hata logu
            QtWidgets.QMessageBox.critical(self, "Video Connection Failed", error_msg)
            if self.command_sender:
                 self.command_sender.close()
            self.stream_connected = False
            self.connect_button.setText("Connect")
            self.update_status_signal.emit("Video Connection Failed", "error") # Durumu güncelle
            # Başarısız olursa UI'ı devre dışı bırak
            self.update_ui_connection_state(False)
        
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    # BU METODU DeskGUI SINIFININ İÇİNE YERLEŞTİRİN VEYA MEVCUT OLANLA DEĞİŞTİRİN
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    def update_processing_mode(self):
        """Update the image processing mode and start/stop relevant modules."""
        # ... (sender, new_mode belirleme kısmı aynı) ...
        sender = self.sender()
        new_mode = 'none'
        if self.face_recognition_radio.isChecked(): new_mode = 'face_recognition'
        elif self.face_find_radio.isChecked(): new_mode = 'face_find'
        elif self.motion_detection_radio.isChecked(): new_mode = 'motion'
        elif self.finger_tracking_radio.isChecked(): new_mode = 'finger'
        elif self.age_emotion_radio.isChecked(): new_mode = 'age_emotion'
        elif self.object_detection_radio.isChecked(): new_mode = 'object_detection'
        elif self.no_processing_radio.isChecked(): new_mode = 'none'


        if not hasattr(self, 'processing_mode') or self.processing_mode != new_mode:
            old_mode = getattr(self, 'processing_mode', 'none')
            self.processing_mode = new_mode
            self.log(f"Processing mode changed from '{old_mode}' to '{self.processing_mode}'")

            # --- Stop the previously active module ---
            if old_mode == 'face_recognition' or old_mode == 'face_find':
                if hasattr(self, 'face_detector') and self.face_detector: self.face_detector.stop()
                if hasattr(self, 'face_tracker') and self.face_tracker:
                    self.face_tracker.stop()
                    self.face_tracker = None
                    self.log("Face specific ObjectTracker stopped.")
                # --- YENİ: Durum değişkenlerini sıfırla ---
                self.face_target_locked = False
                self.face_tracker_consecutive_failures = 0
                # --- YENİ SONU ---
            elif old_mode == 'motion':
                if hasattr(self, 'motion_detector') and self.motion_detector: self.motion_detector.stop()
            elif old_mode == 'finger':
                if hasattr(self, 'finger_tracker') and self.finger_tracker: self.finger_tracker.stop()
            elif old_mode == 'age_emotion':
                if hasattr(self, 'age_emotion_detector') and self.age_emotion_detector: self.age_emotion_detector.stop()
            elif old_mode == 'object_detection':
                if hasattr(self, 'object_detector') and self.object_detector: self.object_detector.stop()
                if hasattr(self, 'object_tracker') and self.object_tracker: self.object_tracker.stop()


            # --- Start the newly selected module ---
            success = False
            if self.processing_mode == 'face_recognition' or self.processing_mode == 'face_find':
                # --- YENİ: Tracker oluşturmadan önce sıfırla ---
                self.face_target_locked = False
                self.face_tracker_consecutive_failures = 0
                if hasattr(self, 'face_tracker') and self.face_tracker: self.face_tracker.stop(); self.face_tracker = None
                # --- YENİ SONU ---
                if VISION_MODULES_AVAILABLE and ObjectTracker:
                    self.face_tracker = ObjectTracker(self.command_sender)
                    self.log("Face specific ObjectTracker created.")
                else:
                    self.log("UYARI: ObjectTracker modülü yüz takibi için kullanılamıyor.")
                    self.face_tracker = None
                if hasattr(self, 'face_detector') and self.face_detector:
                    mode_str = 'recognize' if self.processing_mode == 'face_recognition' else 'find'
                    success = self.face_detector.start(mode=mode_str)
            # ... (diğer modların başlatılması aynı) ...
            elif self.processing_mode == 'motion':
                if hasattr(self, 'motion_detector') and self.motion_detector: success = self.motion_detector.start()
            elif self.processing_mode == 'finger':
                if hasattr(self, 'finger_tracker') and self.finger_tracker: success = self.finger_tracker.start()
            elif self.processing_mode == 'age_emotion':
                if hasattr(self, 'age_emotion_detector') and self.age_emotion_detector: success = self.age_emotion_detector.start()
            elif self.processing_mode == 'object_detection':
                if hasattr(self, 'object_detector') and self.object_detector: success = self.object_detector.start()
                if success and self.tracking_enabled and hasattr(self, 'object_tracker') and self.object_tracker: self.object_tracker.start()


            # --- Handle failure and update robot/tracking ---
            if self.processing_mode != 'none' and not success:
                self.log(f"UYARI: '{self.processing_mode}' modu başlatılamadı!")
                self.no_processing_radio.setChecked(True)
            else:
                if self.command_sender and self.command_sender.connected:
                    try:
                        self.command_sender.send_command('set_processing_mode', {'mode': self.processing_mode})
                        self.log(f"Sent processing mode '{self.processing_mode}' to robot.")
                    except Exception as e:
                        self.log(f"Error sending processing mode update to robot: {e}")

            self.update_tracking()
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    # update_processing_mode SONU
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    def update_tracking(self):
        """Update the tracking state based on the checkbox."""
        self.tracking_enabled = self.tracking_checkbox.isChecked()
        # self.tracking.set_active(self.tracking_enabled) # Eski tracker yerine yenisini kullanacağız

        if hasattr(self, 'object_tracker') and self.object_tracker:
            if self.tracking_enabled:
                # Takip sadece nesne algılama modunda anlamlı
                if self.processing_mode == 'object_detection':
                    self.object_tracker.start()
                    self.log("Object tracking enabled and started.")
                else:
                    self.log("Tracking enabled, but will start when Object Detection mode is active.")
                    self.object_tracker.stop() # Diğer modlarda çalışmasın
            else:
                self.object_tracker.stop()
                self.log("Object tracking disabled.")
        else:
            self.log("Object tracker module not available.")

        # DeskGUI sınıfının içine (örn. update_frame'den önce veya sonra) ekleyin
    def _draw_face_results(self, frame, detected_data):
        """Draws face bounding boxes and names on the frame."""
        drawn_img = frame.copy() # BGR frame üzerinde çalışıyoruz
        for face_info in detected_data:
            try:
                x, y, w, h = face_info['bbox']
                name = face_info.get('name', "Unknown")
                priority = face_info.get('priority', False)

                # Renkleri BGR olarak belirle
                color = (255, 0, 0) if priority else (0, 100, 255) # BGR: Kırmızı (öncelikli), Turuncu (normal)

                # Dikdörtgen çiz
                cv2.rectangle(drawn_img, (x, y), (x + w, y + h), color, 2)

                # Etiket için Y konumu ayarla
                label_y = y - 10 if y - 10 > 10 else y + h + 20

                # Metni yaz
                cv2.putText(drawn_img, name, (x + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception as e:
                self.log(f"Error drawing face result for {face_info.get('name', '?')}: {e}")
        return drawn_img

    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    # BU METODU DeskGUI SINIFININ İÇİNE YERLEŞTİRİN VEYA MEVCUT OLANLA DEĞİŞTİRİN
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    def update_frame(self):
        """Update the video frame display with robust face tracking."""
        # --- Akış Kontrolü (Değişiklik Yok) ---
        if not hasattr(self, 'video_stream') or not self.video_stream or not getattr(self, 'stream_connected', False):
            if hasattr(self, 'timer') and self.timer.isActive(): self.timer.stop()
            if hasattr(self, 'video_label'):
                self.video_label.setText("Video Stream Disconnected"); self.video_label.setStyleSheet("background-color: black; color: gray; font-size: 16pt;")
            return
        start_time = time.time()
        frame = self.video_stream.read()
        if frame is None:
            self.log("Failed to read frame (frame is None). Assuming disconnection.")
            if hasattr(self, 'timer') and self.timer.isActive(): self.timer.stop()
            if hasattr(self, 'disconnect_from_robot'): self.disconnect_from_robot()
            self.stream_connected = False
            return
        # --- Akış Kontrolü Sonu ---

        try:
            processed_frame = frame.copy() # Her zaman işlemek için bir kopya ile başla
            height, width, _ = processed_frame.shape
            target_object_rect = None # Servo takibi için hedef BBox
            is_priority = False       # Hedef öncelikli mi?
            current_mode = getattr(self, 'processing_mode', 'none')

            # ================================================================
            # ----- Yüz Tanıma / Bulma (ROBUST TAKİP MANTIĞI) -----
            # ================================================================
            if current_mode == 'face_recognition' or current_mode == 'face_find':
                # --- Adım 1: Anlık Algılama Sonuçlarını Al ---
                detected_faces, detected_names, detected_priority_flags = [], [], []
                if hasattr(self, 'face_detector') and self.face_detector:
                    detected_faces, detected_names, detected_priority_flags = self.face_detector.detect_faces(processed_frame)
                    # Eğer detector *herhangi* bir yüz gördüyse, zaman damgasını güncelle
                    if detected_faces:
                        self.last_face_detection_time = time.time()

                # --- Adım 2: Yüz Takipçisi Mantığı (Değiştirildi) ---
                if hasattr(self, 'face_tracker') and self.face_tracker:

                    # --- Senaryo A: Hedef Zaten Kilitli Durumda ---
                    if self.face_target_locked and self.face_tracker.tracking_started:
                        # processed_frame'i tracker günceller (çizim dahil)
                        processed_frame, tracker_result = self.face_tracker.process_frame(frame.copy())

                        # tracker_result None ise ciddi bir hata oluşmuştur, kilidi bırak
                        if tracker_result is None:
                            self.log("!!! Tracker process_frame returned None. Breaking lock.")
                            self.face_target_locked = False
                            self.face_tracker.reset_tracking()
                        else:
                            tracker_success = tracker_result.get('success')

                            if tracker_success:
                                # Başarılı: Hedefi al (servo için)
                                target_object_rect = tracker_result.get('bbox')
                                tracked_name = tracker_result.get('object_class', 'Unknown')
                                if hasattr(self, 'face_detector'):
                                    is_priority = self.face_detector.is_priority_person(tracked_name)
                                # Başarısızlık sayacı artık YOK.
                            else:
                                # Başarısız: Kilidi HEMEN BIRAK (Tekrar deneme YOK)
                                self.log(">>> Tracker update failed. Breaking lock immediately.")
                                self.face_target_locked = False
                                self.face_tracker.reset_tracking()
                                target_object_rect = None # Servo hedefi yok
                                is_priority = False


                    # --- Senaryo B: Hedef Kilitli DEĞİL, Yeni Hedef Aranıyor ---
                    if not self.face_target_locked:
                        # Eğer ANLIK algılamada yüz bulunduysa, tracker'ı başlatmayı dene
                        if detected_faces:
                            # En iyi yüzü seç (önceki kodla aynı mantık)
                            best_face_bbox = None; best_face_name = "Unknown"; best_face_is_priority = False
                            priority_indices = [i for i, p in enumerate(detected_priority_flags) if p]
                            if priority_indices:
                                largest_priority_area = -1; best_idx = -1
                                for i in priority_indices:
                                    x, y, w, h = detected_faces[i]
                                    area = w * h
                                    if area > largest_priority_area:
                                        largest_priority_area = area
                                        best_idx = i
                                if best_idx != -1:
                                    best_face_bbox = detected_faces[best_idx]
                                    best_face_name = detected_names[best_idx]
                                    best_face_is_priority = True
                            else:  # Öncelikli yoksa en büyük
                                largest_area = -1
                                best_idx = -1
                                for i in range(len(detected_faces)):
                                    x, y, w, h = detected_faces[i]
                                    area = w * h
                                    if area > largest_area:
                                        largest_area = area
                                        best_idx = i
                            if best_idx != -1: best_face_bbox = detected_faces[best_idx]; best_face_name = detected_names[best_idx]; best_face_is_priority = False

                            if best_face_bbox:
                                # Tracker'ı başlatmadan önce sıfırla (önlem)
                                if self.face_tracker.tracking_started: self.face_tracker.reset_tracking()
                                init_success = self.face_tracker.init_tracker(frame.copy(), best_face_bbox, best_face_name)
                                if init_success:
                                    self.log(f">>> Tracker initialized and locked on {best_face_name}")
                                    self.face_target_locked = True # KİLİT AKTİF
                                    self.face_tracker_consecutive_failures = 0 # Sayacı sıfırla
                                    target_object_rect = best_face_bbox # Bu frame için hedefi başlatma kutusu yap
                                    is_priority = best_face_is_priority
                                    # Tracker çizimi sonraki frame'de yapacak. İsteğe bağlı: Başlangıç kutusunu çiz
                                    x,y,w,h = [int(v) for v in best_face_bbox]; cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 255), 2); cv2.putText(processed_frame, f"Locked: {best_face_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                else:
                                    self.log("Tracker initialization failed.")
                                    # Kilit hala False, hedef None

                        # Eğer kilitli değilse VE anlık algılamada yüz YOKSA, hiçbir şey yapma.

                    # --- Senaryo C: Detector Timeout Kontrolü (Aynı) ---
                    time_since_last_detection = time.time() - self.last_face_detection_time
                    if self.face_target_locked and time_since_last_detection > self.face_detection_timeout:
                        self.log(f">>> Detector timeout ({time_since_last_detection:.1f}s > {self.face_detection_timeout:.1f}s). Breaking lock.")
                        self.face_target_locked = False
                        if self.face_tracker.tracking_started: self.face_tracker.reset_tracking()
                        target_object_rect = None
                        is_priority = False

                    # Eğer tracker bu frame'de işlemi yapmadıysa (örn. kilit yoktu),
                    # ve anlık algılama varsa, onu çizebiliriz.
                    # (ObjectTracker artık kendisi çizdiği için bu bloğa gerek kalmadı)
                    # if not self.face_tracker.tracking_started and detected_faces:
                    #    processed_frame = self._draw_face_results(...)

                # --- PubSub (Anlık algılanan isimler için - Değişiklik Yok) ---
                if detected_names and current_mode == 'face_recognition':
                    current_names_set = set(filter(None, detected_names))
                    if not hasattr(self, 'last_published_names') or self.last_published_names != current_names_set:
                        if current_names_set: pub.sendMessage('face_detected', detected_names=list(current_names_set))
                        self.last_published_names = current_names_set
            # ================================================================
            # ----- Yüz Tanıma / Bulma SONU -----
            # ================================================================

            # ----- Diğer Modlar (Önceki gibi - Değişiklik Yok) -----
            elif current_mode == 'motion':
                if hasattr(self, 'motion_detector') and self.motion_detector:
                    processed_frame, detected_data = self.motion_detector.process_frame(processed_frame)
                    if isinstance(detected_data, dict):
                        motion_detected = detected_data.get('detected', False); motion_areas = detected_data.get('areas', [])
                        pub.sendMessage('motion_detected', motion_detected=motion_detected)
                        if self.tracking_enabled and motion_areas: target_object_rect = max(motion_areas, key=lambda r: r[2] * r[3]); is_priority = False
            elif current_mode == 'finger':
                if hasattr(self, 'finger_tracker') and self.finger_tracker:
                    processed_frame, detected_data = self.finger_tracker.process_frame(processed_frame)
                    if isinstance(detected_data, dict) and 'gesture' in detected_data: pub.sendMessage('finger_gesture_detected', gesture=detected_data['gesture'])
            elif current_mode == 'age_emotion':
                if hasattr(self, 'age_emotion_detector') and self.age_emotion_detector:
                    processed_frame, detected_data = self.age_emotion_detector.process_frame(processed_frame)
                    if detected_data:
                        pub.sendMessage('age_emotion_detected', results=detected_data)
                        if self.tracking_enabled: largest_face = max(detected_data, key=lambda f: f['bbox'][2] * f['bbox'][3]); target_object_rect = largest_face['bbox']; is_priority = False
            elif current_mode == 'object_detection':
                detected_objects_data = None
                if hasattr(self, 'object_detector') and self.object_detector: _, detected_objects_data = self.object_detector.process_frame(frame.copy())
                if hasattr(self, 'object_tracker') and self.object_tracker:
                    processed_frame, tracker_result = self.object_tracker.process_frame(frame.copy(), detected_objects_data)
                    if tracker_result and tracker_result.get('success'): target_object_rect = tracker_result.get('bbox'); is_priority = tracker_result.get('object_class') == 'person'
            else: # İşlem Yok Modu
                processed_frame = frame.copy()
            # ----- Diğer Modlar Sonu -----


            # --- Takip İşlemi (Genel - Servo Komutları) ---
            # target_object_rect doluysa ve takip aktifse servo komutlarını gönderir
            if self.tracking_enabled and target_object_rect and hasattr(self, 'tracking') and self.tracking:
                if not self.tracking.screen_dimensions or self.tracking.screen_dimensions[0] != width: self.tracking.set_dimensions((width, height))
                self.tracking.track_object(target_object_rect, is_priority)
            # --- Takip Sonu ---


            # --- Görüntü Çevirme ve Gösterme (Değişiklik Yok) ---
            if hasattr(self, 'flip_horizontal') and self.flip_horizontal: processed_frame = cv2.flip(processed_frame, 1)
            if hasattr(self, 'flip_vertical') and self.flip_vertical: processed_frame = cv2.flip(processed_frame, 0)
            try: rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            except cv2.error as cvt_error: self.log(f"RGB Conv Err: {cvt_error}"); rgb_image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape; bytes_per_line = ch * w
            qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            if not pixmap.isNull() and hasattr(self, 'video_label') and self.video_label.width() > 0:
                self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            # ... (else loglama) ...
            else:
                if pixmap.isNull(): self.log("Pixmap Null.")
                elif not hasattr(self, 'video_label'): self.log("no video_label.")
                elif self.video_label.width() <= 0: self.log("video_label zero dim.")
            # --- Gösterme Sonu ---

            # --- FPS ve UI Güncelleme (Değişiklik Yok) ---
            elapsed = time.time() - start_time
            if elapsed > 0.05: QtWidgets.QApplication.processEvents()

        # --- Ana Hata Yakalama Bloğu (Değişiklik Yok) ---
        except Exception as e:
            # ... (mevcut hata yakalama ve fallback frame gösterme kodu) ...
            current_mode_str = getattr(self, 'processing_mode', 'N/A'); self.log(f"!!! Vid Proc Err {current_mode_str}: {e}"); self.log(f"TB:\n{traceback.format_exc()}")
            if frame is not None:
                try:
                    rgb_original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); h_orig, w_orig, ch_orig = rgb_original_frame.shape; bytes_per_line_fb = ch_orig * w_orig
                    qimg_fb = QImage(rgb_original_frame.data, w_orig, h_orig, bytes_per_line_fb, QImage.Format_RGB888); pixmap_fb = QPixmap.fromImage(qimg_fb)
                    if not pixmap_fb.isNull() and hasattr(self, 'video_label'): self.video_label.setPixmap(pixmap_fb.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    else:
                        if hasattr(self, 'video_label'): self.video_label.setText(f"Err {current_mode_str}"); self.video_label.setStyleSheet("background-color: red; color: white; font-size: 10pt;")
                except Exception as fallback_e:
                    self.log(f"Fallback Err: {fallback_e}");
                    if hasattr(self, 'video_label'): self.video_label.setText("Disp Err"); self.video_label.setStyleSheet("background-color: darkred; color: yellow; font-size: 12pt;")
            else:
                if hasattr(self, 'video_label'): self.video_label.setText("Frame Read Err"); self.video_label.setStyleSheet("background-color: black; color: red; font-size: 12pt;")

    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    # update_frame SONU
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        
    def send_to_llm(self):
        """Send text input to the LLM API."""
        input_text = self.io_input.text().strip()
        if input_text:
            # Only allow if not already processing
            if self.is_processing_request:
                self.log("⚠️ Already processing a request. Please wait.")
                return
                
            self.log(f"Sending to LLM: {input_text}")
            
            # Set processing flag
            self.is_processing_request = True
            
            # Update status indicator
            self.update_status_signal.emit("Processing Request", "processing")
            
            # UI'da "Thinking..." mesajını göster (ana thread'de)
            self.update_thinking_signal.emit()
            
            # Arka plan thread'inde LLM isteğini başlat
            threading.Thread(target=self._process_llm_request, args=(input_text,), daemon=True).start()

            # Clear the input field
            self.io_input.clear()
        else:
            self.log("Input text is empty")
    
    def llm_request(self, input_text):
        """Send an HTTP request to the LLM API."""
        # Olmayan bir durumda kapanmayı önlemek için tüm fonksiyonu try bloğuna alıyoruz
        try:
            # Ollama API URL'sini düzenleme
            api_url = self.ollama_url
            if not api_url.endswith('/generate'):
                if api_url.endswith('/api'):
                    api_url += '/generate'
                else:
                    api_url += '/api/generate'

            # Kullanılacak model adını alma
            selected_model = None
            for btn in self.model_button_group.buttons():
                if btn.isChecked():
                    selected_model = btn.text()
                    break
            for btn in self.personality_button_group.buttons():
                if btn.isChecked():
                    selected_personality = btn.text()
                    break
            
            # Günlük için model bilgisini yazdırma
            self.log(f"Using LLM model: {selected_model}")
            self.log(f"API URL: {api_url}")
            
            # Ollama API formatına uygun payload hazırlama
            payload = {
                'model': selected_model,
                'prompt': input_text,
                'stream': False  # Streaming yanıtı devre dışı bırakıyoruz
            }
            
            # JSON Content-Type başlığı ekleme
            headers = {
                'Content-Type': 'application/json'
            }
            
            self.log(f"Sending request to Ollama API...")
            
            # İstek yapılırken oluşabilecek tüm hataları yakalıyoruz
            try:
                # İstek timeout'unu uzatıyoruz (büyük modeller için)
                response = requests.post(api_url, json=payload, headers=headers, timeout=120)
                
                # HTTP durum kodunu kontrol et
                self.log(f"API response status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        # Yanıt uzunluğu hakkında bilgi verelim
                        resp_length = len(response.text)
                        self.log(f"Received response ({resp_length} bytes)")
                        
                        # 10 MB'dan büyük yanıtları engelle
                        if resp_length > 10_000_000:  # 10 MB yaklaşık
                            error_msg = "Response too large to process"
                            self.update_output_signal.emit(error_msg, True)
                            self.log(f"Error: {error_msg}")
                            return
                        
                        # JSON yanıtı ayrıştır
                        data = response.json()
                        output = data.get('response', '')
                        
                        # Yanıt uzunluğunu kontrol et
                        if len(output) > 1_000_000:  # ~1 MB metin
                            output = output[:100000] + "\n\n... (yanıt çok uzun olduğu için kısaltıldı) ..."
                        
                        # Yanıtı ana thread'e gönder
                        self.update_output_signal.emit(output, True)
                        
                        # Event sistemine de gönder
                        pub.sendMessage('llm_response', response_text=output)
                        self.log("LLM response processed successfully")
                        
                    except ValueError as json_err:
                        self.log(f"JSON decode error: {json_err}")
                        
                        # İçeriğin ilk birkaç karakterini kontrol et
                        content_preview = response.text[:100]
                        self.log(f"Response preview: {content_preview}")
                        
                        # Satır satır yanıt işleme denemesi
                        try:
                            lines = response.text.strip().split('\n')
                            combined_response = ""
                            
                            for line in lines:
                                if line.strip():
                                    try:
                                        line_data = json.loads(line)
                                        if 'response' in line_data:
                                            combined_response += line_data['response']
                                    except json.JSONDecodeError:
                                        continue  # Bu satırı atlayıp diğerine geç
                            
                            if combined_response:
                                # Ana thread'e yanıtı gönder
                                self.update_output_signal.emit(combined_response, True)
                                pub.sendMessage('llm_response', response_text=combined_response)
                                self.log("Combined streaming responses successfully")
                            else:
                                # Ham veriyi işleyip gönder (en fazla 1000 karakter)
                                truncated_text = response.text[:1000] + ("..." if len(response.text) > 1000 else "")
                                error_msg = f"Could not parse response as JSON. Raw output:\n{truncated_text}"
                                self.update_output_signal.emit(error_msg, True)
                        except Exception as inner_e:
                            self.log(f"Error processing streaming response: {inner_e}")
                            error_msg = f"Error processing response: {str(json_err)}"
                            self.update_output_signal.emit(error_msg, True)
                else:
                    self.log(f"LLM API error: {response.status_code}")
                    self.log(f"Error details: {response.text[:200]}")
                    
                    # Hata mesajlarını ana thread'e gönder
                    if response.status_code == 404:
                        error_msg = "Error: API endpoint not found. Check if Ollama is running and the URL is correct."
                    elif response.status_code == 400:
                        error_msg = "Error: Bad request. Check if the model name is correct and available."
                    elif response.status_code == 500:
                        error_msg = "Error: Server error. Check Ollama server logs for details."
                    else:
                        error_msg = f"Error: HTTP {response.status_code} - {response.text[:200]}"
                    
                    self.update_output_signal.emit(error_msg, True)
            except requests.exceptions.Timeout:
                self.log("Error: API request timed out. The model might be too slow or unavailable.")
                self.update_output_signal.emit("Error: API request timed out. The model might be too slow or unavailable.", True)
            except requests.exceptions.ConnectionError:
                self.log("Error: Could not connect to Ollama API. Make sure Ollama is running.")
                self.update_output_signal.emit("Error: Could not connect to Ollama API. Make sure Ollama is running.", True)
            except requests.exceptions.RequestException as req_err:
                self.log(f"Error sending request: {req_err}")
                self.update_output_signal.emit(f"Error sending request: {req_err}", True)
        except Exception as e:
            # Genel hata yakalama - uygulamanın çökmesini engeller
            err_message = f"Unexpected error in LLM request: {e}"
            try:
                self.log(err_message)
                self.update_output_signal.emit(err_message, True)
            except:
                print(err_message)  # En kötü durumda konsola yaz
            
        # İşlem tamamlandığında UI'yı güncelle
        QtCore.QCoreApplication.processEvents()
        
    # Yeni metod: Ana threadde çalışacak UI güncelleme fonksiyonu
    @QtCore.pyqtSlot(str, bool)
    def update_output_text(self, text, clear_first=False):
        """Main thread'de çalışacak UI güncelleme metodu"""
        try:
            self.log(f"Updating output text: {text[:30]}{'...' if len(text) > 30 else ''}")
            
            if clear_first:
                self.io_output.clear()
            
            # Metni biçimlendirerek göster (Gemma cevaplarını büyük fontla)
            # HTML formatlaması uygula
            formatted_text = self.format_llm_response(text)
            self.io_output.insertHtml(formatted_text)
            
            # Otomatik kaydırma
            cursor = self.io_output.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            self.io_output.setTextCursor(cursor)
            
            # İşlemi yenile
            QtWidgets.QApplication.processEvents()
            
        except Exception as e:
            self.log(f"Error updating output text: {e}")
            # Hata durumunda düz metin olarak ekle
            self.io_output.append(text)

    def format_llm_response(self, text):
        """
        LLM yanıtındaki [cmd:animasyon] komutlarını ayıklar,
        animasyonları gönderir ve temizlenmiş metni döndürür.
        """
        if not text:
            return ""
    
        # [cmd:...] komutlarını ayıkla ve animasyonları gönder
        cmd_pattern = r"\[cmd:([a-zA-Z0-9_]+)\]"
        commands = re.findall(cmd_pattern, text)
        for animation_name in commands:
            self.send_animation(animation_name)
        text = re.sub(cmd_pattern, '', text).strip()
    
        # Eğer metin zaten HTML tag'ı ile başlıyorsa encode etme
        if text.strip().startswith("<") and text.strip().endswith(">"):
            return text
    
        # Metni HTML güvenli hale getir
        import html
        safe_text = html.escape(text)
    
        # Basit markdown dönüştürme
        safe_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe_text)
        safe_text = safe_text.replace("\n", "<br/>")
        safe_text = safe_text.replace("* ", "• ")
        safe_text = safe_text.replace("😊", "&#128522;")
    
        return f"<div style='font-size: 14pt;'>{safe_text}</div>"

    def update_log(self, msg):
        """Update the log display."""
        self.log(msg)
        
    def log(self, message):
        """Append a message to the log display."""
        if hasattr(self, 'log_text') and self.log_text is not None:
            try:
                self.log_text.append(message)
            except RuntimeError:
                print(f"LOG (after close): {message}")
        else:
            print(f"LOG: {message}")

    def stop_stt_if_no_input_after_ww_timeout(self):
        """Eğer wake word sonrası komut gelmezse STT'yi durdurur ve WW'yi yeniden başlatır."""
        if self.audio_manager.speech_active and self.speech_triggered_by_wake_word:
            self.log("DeskGUI: Wake word sonrası komut zaman aşımı. STT durduruluyor...")
            # STT'yi durdur, AudioManager içindeki stop WW'yi (checkbox işaretliyse) yeniden başlatır
            self.audio_manager.stop_speech_recognition(restart_wake_word_if_enabled=True)
        self.speech_triggered_by_wake_word = False # Bayrağı her durumda sıfırla

    def restart_wake_word_listener_safely(self):
        """Wake word dinleyiciyi güvenli bir şekilde yeniden başlatır."""
        # Bu metod önceki yanıttaki gibi kalabilir.
        ww_should_be_running = False
        if hasattr(self, 'wake_word_checkbox') and self.wake_word_checkbox.isChecked():
            ww_should_be_running = True

        # Wake word isteniyorsa VE dedektör varsa VE çalışmıyorsa
        if ww_should_be_running and self.wake_word_detector and not self.wake_word_detector.is_running:
             # STT'nin kapalı olduğundan emin ol
             if self.audio_manager.speech_active:
                  self.log("WW başlatılmadan önce STT durduruluyor...")
                  self.audio_manager.stop_speech_recognition()
                  time.sleep(0.1) # Çok kısa bekleme

             self.log("Wake word dinleyici güvenli modda yeniden başlatılıyor.")
             self.wake_word_detector.start_listening()
             # Buton metnini güncelle
             if hasattr(self, 'toggle_mic_button'): self.toggle_mic_button.setText("Stop Wake Word")

    def closeEvent(self, event):
        """Release resources on application close."""
    # Tüm thread havuzlarını düzgün şekilde kapat
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
    
        # face_detector thread havuzunu kapat
        if hasattr(self, 'face_detector') and hasattr(self.face_detector, 'executor'):
            self.face_detector.executor.shutdown(wait=False) 
        self.disconnect_from_robot()
        self.audio_manager.stop_speech_recognition()
        event.accept()
        
    def send_servo_command(self, servo_id, percentage):
        """Send a servo movement command to the robot."""
        if not self.stream_connected:
            self.log("Not connected to robot.")
            return
            
        self.log(f"Moving {servo_id} servo by {percentage}%")
        self.command_sender.send_command('servo_move', {
            'identifier': servo_id,
            'percentage': percentage,
            'absolute': False
        })
        
    def center_servos(self):
        """Center both pan and tilt servos."""
        if not self.stream_connected:
            self.log("Not connected to robot.")
            return
            
        self.log("Centering servos.")
        self.command_sender.send_command('servo_move', {
            'identifier': 'pan',
            'percentage': 0,
            'absolute': True
        })
        self.command_sender.send_command('servo_move', {
            'identifier': 'tilt',
            'percentage': 0,
            'absolute': True
        })
        
    def run_animation(self):
        """Run the selected animation on the robot."""
        if not self.stream_connected:
            self.log("Not connected to robot.")
            return
            
        animation = self.animation_combo.currentText()
        self.log(f"Running animation: {animation}")
        self.command_sender.send_command('animate', {
            'action': animation
        })
        
    def send_speech(self):
        """Send speech text to be spoken by the robot."""
        if not self.stream_connected:
            self.log("Not connected to robot.")
            return
            
        message = self.speech_input.text().strip()
        if not message:
            return
            
        self.log(f"Sending speech to robot: {message}")
        self.command_sender.send_command('speak', {
            'message': message
        })

    @QtCore.pyqtSlot(bool, str, str) # training_complete_signal'e karşılık gelir
    def handle_training_completion(self, success, message, backup_file_path):
        """Handles the completion of the background training thread."""
        self.log(f"Training thread finished. Success: {success}. Message: {message}")

        # Eğer eğitim başarılıysa veya test başarılıysa sadece bilgi ver
        if success:
            QtWidgets.QMessageBox.information(self, "Training Complete", message)
        # Eğer eğitim/test başarısız olduysa ve backup varsa, kullanıcıya sor
        elif backup_file_path and os.path.exists(backup_file_path):
            reply = QtWidgets.QMessageBox.question(
                self,
                "Training Issue",
                f"{message}\n\nThe previous model was backed up.\nDo you want to restore the previous model?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                try:
                    shutil.copy2(backup_file_path, self.encodings_file)
                    self.log("Restoring previous model...")
                     # Stop current detector thread and reload
                    if hasattr(self, 'face_detector') and self.face_detector:
                        self.face_detector.stop_processing_thread()
                    self.face_detector = FaceDetector(self.encodings_file) # Reload with restored file
                    self.update_priority_combo_signal.emit() # Update combo again after restoring
                    self.log("Previous model restored.")
                    QtWidgets.QMessageBox.information(self, "Restore Complete", "Previous face recognition model restored.")
                except Exception as e:
                    self.log(f"Error restoring backup: {e}")
                    QtWidgets.QMessageBox.warning(self, "Restore Error", f"Could not restore backup: {e}")
        # Eğer eğitim başarısız olduysa ama backup yoksa sadece hata mesajı göster
        else:
             QtWidgets.QMessageBox.warning(self, "Training Failed", message)

        # Ensure the face detector thread is running (it might have been stopped)
        if hasattr(self, 'face_detector') and not self.face_detector.processing_active:
            self.log("Restarting face detector thread after training completion handler.")
            self.face_detector.start_processing_thread()

    def play_buzzer(self):
        """Play the selected sound on the robot's buzzer."""
        if not self.stream_connected:
            self.log("Not connected to robot.")
            return
            
        song = self.buzzer_combo.currentText()
        self.log(f"Playing sound: {song}")
        self.command_sender.send_command('buzzer', {
            'song': song
        })
        
    def train_model(self):
        """Start training the face recognition model."""
        self.log("Training face recognition model. This may take some time...")
        
        # Run in a thread to avoid blocking the UI
        threading.Thread(target=self._train_model_thread).start()
        
    # YENİ (Sadeleştirilmiş) _train_model_thread METODU
    # ----- DeskGUI._train_model_thread Metodu İçinde -----

    def _train_model_thread(self):
        """Background thread ONLY for training and saving the model file."""
        backup_file = ""
        training_message = ""
        training_success = False
        new_encodings_file = self.encodings_file

        try:
            self.log_signal.emit("Starting face recognition model training...")

            # Modülün import edildiğini kontrol et (TRAIN_MODEL_AVAILABLE kullanarak)
            if not TRAIN_MODEL_AVAILABLE:
                raise ImportError("TrainModel module was not loaded correctly at startup.")

            # --- (Backup kodları aynı kalır) ---
            if os.path.exists(new_encodings_file):
                backup_file = f"{new_encodings_file}.bak-{int(time.time())}"
                try:
                    shutil.copy2(new_encodings_file, backup_file)
                    self.log_signal.emit(f"Backed up existing encodings to {backup_file}")
                except Exception as e:
                    self.log_signal.emit(f"Warning: Could not backup encodings: {e}")

            # TrainModel'ı log_signal ile başlat
            trainer = TrainModel(
                log_signal=self.log_signal, # <<< SİNYALİ BURADA VER
                dataset='dataset',
                output=new_encodings_file,
                detection_method='hog',
                jitters=2,
                preserve_priority=True,
                priority_file="priority_persons.json"
            )

            count = trainer.train() # Eğitimi çalıştır

            # --- (Sonuç kontrolü ve mesaj oluşturma kodları aynı kalır) ---
            if count == 0:
                training_message = "[TrainModel] No valid faces found or encoded. Encodings file might be empty or unchanged."
                training_success = False
                # self.log_signal.emit(training_message) # Zaten TrainModel içinde loglandı
            elif not os.path.exists(new_encodings_file):
                 training_message = f"[TrainModel] Error: Encodings file '{new_encodings_file}' was not created/moved correctly after training."
                 training_success = False
                 # self.log_signal.emit(training_message) # Zaten TrainModel içinde loglandı
            else:
                 if os.path.getsize(new_encodings_file) < 100:
                     training_message = f"[TrainModel] Warning: Encodings file '{new_encodings_file}' created but seems empty."
                     training_success = False
                     # self.log_signal.emit(training_message) # Zaten TrainModel içinde loglandı
                 else:
                     training_message = f"Model training process finished. Processed {count} valid encodings. Encodings saved to '{new_encodings_file}'."
                     training_success = True
                     # self.log_signal.emit(training_message) # Zaten TrainModel içinde loglandı

        except ImportError as e:
             training_message = f"Error importing/using training module: {e}"
             training_success = False
             self.log_signal.emit(training_message)
        except Exception as e:
            training_message = f"Error during model training thread setup/execution: {e}\n{traceback.format_exc()}"
            training_success = False
            self.log_signal.emit(training_message)

        finally:
            # Ana thread'e sonucu bildir
            self.training_complete_signal.emit(training_success, training_message, backup_file if not training_success and backup_file and os.path.exists(backup_file) else "")
            
    def _test_recognition(self):
        """Test if the newly trained model can recognize faces"""
        if not self.face_detector or not self.face_detector.data:
            return False
            
        # Check that the model contains encodings
        encodings_count = len(self.face_detector.data.get("encodings", []))
        if encodings_count == 0:
            self.log("WARNING: Model contains no encodings!")
            return False
            
        # Check a sample image from the dataset directory
        try:
            dataset_dir = "dataset"
            if os.path.exists(dataset_dir):
                person_dirs = [d for d in os.listdir(dataset_dir) 
                              if os.path.isdir(os.path.join(dataset_dir, d))]
                
                if not person_dirs:
                    self.log("No person directories found in dataset.")
                    return True  # Can't test without sample images
                    
                # Take the first person as a test case
                test_person = person_dirs[0]
                test_dir = os.path.join(dataset_dir, test_person)
                test_images = [f for f in os.listdir(test_dir) 
                              if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                if not test_images:
                    self.log(f"No test images found for {test_person}.")
                    return True  # Can't test without sample images
                
                # Load and test a sample image
                test_image_path = os.path.join(test_dir, test_images[0])
                test_image = cv2.imread(test_image_path)
                
                if test_image is None:
                    self.log(f"Could not load test image: {test_image_path}")
                    return True  # Can't test without valid image
                
                # Run face detection and recognition on test image
                faces, names, _ = self.face_detector.detect_faces(test_image)
                
                if len(faces) == 0:
                    self.log(f"Warning: No faces detected in test image {test_image_path}")
                    return True  # Inconclusive test
                
                # Check if any face was recognized as the correct person
                recognized = any(name == test_person for name in names)
                
                if recognized:
                    self.log(f"Recognition test passed: {test_person} correctly identified")
                    return True
                else:
                    self.log(f"Recognition test failed: {test_person} not recognized in test image")
                    return False
                    
        except Exception as e:
            self.log(f"Error during recognition test: {e}")
            
        return True  # Default to true if testing fails for any reason

    def _check_remote_speech_status(self):
        """Check if speech recognition is active on the remote server and update UI accordingly"""
        # Yerel modda ise hiçbir işlem yapma
        if not self.using_bluetooth_audio and hasattr(self, 'audio_mode') and self.audio_mode == "direct":
            return  # Yerel modda bu kontrolü tamamen atla
        
        time.sleep(3)  # Allow time for the server to start
        try:
            response = requests.get(f"http://{self.bluetooth_server}:8098/status", timeout=2)
            if response.status_code == 200:
                status = response.json()
                if status.get("listening", False):
                    self.speech_active = True
                    self.log("Speech recognition is already active on the server")
        except Exception as e:
            # Sadece Bluetooth modunda ise hata mesajını göster
            if self.using_bluetooth_audio or (hasattr(self, 'audio_mode') and self.audio_mode == "bluetooth"):
                self.log(f"Could not check remote speech status: {e}")

    def resizeEvent(self, event):
        """Adjust size policies dynamically based on the current window size."""
        total_width = self.width()
        total_height = self.height()
        # Optionally adjust font sizes or control sizes dynamically here.
        # For simplicity, we set new maximum widths:
        self.control_panel.setMaximumWidth(total_width // 3)
        self.log_text.setMaximumWidth(total_width // 3)
        self.io_input.setMaximumWidth(total_width // 3)
        self.io_output.setMaximumWidth(total_width // 3)
        super(DeskGUI, self).resizeEvent(event)
        
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for controlling the robot."""
        if not self.stream_connected:
            super(DeskGUI, self).keyPressEvent(event)
            return
            
        # Arrow keys for movement
        if event.key() == QtCore.Qt.Key_Up:
            self.send_servo_command('tilt', 10)
        elif event.key() == QtCore.Qt.Key_Down:
            self.send_servo_command('tilt', -10)
        elif event.key() == QtCore.Qt.Key_Left:
            self.send_servo_command('pan', 10)
        elif event.key() == QtCore.Qt.Key_Right:
            self.send_servo_command('pan', -10)
        # Space to center
        elif event.key() == QtCore.Qt.Key_Space:
            self.center_servos()
        # S key to start/stop speech recognition
        elif event.key() == QtCore.Qt.Key_S:
            if self.audio_manager.speech_active:
                self.audio_manager.stop_speech_recognition()
                self.log("Speech recognition stopped (keyboard shortcut)")
            else:
                self.audio_manager.start_speech_recognition()
                self.log("Speech recognition started (keyboard shortcut)")
        # T key to toggle tracking
        elif event.key() == QtCore.Qt.Key_T:
            self.tracking_checkbox.setChecked(not self.tracking_checkbox.isChecked())
        else:
            super(DeskGUI, self).keyPressEvent(event)
            
    def capture_screenshot(self):
        """Capture a screenshot of the current video frame."""
        if not self.video_stream:
            self.log("No video stream active")
            return
            
        frame = self.video_stream.read()
        if frame is None:
            self.log("No frame available to capture")
            return
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        
        try:
            cv2.imwrite(filename, frame)
            self.log(f"Screenshot saved to {filename}")
        except Exception as e:
            self.log(f"Error saving screenshot: {e}")
            
    # ----- DeskGUI Sınıfı İçinde -----

    def add_face_from_frame(self):
        """Capture the current frame and add it to the face database."""
        # Face detector var mı ve cascade yüklü mü kontrol et
        if not hasattr(self, 'face_detector') or self.face_detector is None or \
           not hasattr(self.face_detector, 'cascade') or self.face_detector.cascade is None or \
           self.face_detector.cascade.empty():
            self.log("Error: Face detector cascade is not loaded or unavailable.")
            QtWidgets.QMessageBox.critical(self, "Detector Error", "Face detector cascade is not loaded correctly. Cannot add face.")
            return

        if not self.video_stream:
            self.log("No video stream active")
            return

        frame = self.video_stream.read()
        if frame is None:
            self.log("No frame available to capture")
            return

        # Prompt for name
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Add Face", "Enter person's name:"
        )

        if not ok or not name:
            self.log("Face adding cancelled by user.")
            return

        # Ensure dataset directory exists
        dataset_dir = "dataset"
        person_dir = os.path.join(dataset_dir, name)
        try:
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
        except OSError as e:
            self.log(f"Error creating directory {person_dir}: {e}")
            QtWidgets.QMessageBox.critical(self, "Directory Error", f"Could not create directory for {name}: {e}")
            return

        faces = [] # faces listesini başta tanımla
        try:
            # Detect face in the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- YENİ KONTROL: Görüntü geçerli mi? ---
            if gray is None or gray.size == 0:
                 self.log("Error: Failed to convert frame to grayscale or frame is empty.")
                 QtWidgets.QMessageBox.warning(self, "Image Error", "Could not process the video frame.")
                 return
            # --- KONTROL SONU ---

            # --- detectMultiScale'i try-except içine al ---
            try:
                faces = self.face_detector.cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
            except cv2.error as cv_err:
                 self.log(f"!!! OpenCV Error during detectMultiScale: {cv_err}")
                 self.log(traceback.format_exc())
                 QtWidgets.QMessageBox.critical(self, "Detection Error", f"OpenCV error during face detection: {cv_err}")
                 return
            except Exception as detect_e: # Diğer olası hatalar
                 self.log(f"!!! Unexpected Error during detectMultiScale: {detect_e}")
                 self.log(traceback.format_exc())
                 QtWidgets.QMessageBox.critical(self, "Detection Error", f"Unexpected error during face detection: {detect_e}")
                 return
            # --- try-except SONU ---

            if len(faces) == 0:
                self.log("No face detected in the captured frame.")
                QtWidgets.QMessageBox.information(self, "No Face Detected", "Could not detect a face in the current frame. Try again.")
                return

            # Birden fazla yüz varsa kullanıcıyı uyar (isteğe bağlı ama iyi bir pratik)
            if len(faces) > 1:
                 self.log(f"Warning: Multiple faces ({len(faces)}) detected. Saving the frame anyway.")
                 # İsteğe bağlı olarak kullanıcıya hangi yüzü kaydedeceğini sorabilir veya işlemi iptal edebilirsiniz.
                 # Şimdilik devam edelim.

            # Save the image
            timestamp = int(time.time())
            filename = os.path.join(person_dir, f"{timestamp}.jpg")

            # --- Yazma işlemini de try içine al ---
            try:
                success_write = cv2.imwrite(filename, frame) # Orijinal renkli frame'i kaydet
                if not success_write:
                    self.log(f"!!! Error: Failed to write face image to {filename}")
                    QtWidgets.QMessageBox.warning(self, "Save Error", f"Could not save the face image to {filename}.")
                    return

                self.log(f"Face image saved to {filename}")

                # Ask if user wants to train the model now
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Train Model",
                    "Face image saved successfully.\nDo you want to train the face recognition model now?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )

                if reply == QtWidgets.QMessageBox.Yes:
                    self.train_model() # Bu sadece thread'i başlatır, donma burada olmamalı

            except Exception as save_e:
                 self.log(f"Error saving face image or asking to train: {save_e}")
                 self.log(traceback.format_exc())
                 QtWidgets.QMessageBox.critical(self, "Save/Train Error", f"An error occurred after detecting the face: {save_e}")

        except Exception as e:
            # Genel hata yakalama (cvtColor vb. için)
            self.log(f"Error during add_face_from_frame processing: {e}")
            self.log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "Error", f"An unexpected error occurred while adding the face: {e}")

    @QtCore.pyqtSlot(bool, str, str)
    def handle_training_completion(self, training_successful, message, backup_file_path):
        """Handles the completion of the training thread and updates the detector."""
        self.log(f"Training thread finished. Success: {training_successful}. Message: {message}")

        reload_needed = False
        restore_done = False

        # Eğer eğitim işlemi başarılıysa (dosya oluşturuldu ve veri var gibi görünüyor)
        if training_successful:
            self.log("Training process seems successful. Reloading FaceDetector data...")
            reload_needed = True
            # Başarı mesajını göster
            QtWidgets.QMessageBox.information(self, "Training Finished", message)

        # Eğer eğitim işlemi başarısız olduysa ve backup dosyası varsa
        elif backup_file_path:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Training Issue",
                f"{message}\n\nThe previous model was backed up.\nDo you want to restore the previous model?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                try:
                    shutil.copy2(backup_file_path, self.encodings_file)
                    self.log("Restoring previous model...")
                    reload_needed = True # Geri yüklenen veriyi de yüklemek gerek
                    restore_done = True
                    QtWidgets.QMessageBox.information(self, "Restore Complete", "Previous face recognition model restored.")
                except Exception as e:
                    self.log(f"Error restoring backup: {e}")
                    QtWidgets.QMessageBox.warning(self, "Restore Error", f"Could not restore backup: {e}")
            else:
                 # Kullanıcı geri yüklemek istemedi, mevcut (muhtemelen hatalı) dosyayı yüklemeyi dene
                 self.log("User chose not to restore backup. Attempting to load current encodings file anyway.")
                 reload_needed = True
        # Eğitim başarısız ve backup yoksa
        else:
             QtWidgets.QMessageBox.warning(self, "Training Failed", message)
             # Yine de mevcut dosyayı yüklemeyi deneyebiliriz, belki kısmen çalışır
             reload_needed = True

        # Eğer veri yüklemesi gerekiyorsa (başarılı eğitim veya geri yükleme sonrası)
        if reload_needed and hasattr(self, 'face_detector') and self.face_detector:
            self.log("Issuing reload data command to FaceDetector...")
            # FaceDetector'ın reload_data metodu verileri yeniden yüklemeli
            # Bu metot thread-safe olmalı veya ana thread'de çalıştırılmalı
            # Mevcut reload_data sadece load_encodings çağırıyor, bu I/O yapar ama thread yaratmaz.
            # Ana thread'de olduğumuz için doğrudan çağırmak şimdilik kabul edilebilir.
            success_reloading = self.face_detector.reload_data()
            if success_reloading:
                 self.log("FaceDetector data reloaded successfully.")
                 # Veri yüklendikten sonra combobox'ı güncelle
                 self.update_priority_persons_combo() # Doğrudan çağrı (ana thread)
            else:
                 self.log("FaceDetector failed to reload data.")
                 # Eğer geri yükleme yapıldıysa ve yükleme başarısız olduysa bu ciddi bir sorun
                 if restore_done:
                      QtWidgets.QMessageBox.critical(self, "Critical Error", "Restored backup file, but failed to reload data into detector!")
                 else:
                      QtWidgets.QMessageBox.warning(self, "Load Error", "Failed to load new encoding data into the detector.")

        # Eğitim thread'i bittikten sonra FaceDetector thread'inin çalıştığından emin ol
        # (Eğer eğitim sırasında bir şekilde durduysa)
        if hasattr(self, 'face_detector') and not self.face_detector.processing_active:
            self.log("Restarting face detector thread after training handler (if stopped).")
            self.face_detector.start_processing_thread()

    def create_context_menu(self, event):
        """Create a context menu for the video display."""
        try:
            menu = QtWidgets.QMenu(self)
            
            # Add actions
            screenshot_action = menu.addAction("Capture Screenshot")
            screenshot_action.triggered.connect(self.capture_screenshot)
            
            add_face_action = menu.addAction("Add Face to Database")
            add_face_action.triggered.connect(self.add_face_from_frame)
            
            # Only enable face-related actions if face detection is available
            add_face_action.setEnabled(hasattr(self, 'face_detector'))
            
            # Add toggle actions
            menu.addSeparator()
            
            flip_h_action = menu.addAction("Flip Horizontally")
            flip_h_action.setCheckable(True)
            # Güvenli erişim - değişken yoksa False kullan
            flip_h_action.setChecked(getattr(self, 'flip_horizontal', False))
            flip_h_action.toggled.connect(self.toggle_flip_horizontal)
            
            flip_v_action = menu.addAction("Flip Vertically")
            flip_v_action.setCheckable(True)
            # Güvenli erişim - değişken yoksa False kullan
            flip_v_action.setChecked(getattr(self, 'flip_vertical', False))
            flip_v_action.toggled.connect(self.toggle_flip_vertical)
            
            # Display the menu
            menu.exec_(event.globalPos())
        except Exception as e:
            self.log(f"Kontekst menü oluşturulurken hata: {e}")
        
    def mousePressEvent(self, event):
        """Handle mouse clicks."""
        if event.button() == QtCore.Qt.RightButton:
            self.create_context_menu(event)
        else:
            super(DeskGUI, self).mousePressEvent(event)
            
    def apply_wake_word(self):
        """Apply the wake word from the input field"""
        wake_word = self.wake_word_input.text().strip().lower()
        if wake_word:
            self.audio_manager.wake_word = wake_word
            self.log(f"Wake word set to: '{wake_word}'")
        else:
            self.log("Wake word cannot be empty. Using default.")
            self.wake_word_input.setText(self.audio_manager.wake_word)
    
    def set_priority_order(self):
        """Set the priority order for the selected person"""
        name = self.priority_persons_combo.currentText()
        order = self.priority_order_input.value()
        
        if name:
            if self.face_detector.set_priority_order(name, order):
                self.log(f"Set priority order for {name} to {order}")
            else:
                self.log(f"{name} is not a priority person")
    
   
    @QtCore.pyqtSlot(str)
    def handle_llm_response_received(self, response_text):
        """
        LLM'den yanıt alındığında çağrılır (pubsub abonesi).
        Yanıtı GUI'de gösterir, TTS için hazırlar ve seçili TTS motoruna göre seslendirmeyi başlatır.
        TTS'in bitmesini beklemek üzere bayrağı ayarlar.
        """
        self.log(f"LLM yanıtı alındı (handle_llm_response_received sinyal/pubsub).")

        # GUI'de göstermek için formatla (komutları ayıklar ve HTML yapar)
        clean_text_for_display = self.format_llm_response(response_text)
        self.update_output_signal.emit(clean_text_for_display, True) # GUI'de göster (bu zaten thread-safe sinyal)

        # TTS için metni temizle (emoji, vb. kaldırır, motor bağımsız)
        text_for_tts = self.clean_text_for_tts(response_text)

        if text_for_tts:
            self.log(f"LLM yanıtı TTS için hazırlanıyor: {text_for_tts[:50]}...")

            # Seçili TTS motorunu ve otomatik dil ayarını al
            current_tts_engine = getattr(self, 'tts_engine_type', 'piper') # Varsayılan piper
            use_auto_lang = getattr(self, 'use_auto_language', True)

            if current_tts_engine == 'xtts':
                self.log(f"XTTS motoru seçili. Otomatik dil algılama atlanıyor (xTTS dil seçimi yok). Varsayılan XTTS dili kullanılacak.")
                success_speaking = self.speak_text_locally(text_for_tts)
                if success_speaking:
                    self.llm_response_pending_tts_completion = True
                else:
                    self.log("XTTS ile seslendirme başlatılamadı.")
                    self.llm_response_pending_tts_completion = False
                    self.restart_stt_if_needed_after_llm()
            elif use_auto_lang:
                self.log(f"{current_tts_engine.upper()} motoru için otomatik dil algılama ve seslendirme başlatılıyor.")
                self.speak_text_with_auto_language(text_for_tts)
                self.llm_response_pending_tts_completion = True
            else:
                self.log(f"{current_tts_engine.upper()} motoru için varsayılan/seçili dil ile seslendirme başlatılıyor.")
                success_speaking = self.speak_text_locally(text_for_tts)
                if success_speaking:
                    self.llm_response_pending_tts_completion = True
                else:
                    self.log(f"{current_tts_engine.upper()} ile seslendirme başlatılamadı.")
                    self.llm_response_pending_tts_completion = False
                    self.restart_stt_if_needed_after_llm()

            self.last_llm_response_text_for_tts = text_for_tts

        else:
            self.log("LLM yanıtı temizlendikten sonra boş kaldı, TTS atlanıyor.")
            self.llm_response_pending_tts_completion = False
            self.last_llm_response_text_for_tts = ""
            self.restart_stt_if_needed_after_llm()

        # Robot'a log gönder (isteğe bağlı)
        if self.stream_connected:
            log_text_for_robot = response_text[:200] + ("..." if len(response_text) > 200 else "")
            try:
                self.command_sender.send_command('log', {
                    'message': f"LLM Response: {log_text_for_robot}",
                    'level': 'info'
                })
            except Exception as e:
                self.log(f"LLM yanıt logunu robota gönderirken hata: {e}")


    def toggle_speech_to_llm(self, checked):
        """Toggle sending speech input directly to LLM."""
        self.use_speech_for_llm = checked
        self.log(f"Speech to LLM: {'Enabled' if checked else 'Disabled'}")
        
        # Update the recognition status indicator based on whether speech-to-LLM is enabled
        if checked:
            self.recognition_indicator.setText("Ready for LLM input")
            self.recognition_indicator.setStyleSheet("color: green;")
        else:
            self.recognition_indicator.setText("LLM input disabled")
            self.recognition_indicator.setStyleSheet("color: gray;")
        
        # Aktifse ve konuşma tanıma açık değilse başlat
        if checked and not self.audio_manager.speech_active:
            self.start_speech_recognition()

    def update_mic_button_and_indicator_based_on_state(self):
        """Mevcut STT ve WW durumlarına göre mikrofon butonunu ve göstergesini günceller."""
        is_stt_active = hasattr(self.audio_manager, 'speech_active') and self.audio_manager.speech_active
        is_ww_listening = hasattr(self.audio_manager, 'is_wake_word_listening') and self.audio_manager.is_wake_word_listening

        button_updated = False
        indicator_updated = False

        if hasattr(self, 'toggle_mic_button'):
            current_text = self.toggle_mic_button.text()
            new_text = ""
            if is_stt_active:
                new_text = "Mikrofonu Kapat (STT)"
            elif is_ww_listening:
                new_text = "Mikrofonu Kapat (WW)"
            else:
                new_text = "Mikrofonu Aç"
            
            if current_text != new_text:
                blocked = self.toggle_mic_button.blockSignals(True)
                self.toggle_mic_button.setText(new_text)
                self.toggle_mic_button.blockSignals(blocked)
                button_updated = True

        if hasattr(self, 'mic_indicator'):
            current_stylesheet = self.mic_indicator.styleSheet()
            new_stylesheet = ""
            if is_stt_active:
                new_stylesheet = "color: green; font-size: 18pt;"
            elif is_ww_listening:
                new_stylesheet = "color: blue; font-size: 18pt;"
            else:
                new_stylesheet = "color: gray; font-size: 18pt;"
            
            if current_stylesheet != new_stylesheet:
                self.mic_indicator.setStyleSheet(new_stylesheet)
                indicator_updated = True


    @QtCore.pyqtSlot(bool, str)
    def update_speech_status(self, active, message):
        self.log(f"DeskGUI: update_speech_status çağrıldı - active: {active}, message: {message}")
        self.speech_active = active
        # self.mic_active artık burada set edilmiyor, update_mic_button_and_indicator_based_on_state halledecek

        self.update_mic_button_and_indicator_based_on_state() # Merkezi fonksiyonu çağır

        if hasattr(self, 'recognition_indicator'):
            if active:
                self.recognition_indicator.setText("Aktif")
                self.recognition_indicator.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.recognition_indicator.setText("Pasif")
                self.recognition_indicator.setStyleSheet("color: gray;")
        
        if active:
            self.update_audio_status_signal.emit("listening", message if message else "Dinleniyor...")
        else:
            is_ww_listening = hasattr(self.audio_manager, 'is_wake_word_listening') and self.audio_manager.is_wake_word_listening
            if not is_ww_listening:
                self.update_audio_status_signal.emit("idle", message if message else "STT Kapalı")
            else:
                self.log("UI STT Durumu: STT kapalı, WW aktif (audio_status_signal gönderilmedi).")

    @QtCore.pyqtSlot(str, str)
    def update_status_indicator(self, text, status_type="normal"):
        """Update the status indicator with the provided text and type."""
        if self.status_label:
            self.status_label.setText(text)
            
            # Apply styling based on status type
            if status_type == "error":
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
            elif status_type == "warning":
                self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            elif status_type == "success":
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
            elif status_type == "processing":
                self.status_label.setStyleSheet("color: blue; font-weight: bold;")
            else:
                self.status_label.setStyleSheet("color: black; font-weight: normal;")

    @QtCore.pyqtSlot()
    def show_thinking_message(self):
        """Show 'Thinking...' message in the output area."""
        self.update_output_signal.emit("Thinking...", True)
        # Also update status indicator
        self.update_status_signal.emit("Processing Request", "processing")
        
        # Start timeout timer for the request
        self._start_request_timeout_timer()

    def _start_request_timeout_timer(self):
        """Start a timer to ensure the request doesn't get stuck."""
        # Cancel any existing timer
        if self.request_timer:
            self.request_timer.stop()
            
        # Create a new timer for this request
        self.request_timer = QtCore.QTimer()
        self.request_timer.setSingleShot(True)
        self.request_timer.timeout.connect(self._handle_request_timeout)
        self.request_timer.start(self.request_timeout_seconds * 1000)  # Convert to milliseconds
        
        self.log(f"Request timeout timer started ({self.request_timeout_seconds} seconds)")

    def _handle_request_timeout(self):
        """Handle the case where a request times out."""
        if self.is_processing_request:
            self.log("⚠️ Request timed out - resetting processing state")
            self.is_processing_request = False
            
            # Update UI
            self.update_status_signal.emit("Timed Out", "error")
            self.update_output_signal.emit("The request timed out. Please try again.", True)
            
            # Restart speech recognition
            if not self.using_bluetooth_audio:
                QtCore.QTimer.singleShot(1000, self.start_speech_recognition)
                
            # Cancel the timer
            if self.request_timer:
                self.request_timer.stop()
                self.request_timer = None

    @QtCore.pyqtSlot(str)
    def set_input_text(self, text):
        """Set the input text field value."""
        self.io_input.setText(text)
        
        # Also keep a copy in the last speech received buffer
        self.last_speech_text = text

    def eventFilter(self, obj, event):
        """
        Uygulama genelinde olayları filtreler.
        ComboBox'larda fare tekerinin seçenek değiştirmesini engeller.
        """
        if isinstance(obj, QtWidgets.QComboBox) and event.type() == QtCore.QEvent.Wheel:
            # Fare tekeri olayını engelle
            return True
            
        # Diğer tüm olayları normal işleme gönder
        return super(DeskGUI, self).eventFilter(obj, event)

    @QtCore.pyqtSlot()
    def clear_output_text(self):
        """Clear the output text field."""
        self.io_output.clear()

    @QtCore.pyqtSlot(str, str)
    def update_audio_status(self, status_type, message):
        """Update audio status indicators based on the status type."""
        try:
            # Handle microphone indicator
            if hasattr(self, 'recognition_indicator') and hasattr(self, 'mic_indicator'):
                if status_type == "listening":
                    self.recognition_indicator.setText("Aktif")  # "Active" -> "Aktif"
                    self.recognition_indicator.setStyleSheet("color: green; font-weight: bold;")
                    self.mic_indicator.setStyleSheet("color: green; font-size: 18pt;")
                elif status_type == "recognized":
                    # Flash the mic indicator green briefly to show recognition
                    self.recognition_indicator.setText(f"Tanındı: {message[:20]}...")  # "Recognized" -> "Tanındı"
                    self.recognition_indicator.setStyleSheet("color: blue; font-weight: bold;")
                    self.mic_indicator.setStyleSheet("color: blue; font-size: 18pt;")
                elif status_type == "idle":
                    self.recognition_indicator.setText("Pasif")  # "Inactive" -> "Pasif"
                    self.recognition_indicator.setStyleSheet("color: gray;")
                    self.mic_indicator.setStyleSheet("color: gray; font-size: 18pt;")
                elif status_type == "error":
                    self.recognition_indicator.setText(f"Hata: {message[:20]}...")  # "Error" -> "Hata"
                    self.recognition_indicator.setStyleSheet("color: red;")
                    self.mic_indicator.setStyleSheet("color: red; font-size: 18pt;")
            
            # Handle TTS indicator if available
            if hasattr(self, 'tts_indicator'):
                if status_type == "speaking":
                    self.speaking_active = True
                    self.tts_indicator.setStyleSheet("color: blue; font-size: 18pt;")
                else:
                    self.speaking_active = False
            
                # Update audio mode indicator if available
            if hasattr(self, 'audio_mode_indicator'):
                # Güvenli bir şekilde bluetooth durumunu kontrol et
                is_bluetooth = False
                if hasattr(self, 'using_bluetooth_audio'):
                    is_bluetooth = self.using_bluetooth_audio
                    
                if is_bluetooth:
                    self.audio_mode_indicator.setText("Bluetooth")
                    self.audio_mode_indicator.setStyleSheet("color: blue;")
                else:
                    self.audio_mode_indicator.setText("Doğrudan")  # "Direct" -> "Doğrudan"
                    self.audio_mode_indicator.setStyleSheet("color: green;")
            
            # Update audio level bar if available
            if hasattr(self, 'audio_level_bar') and hasattr(self, 'mic_level'):
                self.audio_level_bar.setValue(self.mic_level)
                
                # Update the color based on the level
                if self.mic_level > 80:
                    self.audio_level_bar.setStyleSheet("""
                        QProgressBar { 
                            border: 1px solid gray;
                            border-radius: 3px;
                            background-color: #f0f0f0;
                        }
                        QProgressBar::chunk {
                            background-color: #FF5252;  /* Red for high levels */
                        }
                    """)
                elif self.mic_level > 40:
                    self.audio_level_bar.setStyleSheet("""
                        QProgressBar { 
                            border: 1px solid gray;
                            border-radius: 3px;
                            background-color: #f0f0f0;
                        }
                        QProgressBar::chunk {
                            background-color: #FFC107;  /* Yellow for medium levels */
                        }
                    """)
                else:
                    self.audio_level_bar.setStyleSheet("""
                        QProgressBar { 
                            border: 1px solid gray;
                            border-radius: 3px;
                            background-color: #f0f0f0;
                        }
                        QProgressBar::chunk {
                            background-color: #4CAF50;  /* Green for low levels */
                        }
                    """)
        except Exception as e:
            # Hata durumunda sessizce devam et
            pass

    def update_audio_animations(self):
        """Update animations for audio indicators."""
        try:
            # Mikrofon seviyesi dalgalanmalarını canlandır
            if hasattr(self, 'mic_active') and self.mic_active:
                # "Nefes alma" efekti oluştur
                t = time.time() * 2  # Animasyon hızı için zaman faktörü
                breath = (math.sin(t) + 1) / 2  # 0 ila 1 arasında değer
                
                # Daha doğal görünmesi için rastgele gürültü ekle
                noise = random.uniform(-0.1, 0.1)
                level = breath + noise
                level = max(0, min(1, level))  # 0 ile 1 arasında sınırla
                
                # Ses seviyesi çubuğunu güncelle
                self.update_mic_level(level)
            else:
                # Etkin değilken seviyeyi yavaşça düşür
                if hasattr(self, 'mic_level') and self.mic_level > 0:
                    self.mic_level = max(0, self.mic_level - 5)
                    if hasattr(self, 'audio_level_bar'):
                        self.audio_level_bar.setValue(self.mic_level)
            
            # Konuşurken TTS göstergesini canlandır
            if hasattr(self, 'speaking_active') and self.speaking_active:
                # Renkleri değiştirerek titreşim efekti oluştur
                pulse_time = time.time() * 4  # TTS için daha hızlı titreşim
                if hasattr(self, 'tts_indicator'):
                    if int(pulse_time) % 2 == 0:
                        self.tts_indicator.setStyleSheet("color: blue; font-size: 18pt;")
                    else:
                        self.tts_indicator.setStyleSheet("color: #8080FF; font-size: 18pt;")
            else:
                # Konuşma yokken gri renge döndür
                if hasattr(self, 'tts_indicator'):
                    self.tts_indicator.setStyleSheet("color: gray; font-size: 18pt;")
        except Exception as e:
            # Hata durumunda sessizce devam et - GUI deneyimini bozma
            pass

    # Add these event handlers after the constructor to subscribe to TTS events
    def initialize_audio_events(self):
        """Set up subscribers for audio system events"""
        try:
            # TTS status events
            pub.subscribe(self.on_tts_speaking, 'tts:speaking')
            pub.subscribe(self.on_tts_complete, 'tts:complete')
            pub.subscribe(self.on_tts_error, 'tts:error')
            
            self.log("Audio event handlers registered successfully")
        except Exception as e:
            self.log(f"Error setting up audio event handlers: {e}")

    def on_tts_speaking(self, message):
        """Called when TTS starts speaking a message"""
        self.log(f"🔊 TTS speaking: {message[:50]}...")
        self.speaking_active = True
        self.update_audio_status_signal.emit("speaking", f"Speaking: {message[:20]}...")

    def on_tts_complete(self):
        """Called when TTS completes speaking"""
        self.log("🔊 TTS complete")
        self.speaking_active = False
        self.update_audio_status_signal.emit("idle", "Speech complete")

    def on_tts_error(self, error_msg):
        """Called when TTS encounters an error"""
        self.log(f"🔊 TTS error: {error_msg}")
        self.speaking_active = False
        self.update_audio_status_signal.emit("error", f"TTS error: {error_msg}")

    def populate_voice_combo(self):
        """Ses seçim combobox'ını mevcut tüm seslerle doldur"""
        self.voice_combo.clear()
        
        if not self.tts_engine or not self.tts_voices:
            self.voice_combo.addItem("TTS sesleri bulunamadı")
            return
        
        # Önce dil seçilmişse sadece o dili göster
        selected_lang = None
        if hasattr(self, 'tts_language_combo') and self.tts_language_combo.currentData():
            selected_lang = self.tts_language_combo.currentData()
            return self.populate_voice_combo_for_language(selected_lang)
        
        # Dil seçilmemişse tüm sesleri göster (dil gruplarıyla)
        for lang_code, voices in self.tts_voices.items():
            for voice in voices:
                # "Ses Adı (Dil)" formatında göster
                display_name = f"{voice['name']} ({lang_code})"
                self.voice_combo.addItem(display_name, voice['id'])
                
                # Mevcut sesi seç
                if voice['id'] == self.current_tts_voice:
                    self.voice_combo.setCurrentIndex(self.voice_combo.count() - 1)
    

    def on_tts_engine_changed(self, index):
        """TTS motoru değiştiğinde ayarları güncelle"""
        try:
            engine_name_display = self.tts_engine_combo.currentText()
            
            new_engine_type = None # Seçilen motor türünü saklamak için
            log_message = ""

            if "Piper" in engine_name_display:
                new_engine_type = "piper"
                log_message = "TTS motoru: Piper (Yerel)"
                self.tts_language_combo.setEnabled(True)
                self.voice_combo.setEnabled(True)
                self.speed_slider.setEnabled(True)
            elif "gTTS" in engine_name_display:
                new_engine_type = "gtts"
                log_message = "TTS motoru: Google TTS"
                self.tts_language_combo.setEnabled(True)
                self.voice_combo.setEnabled(False)
                self.speed_slider.setEnabled(True)
            elif "pyttsx3" in engine_name_display:
                new_engine_type = "pyttsx3"
                log_message = "TTS motoru: Sistem TTS (pyttsx3)"
                self.tts_language_combo.setEnabled(False)
                self.voice_combo.setEnabled(True)
                self.speed_slider.setEnabled(True)
            elif "espeak" in engine_name_display:
                new_engine_type = "espeak"
                log_message = "TTS motoru: eSpeak"
                self.tts_language_combo.setEnabled(True)
                self.voice_combo.setEnabled(False)
                self.speed_slider.setEnabled(True)
            elif "xTTS" in engine_name_display: # <<< GÜNCELLENDİ
                new_engine_type = "xtts"
                log_message = "TTS motoru: xTTS (Yerel)" # <<< DOĞRU LOG
                self.tts_language_combo.setEnabled(False) 
                self.voice_combo.setEnabled(False)
                self.speed_slider.setEnabled(False) 
            else:
                log_message = f"Bilinmeyen TTS motoru: {engine_name_display}"
                new_engine_type = None 
            
            self.tts_engine_type = new_engine_type
            if log_message: # Sadece log mesajı varsa yazdır
                self.log(log_message)

            if self.tts_engine_type:
                # `test_tts`'i çağırmadan önce `current_tts_lang` ve `current_tts_voice`'in
                # seçilen motor için uygun olduğundan emin olun.
                # XTTS için bunlar doğrudan kullanılmayabilir, _speak_with_xtts kendi ayarlarını kullanır.
                # Piper için dil ve ses seçimi geçerli olmalı.
                if self.tts_engine_type == "piper":
                    # Ensure current_tts_lang and current_tts_voice are valid for Piper
                    if self.tts_language_combo.count() > 0:
                        current_lang_data = self.tts_language_combo.currentData()
                        if current_lang_data:
                            self.current_tts_lang = current_lang_data
                            # Populate voices for this language and select the first one if voice_combo is used
                            self.populate_voice_combo_for_language(self.current_tts_lang)
                            if self.voice_combo.count() > 0:
                                self.current_tts_voice = self.voice_combo.itemData(0)
                            else:
                                self.current_tts_voice = None # No voice for this lang
                        else: # No language selected or invalid data
                            self.current_tts_lang = None
                            self.current_tts_voice = None
                    else: # No languages in combo
                        self.current_tts_lang = None
                        self.current_tts_voice = None

                # Sadece geçerli bir motor seçildiyse test et
                QtCore.QTimer.singleShot(500, self.test_tts)
                
        except Exception as e:
            self.log(f"TTS motoru değiştirilirken hata: {e}")
            self.log(traceback.format_exc())
    
    def on_voice_changed(self, index):
        """Ses değiştiğinde çağrılır"""
        if index < 0 or not self.tts_engine:
            return
        
        voice_id = self.voice_combo.itemData(index)
        if voice_id:
            # Sesi değiştir ve sonucu kaydet
            success = self.set_tts_voice(voice_id)
            self.current_tts_voice = voice_id
            
            voice_name = self.voice_combo.currentText()
            self.log(f"TTS sesi {voice_name} olarak değiştirildi")
            
            # Speed slider her zaman aktif olsun
            self.speed_slider.setEnabled(True)
    
    def test_tts(self):
        """Seçilen TTS motorunu test et"""
        try:
            engine_name = ""
            if self.tts_engine_type == "piper":
                engine_name = "Piper"
            elif self.tts_engine_type == "gtts":
                engine_name = "Google TTS"
            elif self.tts_engine_type == "pyttsx3":            
                engine_name = "Sistem TTS"
            elif self.tts_engine_type == "espeak":
                engine_name = "eSpeak"
            elif self.tts_engine_type == "xtts":
                engine_name = "xTTS"    
            test_text = f"Bu bir {engine_name} test mesajıdır."
            self.speak_text_locally(test_text)
        except Exception as e:
            self.log(f"TTS testi başlatılırken hata: {e}")
    
    def apply_audio_settings(self):
        """Apply audio settings from the UI"""
        # Get port from input field
        try:
            port = int(self.port_input.text().strip())
            if port != self.command_port:
                self.command_port = port
                self.log(f"Updated command port to {port}")
                
                # Reconnect if already connected
                if self.stream_connected:
                    self.log("Port changed, please reconnect to apply changes")
        except ValueError:
            self.log("Invalid port number")
        
        # Update speech recognition language
        lang_text = self.language_combo.currentText()
        lang_code = lang_text.split(" - ")[-1]
        self.audio_manager.change_language(lang_code)
        self.log(f"Updated speech recognition language to {lang_code}")
        
        # Restart speech recognition
        QtCore.QTimer.singleShot(1000, self.start_speech_recognition)
    
    def _process_llm_request(self, input_text):
        """Seçilen servisi kullanarak LLM isteğini işler ve yanıtı yönetir."""
        try:
            self.log(f"LLM isteği işleniyor: '{input_text[:100]}' (Servis: {self.current_llm_service})")

            output_text_final = ""
            llm_response_successful = False
            error_message_for_status = "" # Durum çubuğu için hata mesajı

            if self.current_llm_service == "ollama":
                # --- Ollama İşleme Mantığı ---
                # (Ollama logic remains unchanged as per the problem description focusing on Gemini)
                self.log("Ollama servisi kullanılıyor.")
                recog_lang_code = getattr(self.audio_manager, 'language', 'tr-TR')
                if not recog_lang_code: recog_lang_code = "tr-TR"
                src_lang_for_translate = recog_lang_code.split('-')[0] if recog_lang_code else "tr"
                llm_input_for_ollama = input_text

                if src_lang_for_translate != "en" and TRANSLATE_MODULE_AVAILABLE and TranslateHelper:
                    self.log(f"Ollama için giriş İngilizce'ye çevriliyor ({src_lang_for_translate} -> en)...")
                    try:
                        llm_input_for_ollama = TranslateHelper.translate(input_text, src_lang_for_translate, "en")
                        if not llm_input_for_ollama:
                            self.log("Çeviri boş döndü, orijinal metin kullanılıyor.")
                            llm_input_for_ollama = input_text
                        else:
                            self.log(f"Ollama için çevrilmiş giriş: {llm_input_for_ollama[:100]}...")
                    except Exception as e:
                        self.log(f"Ollama için giriş çevrilirken hata: {e}. Orijinal metin kullanılıyor.")
                        llm_input_for_ollama = input_text

                api_url = self.ollama_url.rstrip('/')
                if not api_url.endswith('/api/generate'):
                    if api_url.endswith('/api'):
                        api_url += '/generate'
                    else:
                        api_url += '/api/generate'
                if '\\' in api_url:
                    self.log(f"Uyarı: Ollama API URL'sinde ters slash tespit edildi, istek başarısız olabilir: {api_url}")

                self.log(f"Ollama API URL'si: {api_url}")
                selected_ollama_model = getattr(self, 'ollama_model', 'SentryBOT:4b')
                self.log(f"Kullanılan Ollama modeli: {selected_ollama_model}")
                payload = {'model': selected_ollama_model, 'prompt': llm_input_for_ollama, 'stream': False}
                self.log("Ollama API'sine istek gönderiliyor...")
                response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=self.request_timeout_seconds)

                if response.status_code == 200:
                    self.log("Ollama API'sinden başarılı yanıt alındı.")
                    try:
                        data = response.json()
                        raw_ollama_output = data.get('response', '')
                        if not raw_ollama_output:
                            self.log("Uyarı: Ollama API'sinden boş yanıt.")
                            raw_ollama_output = "Ollama'dan yanıt alamadım. Lütfen tekrar deneyin."

                        if src_lang_for_translate != "en" and TRANSLATE_MODULE_AVAILABLE and TranslateHelper:
                            self.log(f"Ollama yanıtı '{src_lang_for_translate}' diline geri çevriliyor...")
                            try:
                                output_text_final = TranslateHelper.translate(raw_ollama_output, "en", src_lang_for_translate)
                                if not output_text_final:
                                    self.log("Ollama yanıt çevirisi boş döndü, ham İngilizce yanıt kullanılıyor.")
                                    output_text_final = raw_ollama_output
                                else:
                                    self.log(f"Çevrilmiş Ollama yanıtı: {output_text_final[:100]}...")
                            except Exception as e:
                                self.log(f"Ollama yanıtı çevrilirken hata: {e}. Ham İngilizce yanıt kullanılıyor.")
                                output_text_final = raw_ollama_output
                        else:
                            output_text_final = raw_ollama_output
                        llm_response_successful = True
                    except ValueError as json_err:
                        self.log(f"Ollama JSON ayrıştırma hatası: {json_err}")
                        output_text_final = f"Ollama yanıtı ayrıştırılırken hata: {json_err}"
                        error_message_for_status = "Ollama Yanıt Hatası"
                else:
                    error_message_for_status = f"Ollama API Hatası ({response.status_code})"
                    error_message = f"Ollama API Hatası: HTTP {response.status_code}"
                    try:
                        error_details = response.json()
                        detail_msg = error_details.get("error", response.text[:200])
                        error_message += f" - {detail_msg}"
                    except:
                        error_message += f" - {response.text[:200]}"
                    self.log(error_message)
                    output_text_final = error_message

            elif self.current_llm_service == "gemini":
                # --- Gemini İşleme Mantığı ---
                self.log("Gemini servisi kullanılıyor.")
                gemini_response_obj = None # API'den dönen orijinal yanıtı tutmak için

                if not self.gemini_helper_instance:
                    error_message_for_status = "Gemini İstemci Hatası"
                    error_message = "Gemini istemcisi başlatılmadı. Lütfen API anahtarını ve ayarları yapılandırın."
                    self.log(error_message)
                    output_text_final = error_message
                    QMetaObject.invokeMethod(self, "_show_warning_message_slot",
                                            QtCore.Qt.QueuedConnection,
                                            Q_ARG(str, "Gemini Hatası"),
                                            Q_ARG(str, error_message))
                    QMetaObject.invokeMethod(self, "show_gemini_settings_menu", QtCore.Qt.QueuedConnection)
                else:
                    self.log(f"Gemini modeli '{self.gemini_helper_instance.model}' için prompt gönderiliyor: '{input_text[:100]}...'")
                    try:
                        gemini_response_obj = self.gemini_helper_instance.send_prompt(prompt=input_text) # API'den dönen objeyi al

                        if isinstance(gemini_response_obj, str):
                            output_text_final = gemini_response_obj
                            if output_text_final.strip(): # Dolu bir string ise başarılı say
                                self.log(f"GeminiHelper'dan dolu string yanıt alındı: {output_text_final[:200]}...")
                                llm_response_successful = True
                            else: # Boş string ise hata veya metin yok
                                self.log("GeminiHelper'dan boş string yanıt alındı.")
                                output_text_final = "Gemini'den boş metin yanıtı alındı." # Kullanıcıya gösterilecek mesaj
                                error_message_for_status = "Gemini Yanıtı Boş"

                        elif hasattr(gemini_response_obj, 'text') and gemini_response_obj.text: # Başarılı yanıt objesi, text içeriği dolu
                            output_text_final = gemini_response_obj.text
                            self.log(f"Gemini yanıt objesinden dolu metin alındı: {output_text_final[:100]}...")
                            llm_response_successful = True

                        elif hasattr(gemini_response_obj, 'prompt_feedback') and \
                            hasattr(gemini_response_obj.prompt_feedback, 'safety_ratings') and \
                            gemini_response_obj.prompt_feedback.safety_ratings: # Güvenlik filtresi
                            safety_ratings = gemini_response_obj.prompt_feedback.safety_ratings
                            safety_issues_list = []
                            for rating in safety_ratings:
                                category_name = str(rating.category).split('.')[-1]
                                probability_name = str(rating.probability).split('.')[-1]
                                if probability_name not in ["NEGLIGIBLE", "LOW"]:
                                    safety_issues_list.append(f"{category_name}: {probability_name}")

                            if safety_issues_list:
                                safety_issues = ", ".join(safety_issues_list)
                                error_message_for_status = "Gemini Filtrelendi"
                                error_message = f"Gemini yanıtı filtrelendi. Güvenlik sorunları: {safety_issues}"
                                self.log(error_message)
                                output_text_final = f"Üzgünüm, bu istekle ilgili güvenlik yönergelerime takıldım. ({safety_issues})"
                                QMetaObject.invokeMethod(self, "_show_warning_message_slot",
                                                        QtCore.Qt.QueuedConnection,
                                                        Q_ARG(str, "Gemini Filtrelendi"),
                                                        Q_ARG(str, f"Güvenlik sebebiyle yanıt filtrelendi:\n{safety_issues}"))
                            else: # Safety rating var ama sorunlu değil ve text yoksa
                                error_message_for_status = "Gemini Boş Yanıt"
                                error_message = "Gemini API'sinden .text özelliği olmayan bir yanıt alındı (safety rating sorunsuz)."
                                self.log(error_message)
                                output_text_final = "Gemini'den geçerli bir metin yanıtı alamadım (güvenlik sorunu yok)."

                        else: # Beklenmeyen yanıt tipi veya boş obje
                            error_message_for_status = "Gemini Beklenmedik Yanıt"
                            error_message = f"Gemini API'sinden boş veya beklenmedik yanıt objesi alındı: {type(gemini_response_obj)}"
                            self.log(error_message)
                            output_text_final = "Gemini'den beklenmedik bir yanıt aldım. Lütfen tekrar deneyin."
                            QMetaObject.invokeMethod(self, "_show_warning_message_slot",
                                                    QtCore.Qt.QueuedConnection,
                                                    Q_ARG(str, "Gemini API Hatası"),
                                                    Q_ARG(str, f"Beklenmeyen yanıt türü: {type(gemini_response_obj)}"))

                    except Exception as gemini_err:
                        self.log(f"Gemini API çağrısı veya yanıt işleme sırasında hata: {gemini_err}")
                        self.log(traceback.format_exc())
                        error_message_for_status = "Gemini API Hatası"
                        error_message = f"Gemini API hatası: {str(gemini_err)}"
                        output_text_final = error_message
                        QMetaObject.invokeMethod(self, "_show_warning_message_slot",
                                                QtCore.Qt.QueuedConnection,
                                                Q_ARG(str, "Gemini API Hatası"),
                                                Q_ARG(str, f"Gemini API çağrısı sırasında hata: {gemini_err}.\nLütfen ayarlarınızı ve API anahtarınızı kontrol edin."))
                        if "API_KEY" in str(gemini_err).upper() or "PERMISSION" in str(gemini_err).upper():
                            if hasattr(self, 'ollama_radio'):
                                QMetaObject.invokeMethod(self.ollama_radio, "setChecked",
                                                        QtCore.Qt.QueuedConnection,
                                                        QtCore.Q_ARG(bool, True))
            else:
                error_message_for_status = "Bilinmeyen LLM"
                error_message = f"Bilinmeyen LLM servisi seçili: {self.current_llm_service}"
                self.log(error_message)
                output_text_final = error_message
                QMetaObject.invokeMethod(self, "_show_critical_message_slot",
                                        QtCore.Qt.QueuedConnection,
                                        Q_ARG(str, "LLM Servis Hatası"),
                                        Q_ARG(str, f"Bilinmeyen LLM servisi: {self.current_llm_service}"))

        except requests.exceptions.Timeout:
            error_message_for_status = "İstek Zaman Aşımı"
            timeout_msg = f"Hata: {self.current_llm_service.capitalize()} API isteği zaman aşımına uğradı."
            self.log(timeout_msg)
            output_text_final = timeout_msg
        except requests.exceptions.ConnectionError:
            error_message_for_status = "Bağlantı Hatası"
            conn_err_msg = f"Hata: {self.current_llm_service.capitalize()} API'sine bağlanılamadı. Servisin çalıştığından emin olun."
            self.log(conn_err_msg)
            output_text_final = conn_err_msg
        except requests.exceptions.RequestException as req_err:
            error_message_for_status = "İstek Hatası"
            req_err_msg = f"{self.current_llm_service.capitalize()} API isteği gönderilirken hata: {req_err}"
            self.log(req_err_msg)
            output_text_final = req_err_msg
        except Exception as e:
            error_message_for_status = "İşlem Hatası"
            gen_err_msg = f"{self.current_llm_service.capitalize()} isteği sırasında genel hata: {e}"
            self.log(gen_err_msg)
            self.log(traceback.format_exc())
            output_text_final = gen_err_msg
            QMetaObject.invokeMethod(self, "update_audio_status_signal", Qt.QueuedConnection,
                                    Q_ARG(str, "error"), Q_ARG(str, f"Hata: {str(e)}"))
        finally:
            if self.request_timer:
                self.request_timer.stop()
                self.request_timer = None
                self.log("Request timeout timer durduruldu.")

            QMetaObject.invokeMethod(self, "update_output_signal",
                                    Qt.QueuedConnection,
                                    Q_ARG(str, output_text_final),
                                    Q_ARG(bool, True)) # clear_first=True

            status_type_final = "success" if llm_response_successful else "error"
            status_text_final = "İşlem Tamamlandı" if llm_response_successful else (error_message_for_status if error_message_for_status else f"{self.current_llm_service.capitalize()} Hatası")

            QMetaObject.invokeMethod(self, "update_status_signal",
                                    Qt.QueuedConnection,
                                    Q_ARG(str, status_text_final),
                                    Q_ARG(str, status_type_final))

            if llm_response_successful and output_text_final.strip(): # Metin doluysa ve başarılıysa
                try:
                    pub.sendMessage('llm_response', response_text=output_text_final)
                    self.log("LLM yanıtı için pubsub mesajı gönderildi, TTS abone tarafından yönetilecek.")
                except Exception as e:
                    self.log(f"llm_response olayı gönderilirken hata: {e}")
                QMetaObject.invokeMethod(self, "update_audio_status_signal",
                                        Qt.QueuedConnection,
                                        Q_ARG(str, "idle"),
                                        Q_ARG(str, "Yanıt hazır"))
                self.llm_response_pending_tts_completion = True # TTS'in bitmesini bekleyeceğiz
            else:
                self.log(f"LLM yanıtı başarısız veya boş ('{output_text_final[:30]}...'), TTS atlanıyor.")
                self.llm_response_pending_tts_completion = False
                # LLM başarısızsa veya yanıt boşsa, doğrudan STT'yi yeniden başlatma adımına geç
                QMetaObject.invokeMethod(self, "restart_stt_if_needed_after_llm", Qt.QueuedConnection)

            self.log(f"LLM ({self.current_llm_service}) istek işleme (worker thread) tamamlandı.")
            
    def test_tts_connection(self):
        """Test the TTS connection with the robot"""
        try:
            # Speak locally to confirm TTS works
            self.speak_text_locally("Testing TTS system.")
            
            # If connected to robot, send a command to make the robot 
            # forward a TTS request back to us
            if self.stream_connected:
                self.log("Sending TTS test command to robot")
                self.command_sender.send_command('tts_test', {
                    'text': 'This is a test of the robot to laptop TTS system.'
                })
        except Exception as e:
         self.log(f"TTS test error: {e}")