# audio_manager.py
import os
import time
import requests
import traceback
import threading
import numpy as np
from pubsub import pub
from .speech_input import SpeechInput # .tts import'u kaldırıldı, DeskGUI yönetecek
import sounddevice as sd
try:
    import onnxruntime as ort  # Wake word modeli için
    ONNXRUNTIME_AVAILABLE = True
except Exception as _ort_err:
    ort = None  # type: ignore
    ONNXRUNTIME_AVAILABLE = False
    _ORT_IMPORT_ERROR = _ort_err
    # Sessiz kalmak yerine log() henüz yok, bu yüzden print. DeskGUI açılınca log kanalına bilgi gönderilebilir.
    print(f"UYARI: onnxruntime import edilemedi: {_ort_err}\n  -> Wake word geçici olarak devre dışı. Çözüm: 'pip install \"numpy<2\" --force-reinstall' veya 'pip install --upgrade onnxruntime'.")

class AudioManager:
    def __init__(self, bluetooth_server="192.168.1.100"):
        # self.tts = None # TTS artık AudioManager'da değil
        self.speech_input = None
        self.bluetooth_server = bluetooth_server
        self.audio_mode = "direct" # Varsayılan
        self.language = "tr-TR"
        self.speech_active = False # STT'nin aktif olup olmadığını gösterir

        # Wake Word
        self.wake_word_detector_thread = None
        self.is_wake_word_listening = False
        self.wake_word_model_path = "hey_sen_tree_bot.onnx" # Modelinizin yolu
        self.WAKE_WORD_DETECTED_TOPIC = "wake_word_detected_event" # Pubsub için

        if not ONNXRUNTIME_AVAILABLE:
            self.log("AudioManager: onnxruntime mevcut değil, wake word özelliği devre dışı (NumPy / binary uyumsuzluğu olabilir).")

        # SpeechInput'u __init__ içinde oluştur
        try:
            self.speech_input = SpeechInput(
                protocol=self.audio_mode,
                bluetooth_server=self.bluetooth_server,
                language=self.language
            )
            self.log(f"AudioManager: SpeechInput ({self.language}) başlatıldı.")
        except Exception as e:
            self.log(f"HATA: AudioManager: SpeechInput başlatılamadı: {e}")
            self.speech_input = None

    def log(self, message):
        pub.sendMessage('log', msg=message)

    def initialize(self, audio_mode="direct", bluetooth_server=None, language="tr-TR"):
        # Bu metod artık sadece dil ve mod ayarı için kullanılabilir.
        # TTS ve SpeechInput __init__ içinde veya gerektiğinde oluşturuluyor.
        if bluetooth_server:
            self.bluetooth_server = bluetooth_server
        self.audio_mode = audio_mode
        
        if self.language != language:
            self.change_language(language) # Dil değişirse SpeechInput'u yeniden başlat
        
        self.log(f"AudioManager: Ayarlar güncellendi. Mod: {self.audio_mode}, Dil: {self.language}")

    def start_speech_recognition(self):
        """STT'yi başlatır. Wake word dinliyorsa onu durdurur."""
        if self.is_wake_word_listening:
            self.log("AudioManager: STT başlatılıyor, aktif wake word dinleyici durduruluyor.")
            self.stop_wake_word_listener() # Önce wake word'ü durdur

        if not self.speech_input:
            self.log("HATA: AudioManager: SpeechInput mevcut değil, STT başlatılamıyor.")
            return False, "STT Altyapısı Yok"

        if self.speech_active:
            self.log("AudioManager: STT zaten aktif.")
            return True, "STT Zaten Aktif"

        self.log("AudioManager: SpeechInput üzerinden STT başlatılıyor...")
        started, msg = self.speech_input.start() # SpeechInput.start() bool, str dönecek şekilde güncellenmeli
        if started:
            self.speech_active = True
            self.log(f"AudioManager: STT başarıyla başlatıldı: {msg}")
            return True, msg
        else:
            self.speech_active = False
            self.log(f"HATA: AudioManager: STT başlatılamadı: {msg}")
            return False, msg

    def stop_speech_recognition(self, restart_wake_word_if_enabled=True):
        """STT'yi durdurur. İstenirse wake word'ü yeniden başlatır."""
        if not self.speech_input:
            self.log("HATA: AudioManager: SpeechInput mevcut değil, STT durdurulamaz.")
            self.speech_active = False
            return False, "STT Altyapısı Yok"

        if not self.speech_active:
            self.log("AudioManager: STT zaten kapalı.")
            # Wake word'ü yine de kontrol edip başlatabiliriz (GUI isteğine bağlı)
            if restart_wake_word_if_enabled and hasattr(pub, 'sendMessage'): # DeskGUI'den gelen bilgiye göre WW başlatılabilir
                 pub.sendMessage("am_stt_stopped_check_ww") # DeskGUI'ye WW kontrolü için sinyal
            return True, "STT Zaten Kapalı"

        self.log("AudioManager: SpeechInput üzerinden STT durduruluyor...")
        stopped, msg = self.speech_input.stop_listening()
        if stopped:
            self.speech_active = False
            self.log(f"AudioManager: STT başarıyla durduruldu: {msg}")
            if restart_wake_word_if_enabled and hasattr(pub, 'sendMessage'):
                 pub.sendMessage("am_stt_stopped_check_ww")
            return True, msg
        else:
            # self.speech_active True kalmalı, durdurma başarısız
            self.log(f"HATA: AudioManager: STT durdurulamadı: {msg}")
            return False, msg

    def change_language(self, language):
        self.log(f"AudioManager: Dil şuna değiştiriliyor: {language}")
        was_active_stt = self.speech_active
        was_active_ww = self.is_wake_word_listening

        if was_active_stt:
            self.stop_speech_recognition(restart_wake_word_if_enabled=False) # WW'yi biz yöneteceğiz
        if was_active_ww:
            self.stop_wake_word_listener()

        self.language = language
        # SpeechInput'u yeni dille yeniden oluştur
        try:
            self.speech_input = SpeechInput(
                protocol=self.audio_mode,
                bluetooth_server=self.bluetooth_server,
                language=self.language
            )
            self.log(f"AudioManager: SpeechInput yeni dil '{self.language}' ile yeniden başlatıldı.")
        except Exception as e:
            self.log(f"HATA: AudioManager: Dil değiştirilirken SpeechInput yeniden başlatılamadı: {e}")
            self.speech_input = None

        if was_active_stt and self.speech_input:
            self.start_speech_recognition()
        elif was_active_ww: # STT kapalıydı ama WW açıktıysa, WW'yi tekrar başlat
            self.start_wake_word_listener(model_path=self.wake_word_model_path)


    def start_wake_word_listener(self, model_path="hey_sen_tree_bot.onnx", sample_rate=16000):
        if not ONNXRUNTIME_AVAILABLE or ort is None:
            self.log("AudioManager: Wake word başlatılamaz çünkü onnxruntime import edilemedi.")
            return
        if self.is_wake_word_listening:
            self.log("AudioManager: Wake word dinleyici zaten çalışıyor.")
            return
        if self.speech_active:
            self.log("AudioManager: STT aktifken wake word dinleyici başlatılamaz. Önce STT'yi durdurun.")
            return

        self.wake_word_model_path = model_path # Yolu sakla
        self.log(f"AudioManager: Wake word dinleyici başlatılıyor (Model: {model_path})...")
        self.is_wake_word_listening = True # Bayrağı hemen set et

        def listen_thread_func():
            try:
                ort_session = ort.InferenceSession(model_path)
                input_name = ort_session.get_inputs()[0].name
                input_shape = ort_session.get_inputs()[0].shape
                self.log(f"AudioManager WW: Model yüklendi. Beklenen girdi şekli: {input_shape}")

                positive_dims = [dim for dim in input_shape[1:] if isinstance(dim, int) and dim > 0]
                if not positive_dims:
                    chunk_samples = 1536 # Varsayılan
                    self.log(f"UYARI: AudioManager WW: Model şeklinden örnek sayısı çıkarılamadı, varsayılan: {chunk_samples}")
                else:
                    chunk_samples = np.prod(positive_dims)
                self.log(f"AudioManager WW: Gerekli ses örneği sayısı: {chunk_samples}")

                if chunk_samples <= 0:
                    self.log(f"HATA: AudioManager WW: Geçersiz örnek sayısı: {chunk_samples}.")
                    self.is_wake_word_listening = False # Bayrağı resetle
                    pub.sendMessage("wake_word_status_update", status_text=f"WW Hata: Örnek Sayısı")
                    return

                pub.sendMessage("wake_word_status_update", status_text=f"WW Dinliyor ({os.path.basename(model_path)})...")
                while self.is_wake_word_listening: # Bayrakla kontrol et
                    audio = sd.rec(int(chunk_samples), samplerate=sample_rate, channels=1, dtype='float32', blocking=True)
                    # sd.wait() # blocking=True ise gerek yok

                    if not self.is_wake_word_listening: break # Döngü içinde tekrar kontrol et

                    audio_flat = audio.flatten()
                    target_shape_ww = list(input_shape)
                    if target_shape_ww[0] is None or not isinstance(target_shape_ww[0], int) or target_shape_ww[0] < 1:
                        target_shape_ww[0] = 1
                    try:
                        input_data = audio_flat.astype(np.float32).reshape(target_shape_ww)
                    except ValueError as reshape_error:
                        self.log(f"UYARI: AudioManager WW: Yeniden şekillendirme hatası: {reshape_error}. Bu chunk atlanıyor.")
                        continue

                    ort_inputs = {input_name: input_data}
                    output = ort_session.run(None, ort_inputs)[0]

                    # Modelinizin çıktısına göre bu koşulu ayarlayın
                    # Örneğin, çıktı (1,1) ve eşik 0.6 ise:
                    if output.ndim == 2 and output.shape[0] == 1 and output.shape[1] > 0 : # çıktı (1, N) ise
                        # Eğer modeliniz tek bir skor döndürüyorsa (örn: wake word olasılığı)
                        # veya birden fazla skor döndürüyorsa (örn: [background_prob, wakeword_prob])
                        # doğru index'i ve eşiği kullanın.
                        # Bu örnekte, tek bir skor ve 0.6 eşiği varsayılıyor.
                        # Eğer modeliniz 2 sınıf için olasılık döndürüyorsa (örn: [no_ww, ww_detected])
                        # ve ww_detected index 1'deyse:
                        # ww_score = output[0][1]
                        ww_score = output[0][0] # Modelinize göre düzeltin
                        if ww_score > 0.7: # Eşiği ayarlayın
                            self.log(f"AudioManager: ===> WAKE WORD ALGILANDI! Skor: {ww_score:.4f} <===")
                            pub.sendMessage("wake_word_status_update", status_text=f"WW Algılandı! ({ww_score:.2f})")
                            # Wake word dinleyiciyi burada durdur, STT başlayacak
                            self.is_wake_word_listening = False # Önce bayrağı set et ki thread çıksın
                            pub.sendMessage(self.WAKE_WORD_DETECTED_TOPIC) # DeskGUI'ye haber ver
                            break # Dinleme thread'inden çık
                        # else:
                        #     print(f"WW Score: {ww_score:.3f}") # Düşük skorları logla (debug)
                self.log("AudioManager: Wake word dinleme thread'i sonlanıyor (iç döngüden çıkıldı).")

            except FileNotFoundError:
                self.log(f"HATA: AudioManager WW: Model dosyası '{model_path}' bulunamadı.")
                pub.sendMessage("wake_word_status_update", status_text=f"WW Hata: Model Yok")
            except Exception as load_error:  # onnxruntime spesifik hata türü yerine genel
                self.log(f"HATA: AudioManager WW: ONNX modeli '{model_path}' yüklenirken hata: {load_error}")
                if 'numpy' in str(load_error).lower():
                    self.log("İPUCU: NumPy sürümünü 'pip show numpy' ile kontrol edin. Gerekirse 'pip install \"numpy<2\" --force-reinstall'.")
                pub.sendMessage("wake_word_status_update", status_text=f"WW Hata: Model Yüklenemedi")
            except Exception as e:
                self.log(f"HATA: AudioManager WW: Dinleyici thread'inde kritik hata: {e}\n{traceback.format_exc()}")
                pub.sendMessage("wake_word_status_update", status_text=f"WW Hata: {e}")
            finally:
                self.is_wake_word_listening = False # Her durumda bayrağı resetle
                pub.sendMessage("wake_word_status_update", status_text="WW Dinleyici Sonlandı") # <<< DEĞİŞTİRİLDİ
                self.log("AudioManager: Wake word dinleyici thread'i tamamen sonlandı.")

        self.wake_word_detector_thread = threading.Thread(target=listen_thread_func, daemon=True)
        self.wake_word_detector_thread.start()
        self.log("AudioManager: Wake word dinleyici thread'i başlatıldı.")

    def stop_wake_word_listener(self):
        if not self.is_wake_word_listening:
            self.log("AudioManager: Wake word dinleyici zaten kapalı.")
            return

        self.log("AudioManager: Wake word dinleyici durduruluyor...")
        self.is_wake_word_listening = False # Thread'in çıkmasını sağlar
        if self.wake_word_detector_thread and self.wake_word_detector_thread.is_alive():
            self.wake_word_detector_thread.join(timeout=1.0) # Thread'in bitmesini bekle (kısa timeout)
        self.wake_word_detector_thread = None
        self.log("AudioManager: Wake word dinleyici durduruldu.")
        pub.sendMessage("wake_word_status_update", status_text="WW Durduruldu (manuel)")