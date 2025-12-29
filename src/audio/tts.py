import os
import time
import tempfile
import subprocess
import traceback
import io
import pygame
import gtts
import pyttsx3
import requests
from pubsub import pub
from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
import threading
import json
import shutil
from pydub import AudioSegment
from pydub.playback import play

class TTS:
    """Metin-Konuşma (Text-to-Speech) dönüşümü için sınıf"""
    
    def __init__(self, service='piper', protocol='direct', bluetooth_server=None, voice_index=0, auto_discover=True):
        """TTS sınıfını belirtilen servisle başlat"""
        self.service = service
        self.protocol = protocol
        self.bluetooth_server = bluetooth_server
        self.voice_index = voice_index
        self.auto_discover = auto_discover
        
        # TTS durumları
        self.is_speaking = False
        self.tts_speed = 1.0
        self.current_tts_lang = 'tr-TR'
        self.current_tts_voice = None
        self.voices = {}
        
        # PyDub kontrolü
        try:
            from pydub import AudioSegment
            self.pydub_available = True
            self.pydub_audioseq = AudioSegment
        except ImportError:
            self.pydub_available = False
            pub.sendMessage('log', msg="PyDub bulunamadı, ses hızlandırma devre dışı. pip install pydub ile kurabilirsiniz")
            
        # Pygame mixer başlatma
        try:
            pygame.mixer.init()
        except Exception as e:
            pub.sendMessage('log:error', msg=f"Pygame mixer başlatılamadı: {e}")
        
        # Temp klasör referansı
        self.tempfile_module = tempfile
        self.temp_dir = tempfile.gettempdir()
        
        # Piper için özel ayarlar
        if service == 'piper':
            self.piper_executable = self.find_piper_executable()
            self.tts_voices = self.find_piper_voices()
            
            if self.tts_voices:
                # İlk sesi varsayılan olarak ayarla
                if 'tr-TR' in self.tts_voices and self.tts_voices['tr-TR']:
                    self.current_tts_voice = self.tts_voices['tr-TR'][0]['id']
                    self.current_tts_lang = 'tr-TR'
                elif len(self.tts_voices) > 0:
                    # İlk mevcut dili kullan
                    first_lang = list(self.tts_voices.keys())[0]
                    self.current_tts_voice = self.tts_voices[first_lang][0]['id']
                    self.current_tts_lang = first_lang
                    
                pub.sendMessage('log', msg=f"Piper TTS başlatıldı: {len(self.tts_voices)} dil kullanılabilir")
            else:
                pub.sendMessage('log:warning', msg="Piper ses dosyaları bulunamadı!")
                
        elif service == 'pyttsx3':
            # pyttsx3 başlatma
            try:
                self.engine = pyttsx3.init()
                voices = self.engine.getProperty('voices')
                
                # Sesleri düzenli bir şekilde sakla
                self.voices = {}
                for i, voice in enumerate(voices):
                    lang = voice.languages[0] if voice.languages else 'tr-TR'
                    if not lang in self.voices:
                        self.voices[lang] = []
                    
                    self.voices[lang].append({
                        'id': i, 
                        'name': voice.name, 
                        'gender': 'female' if 'female' in voice.name.lower() else 'male',
                        'raw': voice
                    })
                
                # Varsayılan sesi ayarla
                if voice_index < len(voices):
                    self.engine.setProperty('voice', voices[voice_index].id)
                    self.current_tts_voice = voice_index
                
                pub.sendMessage('log', msg=f"pyttsx3 TTS başlatıldı: {len(voices)} ses bulundu")
            except Exception as e:
                pub.sendMessage('log:error', msg=f"pyttsx3 başlatılamadı: {e}")
    
        pub.sendMessage('log', msg=f"TTS servisi başlatıldı: {self.service}")


    def speak(self, text):
        """Metni seslendirir"""
        if not text or len(text) == 0:
            pub.sendMessage('log:warning', msg="TTS speak çağrıldı ama metin boş.")
            return False
            
        # Metni temizle
        clean_text = self.clean_text_for_tts(text)
        
        if len(clean_text) == 0:
            pub.sendMessage('log:warning', msg="TTS speak çağrıldı ama temizlenmiş metin boş.")
            return False
            
        # Kullanılan TTS servisine göre işlemi yönlendir
        if self.service == 'piper':
            return self._speak_with_piper(clean_text)
        elif self.service == 'gtts':
            return self._speak_with_gtts(clean_text)
        elif self.service == 'espeak':
            return self._speak_with_espeak(clean_text)
        elif self.service == 'pyttsx3':
            return self._speak_with_pyttsx3(clean_text)
        elif self.service == 'xtts':
            return self._speak_with_xtts(clean_text)
        else:
            pub.sendMessage('log:error', msg=f"Bilinmeyen TTS servisi: {self.service}")
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
                if not self.current_tts_voice or not os.path.exists(self.current_tts_voice):
                    raise FileNotFoundError(f"Piper model dosyası bulunamadı: {self.current_tts_voice}")
                    
                # Ses dosyası için geçici dosya oluştur
                temp_dir = self.temp_dir
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
                
                # Ses dosyasını çal
                self._play_audio_file(temp_file)
                
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
            pub.sendMessage('tts:complete')
            self.is_speaking = False
        
        def on_error(error_msg):
            pub.sendMessage('tts:error', error_msg=error_msg)
            self.is_speaking = False
        
        # UI'ı güncelle - konuşma başladı
        self.is_speaking = True
        pub.sendMessage('tts:speaking', message=text)
        
        # İşlemi ayrı bir thread'de başlat
        threading.Thread(
            target=_piper_worker_thread, 
            args=(text, on_done, on_error), 
            daemon=True
        ).start()
        
        # Hemen True dön, işlem arka planda devam edecek
        return True
        
    def _speak_with_gtts(self, text):
        """Google TTS (gTTS) kullanarak konuşma - thread güvenli versiyon"""
        # Arka planda çalışacak fonksiyon
        def _gtts_worker_thread(text, callback_done, callback_error):
            try:
                # Ses dosyası için geçici dosya oluştur
                temp_dir = self.temp_dir
                temp_file = os.path.join(temp_dir, f"gtts_temp_{int(time.time())}.mp3")
                
                # Dil kodunu belirle
                lang_code = self.current_tts_lang if hasattr(self, 'current_tts_lang') else 'tr'
                if '-' in lang_code:
                    lang_code = lang_code.split('-')[0]  # "tr-TR" -> "tr"
                
                # gTTS ile ses oluştur - bu uzun sürebilir
                tts = gtts.gTTS(text, lang=lang_code)
                tts.save(temp_file)
                
                # Ses dosyasını çal
                self._play_audio_file(temp_file)
                
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
            pub.sendMessage('tts:complete')
            self.is_speaking = False
        
        def on_error(error_msg):
            pub.sendMessage('tts:error', error_msg=error_msg)
            self.is_speaking = False
                
        # UI güncellemesi
        self.is_speaking = True
        pub.sendMessage('tts:speaking', message=text)
        
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
                speed_value = int(175 * self.tts_speed)
                cmd.extend(["-s", str(speed_value)])
                    
                # Çıktıyı bir ses dosyasına yönlendir
                temp_dir = self.temp_dir
                temp_file = os.path.join(temp_dir, f"espeak_temp_{int(time.time())}.wav")
                cmd.extend(["-w", temp_file])
                    
                # Metni ekle
                cmd.append(text)
                
                # Süreci başlat ve tamamlanmasını bekle
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                _, stderr = process.communicate()
                
                if process.returncode != 0:
                    raise Exception(f"eSpeak hata kodu: {process.returncode}, hata: {stderr.decode('utf-8')}")
                    
                # Ses dosyasını çal
                self._play_audio_file(temp_file)
                    
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
            pub.sendMessage('tts:complete')
            self.is_speaking = False
        
        def on_error(error_msg):
            pub.sendMessage('tts:error', error_msg=error_msg)
            self.is_speaking = False
        
        # UI güncelleme
        self.is_speaking = True
        pub.sendMessage('tts:speaking', message=text)
        
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
                temp_dir = self.temp_dir
                temp_file = os.path.join(temp_dir, f"pyttsx3_temp_{int(time.time())}.wav")
                
                # pyttsx3 motorunu başlat
                engine = pyttsx3.init()
                
                # Hız ayarını uygula
                engine.setProperty('rate', int(engine.getProperty('rate') * self.tts_speed))
                
                # Ses dosyasına kaydet
                engine.save_to_file(text, temp_file)
                engine.runAndWait()
                
                # Dosya kontrolü
                if not os.path.exists(temp_file) or os.path.getsize(temp_file) < 100:
                    raise Exception("Ses dosyası oluşturulamadı")
                    
                # Ses dosyasını çal
                self._play_audio_file(temp_file)
                    
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
            pub.sendMessage('tts:complete')
            self.is_speaking = False
        
        def on_error(error_msg):
            pub.sendMessage('tts:error', error_msg=error_msg)
            self.is_speaking = False
        
        # UI güncelleme
        self.is_speaking = True
        pub.sendMessage('tts:speaking', message=text)
        
        # Ayrı thread'de başlat
        threading.Thread(
            target=_pyttsx3_worker_thread,
            args=(text, on_done, on_error),
            daemon=True
        ).start()
        
        return True
    
    def set_xtts_config(self, api_url, speaker_wav):
        """XTTS API konfigürasyonunu ayarlar."""
        self.xtts_api_url = api_url
        self.xtts_speaker_wav = speaker_wav
        pub.sendMessage('log', msg=f"TTS Sınıfı XTTS yapılandırması: URL='{api_url}', Speaker='{speaker_wav}'")

    def _speak_with_xtts(self, text):
        """XTTS API sunucusu ile konuşma sentezler ve oynatır (TTS sınıfı versiyonu)."""
        api_url = getattr(self, 'xtts_api_url', None)
        speaker_wav_for_api = getattr(self, 'xtts_speaker_wav', None)

        if not api_url or not speaker_wav_for_api:
            err_msg = "XTTS API URL veya referans ses dosyası TTS sınıfında ayarlanmamış."
            pub.sendMessage('tts:error', error_msg=err_msg)
            self.is_speaking = False
            return False

        if not os.path.exists(speaker_wav_for_api):
            err_msg = f"XTTS referans ses dosyası bulunamadı: {speaker_wav_for_api}"
            pub.sendMessage('tts:error', error_msg=err_msg)
            self.is_speaking = False
            return False

        # Dil kodu: TTS sınıfı kendi self.current_tts_lang'ını kullanabilir
        # veya DeskGUI'deki gibi 2 harfli koda dönüştürme mantığı eklenebilir.
        # Şimdilik basitçe ilk iki harfi alalım.
        language_to_send = "tr" # Varsayılan
        if hasattr(self, 'current_tts_lang') and self.current_tts_lang:
            lang_parts = self.current_tts_lang.split('-')
            if lang_parts[0]:
                language_to_send = lang_parts[0].lower()
        # Burada DeskGUI'deki gibi desteklenen diller listesiyle kontrol daha iyi olurdu.

        def _xtts_worker_thread_tts_class(text_to_speak, api_url_worker, speaker_wav_path_worker, language_worker, callback_done, callback_error):
            # Bu kısım DeskGUI._speak_with_xtts._xtts_worker_thread ile 거의 동일 olmalı
            # Sadece self.log yerine pub.sendMessage('log', msg=...) kullanılmalı
            # ve QMetaObject.invokeMethod yerine doğrudan callback_done/error çağrılmalı.
            pub.sendMessage('log', msg=f"TTS Sınıfı XTTS Worker: Başladı. URL: {api_url_worker}")
            try:
                if not text_to_speak.strip():
                    callback_error("XTTS için seslendirilecek metin yok.")
                    return

                payload = {
                    "text": text_to_speak.strip(),
                    "speaker_wav": speaker_wav_path_worker,
                    "language": language_worker
                }
                pub.sendMessage('log', msg=f"TTS Sınıfı XTTS Worker: İstek: {payload}")

                headers = {'Content-Type': 'application/json'}
                response = requests.post(api_url_worker, json=payload, headers=headers, timeout=300)
                pub.sendMessage('log', msg=f"TTS Sınıfı XTTS Worker: Yanıt Kodu: {response.status_code}")

                if response.status_code == 200 and 'audio/wav' in response.headers.get('Content-Type','').lower():
                    audio_data = response.content
                    if len(audio_data) < 100:
                        callback_error("XTTS API'den geçersiz (çok küçük) ses verisi.")
                        return

                    if not pygame.mixer.get_init():
                        pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024)
                    if pygame.mixer.music.get_busy(): pygame.mixer.music.stop(); pygame.mixer.music.unload(); time.sleep(0.2)

                    audio_stream = io.BytesIO(audio_data)
                    pygame.mixer.music.load(audio_stream)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy(): time.sleep(0.01)
                    pygame.mixer.music.unload()
                    callback_done()
                else:
                    err_text = response.text[:500]
                    callback_error(f"XTTS API Hatası ({response.status_code}): {err_text}")

            except Exception as e:
                pub.sendMessage('log', msg=f"TTS Sınıfı XTTS Worker Hata: {e}\n{traceback.format_exc()}")
                callback_error(f"XTTS worker hatası: {str(e)}")

        def on_done_xtts_tts_class():
            pub.sendMessage('tts:complete')
            self.is_speaking = False

        def on_error_xtts_tts_class(error_msg):
            pub.sendMessage('tts:error', error_msg=error_msg)
            self.is_speaking = False

        self.is_speaking = True
        pub.sendMessage('tts:speaking', message=text)

        threading.Thread(
            target=_xtts_worker_thread_tts_class,
            args=(text, api_url, speaker_wav_for_api, language_to_send, on_done_xtts_tts_class, on_error_xtts_tts_class),
            daemon=True
        ).start()
        return True

    def _play_audio_file(self, audio_file):
        """Ses dosyasını çal"""
        try:
            # Pygame mixer başlatma - bir kere başlatılmışsa tekrar başlatma
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            
            # Ses dosyasını yükle ve çalmaya başla
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Çalma tamamlanana kadar bekle (bloklamadan)
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            return True
            
        except Exception as e:
            pub.sendMessage('log:error', msg=f"Ses dosyası çalınırken hata: {e}")
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
        
        # Minimum uzunluk kontrolü
        if len(filtrelenmis_metin) < 2:
            filtrelenmis_metin = "Mesaj anlaşılamadı."
            
        return filtrelenmis_metin
    def find_piper_executable(self):
        """İşletim sistemine göre uygun Piper çalıştırılabilir dosyasını bulur"""
        user_piper_dir = "C:\\Users\\emirh\\piper"
        user_piper_exe = os.path.join(user_piper_dir, "piper.exe")
        if os.path.exists(user_piper_exe):
            return user_piper_exe
        return None

    def find_piper_voices(self):
        """Piper için mevcut ses modellerini bulur"""
        voices = {}
        user_piper_dir = "C:\\Users\\emirh\\piper"
        if not os.path.exists(user_piper_dir):
            return voices
        for lang_dir in os.listdir(user_piper_dir):
            lang_path = os.path.join(user_piper_dir, lang_dir)
            if not os.path.isdir(lang_path) or lang_dir.startswith('.'):
                continue
            lang_code = lang_dir.lower()
            for root, dirs, files in os.walk(lang_path):
                for file in files:
                    if file.endswith(".onnx") or file.endswith(".onyx"):
                        full_path = os.path.join(root, file)
                        model_info = {'id': full_path}
                        file_parts = os.path.splitext(file)[0].split('-')
                        model_name = file_parts[0]
                        quality = file_parts[1] if len(file_parts) > 1 else "medium"
                        voice_name = f"{lang_code}_{model_name}_{quality}"
                        model_info['name'] = voice_name
                        model_info['gender'] = 'unknown'
                        iso_lang = lang_code
                        if len(lang_code) == 2:
                            iso_lang = f"{lang_code}-{lang_code.upper()}"
                        if iso_lang not in voices:
                            voices[iso_lang] = []
                        voices[iso_lang].append(model_info)
                        pub.sendMessage('log', msg=f"Ses modeli bulundu: {iso_lang} -> {voice_name}")
        return voices



    def set_speed(self, speed):
        """TTS konuşma hızını ayarla"""
        self.tts_speed = speed
        pub.sendMessage('log', msg=f"TTS hızı {speed:.1f}x olarak ayarlandı")
        return True

    def set_voice(self, voice_id):
        """Ses seçimini değiştir"""
        self.current_tts_voice = voice_id
        pub.sendMessage('log', msg=f"TTS sesi değiştirildi: {voice_id}")
        return True

    def set_language(self, lang_code):
        """TTS dilini değiştir"""
        self.current_tts_lang = lang_code
        pub.sendMessage('log', msg=f"TTS dili değiştirildi: {lang_code}")
        return True

    def get_available_languages(self):
        """Kullanılabilir dilleri döndür"""
        if self.service == 'piper':
            return list(self.tts_voices.keys())
        elif self.service == 'pyttsx3':
            return list(self.voices.keys())
        else:
            # Diğer motorlar için varsayılan diller
            return ['tr-TR', 'en-US', 'de-DE', 'fr-FR', 'es-ES']

    def get_voices_for_language(self, lang_code):
        """Belirli bir dil için kullanılabilir sesleri döndür"""
        if self.service == 'piper':
            return self.tts_voices.get(lang_code, [])
        elif self.service == 'pyttsx3':
            return self.voices.get(lang_code, [])
        else:
            # Diğer motorlar için boş liste döndür
            return []

    def select_random_voice_for_language(self, lang_code):
        """Belirli bir dil için rastgele bir ses seç"""
        voices = self.get_voices_for_language(lang_code)
        if not voices:
            return None
            
        import random
        voice = random.choice(voices)
        return voice['id']