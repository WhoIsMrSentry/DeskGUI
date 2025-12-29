# DeskGUI

SentryBOT robot platformu için geliştirilmiş, PyQt5 tabanlı masaüstü kontrol ve izleme arayüzüdür. Gerçek zamanlı video akışı, sesli komutlar, yüz ve nesne tanıma, robot durum takibi ve LLM (büyük dil modeli) entegrasyonları ile robotunuzu kolayca yönetmenizi sağlar.

## Özellikler

- **Gerçek Zamanlı Video Akışı:** Robotun kamerasından canlı görüntü izleyin.
- **Sesli Komut ve TTS:** Mikrofon ile komut verin, robotun yanıtlarını sesli dinleyin.
- **Yüz ve Nesne Tanıma:** Gelişmiş görüntü işleme modülleri ile yüz, nesne, yaş ve duygu tespiti.
- **Bluetooth Ses Sunucusu:** Robotun ses giriş/çıkışını bilgisayarınız üzerinden yönetin.
- **Robot Durum Takibi:** Bağlantı, göz rengi, kişilik gibi robot durumlarını anlık izleyin.
- **LLM ve Gemini Entegrasyonu:** Ollama ve Gemini gibi büyük dil modelleriyle sohbet ve komut desteği.
- **Tema Desteği:** Koyu, açık ve kırmızı temalar arasında geçiş yapabilme.
- **Gelişmiş Loglama ve Hata Yönetimi:** Tüm olaylar ve hatalar için detaylı log paneli.
- **Dil Desteği:** Çoklu dil desteği ve otomatik dil tanıma özelliği.
- **El ve Parmak Hareketleri Tanıma:** Kamera üzerinden el hareketleri ile robot kontrolü.
- **Animasyon Kontrolü:** Robot üzerindeki LED ve servo animasyonlarını yönetme.
- **Yaş ve Duygu Tespiti:** Yüz görüntüsünden yaklaşık yaş ve duygusal ifade tespiti.

## Kurulum

### Sistem Gereksinimleri

- **İşletim Sistemi:** Windows 10/11, Ubuntu 20.04+ veya macOS 10.15+
- **Python:** 3.8 veya üzeri (3.10 önerilen)
- **Mikrofon:** STT için çalışır durumda bir mikrofon
- **Hoparlörler:** TTS çıktısı için ses sistemi
- **Kamera:** (Opsiyonel) Test için yerel kamera
- **GPU:** (Opsiyonel) Görüntü işleme işlevleri için NVIDIA GPU önerilir

### Kurulum Adımları

1. Proje dosyalarını indirin ve gerekli Python paketlerini yükleyin:
   ```powershell
   # Python sanal ortam oluşturmak (tavsiye edilir)
   python -m venv venv
   .\venv\Scripts\activate

   # Gerekli paketleri yükle
   pip install -r requirements.txt
   
   # Görüntü işleme modülleri için ekstra paketler (opsiyonel)
   pip install mediapipe cvzone tensorflow
   ```

2. Görüntü İşleme modelleri için gereken dosyaları yerleştirin:
   - `encodings.pickle`: Yüz tanıma modeli dosyası (örnek dosya pakette mevcuttur)
   - `haarcascade_frontalface_default.xml`: Yüz tespiti için OpenCV modeli
   - `hey_sen_tree_bot.onnx`: Wake word algılama modeli
   - Ayrıca `modules/vision/__init__.py` dosyasında `MODELS_DIR` değişkenini güncelleyin:
     ```python
     MODELS_DIR = r"C:\path\to\your\models" # Kendi modellerinizin konumuna göre değiştirin
     ```

3. GUI'yi başlatmak için:
   ```powershell
   python run_gui.py --robot-ip <ROBOT_IP_ADRESI>
   ```
   veya hem GUI hem ses sunucusunu birlikte başlatmak için:
   ```powershell
   python run_all.py
   ```

## Komut Satırı Argümanları

### run_gui.py için Argümanlar

- `--robot-ip` - Robotun IP adresi (varsayılan: 192.168.137.52)
- `--video-port` - Video akış portu (varsayılan: 8000)
- `--command-port` - Komut portu (varsayılan: 8090)
- `--ollama-url` - Ollama API URL (varsayılan: http://localhost:11435)
- `--ollama-model` - Kullanılacak Ollama modeli (varsayılan: SentryBOT:4b)
- `--encodings-file` - Yüz tanıma modeli dosyası (varsayılan: encodings.pickle)
- `--bluetooth-server` - Bluetooth ses sunucusu IP adresi (varsayılan: 192.168.1.100)
- `--enable-fastapi` - FastAPI desteğini etkinleştir
- `--retry-on-error` - Hata durumunda otomatik yeniden başlatma
- `--log-file` - Log dosyası (varsayılan: sentry_gui.log)
- `--debug` - Hata ayıklama bilgilerini göster

### run_audio_server.py için Argümanlar

- `--host` - Sunucunun bağlanacağı host (varsayılan: 0.0.0.0)
- `--tts-port` - TTS hizmeti için port (varsayılan: 8095)
- `--speech-port` - Konuşma tanıma hizmeti için port (varsayılan: 8096)
- `--fastapi-port` - FastAPI WebSocket sunucusu için port (varsayılan: 8098)
- `--use-fastapi` - Performans için FastAPI kullan
- `--device-name` - Mikrofon cihaz adı
- `--device-index` - Mikrofon cihaz indeksi (cihaz adına alternatif)
- `--list-devices` - Mevcut mikrofon cihazlarını listele
- `--voice-idx` - TTS için ses indeksi (varsayılan: 0)
- `--auto-start-speech` - Başlangıçta konuşma tanımayı otomatik başlat
- `--language` - Konuşma tanıma dili (örn: en-US, tr-TR)
- `--test-audio` - Başlangıçta ses çıkışını test et
- `--verbose` - Detaylı loglama

### run_all.py için Argümanlar

- `--robot-ip` - Robotun IP adresi
- `--video-port` - Video akış portu
- `--command-port` - Komut portu
- `--ollama-url` - Ollama API URL (varsayılan port 11435)
- `--encodings-file` - Yüz tanıma modeli dosyası
- `--debug` - Hata ayıklama bilgilerini göster
- `--theme` - Uygulama teması (light, dark, auto seçenekleri)
- `--xtts` - XTTS API sunucusunu ayrı terminalde başlat (Windows)

## TTS (Text-to-Speech) Yapılandırması

### Piper TTS Kurulumu

1. [Piper TTS](https://github.com/rhasspy/piper)'ı indirin (Windows, Linux, MacOS):
   
   ```powershell
   # Windows için örnek kurulum
   mkdir C:\Users\<KULLANICI>\piper
   cd C:\Users\<KULLANICI>\piper
   
   # İndirme bağlantısı (Windows için)
   $url = "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_windows_amd64.zip"
   Invoke-WebRequest -Uri $url -OutFile "piper.zip"
   Expand-Archive -Path "piper.zip" -DestinationPath "."
   ```

2. İhtiyacınız olan dil modellerini indirin:

   ```powershell
   # Türkçe model için örnek
   mkdir C:\Users\<KULLANICI>\piper\tr-TR
   cd C:\Users\<KULLANICI>\piper\tr-TR
   
   # Türkçe model indirme
   $model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/sinem/medium/tr_TR-sinem-medium.onnx"
   $json_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/sinem/medium/tr_TR-sinem-medium.onnx.json"
   
   Invoke-WebRequest -Uri $model_url -OutFile "tr_TR-sinem-medium.onnx"
   Invoke-WebRequest -Uri $json_url -OutFile "tr_TR-sinem-medium.onnx.json"
   ```

3. Ses modellerini aşağıdaki dizin yapısına yerleştirin:
   - Windows: `C:\Users\<KULLANICI>\piper\<DIL_KODU>\<MODEL>.onnx`
   - Linux: `~/piper/<DIL_KODU>/<MODEL>.onnx`
   
4. Test edin (Opsiyonel):
   ```powershell
   cd C:\Users\<KULLANICI>\piper
   .\piper.exe --model .\tr-TR\tr_TR-sinem-medium.onnx --output_file test.wav --text "Merhaba, ben bir robot sesiyim."
   ```

5. GUI içinde TTS hizmetini "piper" olarak ayarlayın. DeskGUI otomatik olarak modellerinizi bulacaktır.

### XTTS (XTalker TTS) Kurulumu

1. XTTS için sanal ortam oluşturun:

   ```powershell
   # Sanal ortam için dizin oluştur
   mkdir C:\Users\<KULLANICI>\xTTS
   cd C:\Users\<KULLANICI>\xTTS
   
   # Python sanal ortam oluştur ve etkinleştir
   python -m venv tts_env
   .\tts_env\Scripts\Activate.ps1
   
   # Gerekli paketleri kur
   pip install TTS uvicorn fastapi python-multipart
   ```

2. XTTS API sunucusu için aşağıdaki içerikle `xtts.py` dosyası oluşturun:

   ```python
   from fastapi import FastAPI, File, UploadFile, Form
   from fastapi.responses import FileResponse
   from fastapi.middleware.cors import CORSMiddleware
   import os
   import tempfile
   import uvicorn
   from TTS.api import TTS
   
   app = FastAPI()
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   
   # Initialize TTS
   tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
   
   @app.post("/synthesize")
   async def synthesize_speech(
       text: str = Form(...),
       speaker_wav: UploadFile = File(...),
       language: str = Form("tr")
   ):
       print(f"Generating speech for: {text[:50]}... in {language}")
       
       # Save the uploaded speaker file
       temp_dir = tempfile.gettempdir()
       speaker_path = os.path.join(temp_dir, "speaker.wav")
       with open(speaker_path, "wb") as f:
           f.write(await speaker_wav.read())
       
       # Generate output path
       output_path = os.path.join(temp_dir, "output.wav")
       
       # Generate speech
       tts.tts_to_file(text=text, 
                       file_path=output_path,
                       speaker_wav=speaker_path, 
                       language=language)
       
       return FileResponse(output_path, media_type="audio/wav")
   
   if __name__ == "__main__":
       uvicorn.run(app, host="0.0.0.0", port=5002)
   ```

3. API sunucusunu başlatmak için `start_xtts_api.bat` dosyası (veya `run_all.py` ile `--xtts` parametresi):

   ```batch
   @echo off
   echo XTTS API Sunucusu baslatiliyor...
   
   REM Sanal ortami aktif et
   call "C:\Users\<KULLANICI>\xTTS\tts_env\Scripts\activate.bat"
   
   echo Sanal ortam (tts_env) aktif.
   
   REM Uvicorn'u calistir
   echo Uvicorn sunucusu baslatiliyor (0.0.0.0:5002)...
   C:\Users\<KULLANICI>\xTTS\tts_env\Scripts\python.exe -m uvicorn xtts:app --reload --host 0.0.0.0 --port 5002
   
   echo Sunucu durduruldu.
   pause
   ```

4. Ses örneği için bir WAV dosyası hazırlayın (16kHz, mono, WAV formatında olmalı)
5. GUI içinde TTS hizmetini "xtts" olarak ayarlayın
6. Referans ses dosyası yolunu ayarlarda belirtin

### Diğer TTS Seçenekleri

- **pyttsx3** - Yerel TTS motoru, ek kurulum gerektirmez
- **gtts** - Google'ın TTS hizmeti (internet bağlantısı gerektirir)
- **espeak** - Hafif TTS motoru (önceden kurulmalıdır)

## Kullanım

- Robotun IP adresini ve portlarını komut satırı argümanları ile belirtebilirsiniz.
- GUI üzerinden video, ses, animasyon ve komut kontrollerini kolayca gerçekleştirebilirsiniz.
- Gelişmiş ayarlar ve LLM/Gemini API anahtarlarını GUI içinden yapılandırabilirsiniz.

## Bağımlılıklar

- Python 3.8+
- PyQt5
- OpenCV

## Sorun Giderme

### Ollama 404 ( /api\\generate ) Hatası
Windows ortamında bazı sürümlerde `os.path.join` kullanımı URL parçalarını birleştirirken ters slash (\\) ekleyerek `http://localhost:11435/api\generate` gibi hatalı bir yol oluşturabilir ve Ollama sunucusu 404 döndürür.

Çözüm:
1. GUI içinde Ollama URL alanına `http://localhost:11435` veya `http://localhost:11435/api` yazın (sonunda ters slash yok).
2. Loglarda `api\generate` görüyorsanız güncel koddaki düzeltmenin çekildiğinden emin olun (URL birleştirme artık düz string ekleme ile yapılıyor).
3. Manuel test: PowerShell'de
   ```powershell
   Invoke-RestMethod -Method Post -Uri http://localhost:11435/api/generate -Body (@{model='SentryBOT:4b'; prompt='test'; stream=$false} | ConvertTo-Json) -ContentType 'application/json'
   ```
   Yanıt JSON dönmeli; dönmüyorsa Ollama servisinin çalıştığını doğrulayın: `ollama list`.

### NumPy / onnxruntime DLL Hataları
`ImportError: DLL load failed while importing _framework_bindings` veya `_ARRAY_API` hataları alıyorsanız büyük olasılıkla NumPy 2.x ile eski onnxruntime sürümü uyumsuzdur.

Çözüm:
1. `pip install 'numpy<2' --upgrade --force-reinstall`
2. Gerekirse `pip install --upgrade onnxruntime` (veya GPU için `onnxruntime-gpu`).
3. `python debug_imports.py` çalıştırarak ayrıntılı tanılama alın; script uyumsuzlukları raporlar.

### Boş / Çok Uzun LLM Yanıtları
Yanıt boşsa model ilk prompt'u yüklerken zaman aşımı olmuş olabilir. Timeout değerini ayarlardan yükseltin. Çok uzun yanıtlar otomatik kısaltılır; ihtiyacınıza göre limitleri `desk_gui_app.py` içinde arayın: `resp_length` ve `output` kesme noktaları.

### Çeviri Modülü Yok / Hata
Çeviri devre dışıysa giriş metni çevrilmeden Ollama'ya gönderilir. `modules/translate_helper.py` mevcut ve `googletrans` (veya kullanılan kütüphane) kurulu olduğundan emin olun.

- face_recognition
- numpy
- sounddevice, pyaudio, pyttsx3, gtts
- requests, pubsub, langdetect, pygame, onnxruntime
- (ve diğerleri, detay için requirements.txt)

### Önerilen Paketler (isteğe bağlı)

```powershell
pip install PyQt5 opencv-python-headless face_recognition numpy sounddevice pyaudio pyttsx3 gtts requests pubsub pygame onnxruntime pydub langdetect fastapi uvicorn
```

Gelişmiş yüz ve nesne tanıma özellikleri için:
```powershell
pip install mediapipe cvzone tensorflow keras
```

## Dosya Yapısı

- `desk_gui.py`, `run_gui.py`, `run_all.py`: Ana başlatıcı ve GUI dosyaları
- `modules/`: Ses, görüntü işleme, komut, robot veri dinleyici ve yardımcı modüller
- `modules/gui/desk_gui_app.py`: Tüm GUI ve işlevselliklerin merkezi
- `modules/vision/`: Görüntü işleme (yüz, nesne, parmak, yaş-duygu tespiti)
- `encodings.pickle`, `haarcascade_frontalface_default.xml`: Model ve yardımcı dosyalar

## Modüller ve Bileşenler

DeskGUI çok sayıda modülden oluşan modüler bir yapıya sahiptir. İşte ana modüllerin açıklamaları:

### Ana Modüller

- **desk_gui_app.py**: Ana GUI uygulaması, tüm arayüz ve kontrolleri içerir
- **audio_manager.py**: Ses giriş/çıkış işlemlerini ve cihazlarını yönetir
- **audio_thread_manager.py**: Ses işlemleri için çoklu thread yönetimi sağlar
- **command_sender.py**: Robota komut göndermek için TCP protokolü kullanır
- **command_helpers.py**: Komut oluşturma ve işleme yardımcı fonksiyonları
- **face_detector.py**: Yüz tespiti ve tanıma işlemlerini gerçekleştirir
- **gemini_helper.py**: Google Gemini AI API entegrasyonu sağlar
- **motion_detector.py**: Kamera görüntüsünde hareket algılama yapar
- **remote_video_stream.py**: Robottan gelen video akışını alır ve işler
- **robot_data_listener.py**: Robot durum mesajlarını dinler ve işler
- **speech_input.py**: Konuşma tanıma ve ses giriş işlemlerini yönetir
- **tracking.py**: Nesne ve yüz takibi için konum hesaplamaları yapar
- **translate_helper.py**: Çeşitli diller arasında çeviri işlemleri sağlar
- **tts.py**: Metin-Konuşma (TTS) sistemi, çeşitli TTS motorlarını destekler

### Görüntü İşleme Modülleri (vision/)

- **age_emotion.py**: Yüzden yaş ve duygu tespiti modülü
- **finger_tracking.py**: El ve parmak hareketi tanıma modülü
- **object_detection.py**: Tensorflow tabanlı nesne tanıma modülü
- **object_tracking.py**: Tespit edilen nesnelerin takibi için algoritma

### GUI Öğeleri (modules/gui/)

- **desk_gui_app.py**: DeskGUI uygulamasının ana sınıfı ve arayüzü

### Başlatıcı Dosyalar

- **run_gui.py**: Sadece GUI bileşenini başlatır
- **run_audio_server.py**: Sadece ses sunucusunu başlatır 
- **run_all.py**: Hem GUI hem ses sunucusunu birlikte başlatır

## LLM (Dil Modeli) Entegrasyonu

### Ollama

SentryBOT varsayılan olarak [Ollama](https://ollama.ai/) ile entegre çalışır:

1. Ollama'yı bilgisayarınıza kurun:
   ```powershell
   # Windows için tavsiye edilen kurulum
   winget install Ollama.Ollama
   ```

2. Ollama modelini indirin: 
   ```powershell
   ollama pull [MODEL_ADI]
   ```
   veya tercih ettiğiniz bir modeli kullanın (Llama3, Mistral, vb.)

3. `--ollama-url` ve `--ollama-model` argümanlarıyla yapılandırın:
   ```powershell
   python run_gui.py --ollama-url http://localhost:11435 --ollama-model [MODEL_ADI]
   ```

### Gemini AI

Google Gemini API'sini kullanmak için:

1. [Google AI Studio](https://ai.google.dev/)'dan API anahtarı alın
2. GUI içinden Gemini ayarları menüsüne erişin
3. API anahtarınızı ve diğer parametreleri ayarlayın (model, temperature, top-k, vb.)

### API Yanıt İşleme

LLM yanıtlarında özel komut işaretçileri kullanılabilir:
- `!command:name` - Doğrudan robot komutları tetikleme
- `!animate:name` - Animasyonları başlatma
- `!eye:color` - LED göz rengini değiştirme

## Gelişmiş Özellikler

### Yüz Tanıma

Yüz tanıma için kişilerin yüz kodlarını `encodings.pickle` dosyasında saklayın:

```python
import face_recognition
import pickle

# Yüz kodlarını oluşturun ve kaydedin
known_face_encodings = []  # face_recognition ile oluşturulan yüz kodları
known_face_names = []      # Kişi adları
data = {"encodings": known_face_encodings, "names": known_face_names}

with open('encodings.pickle', 'wb') as f:
    pickle.dump(data, f)
```

### Wake Word Algılama

GUI içinden "Wake Word" özelliğini etkinleştirerek ses tetikleme kelimesiyle komut verebilirsiniz. Varsayılan tetik ifadesi "Hey Sentrybot"tur.

### Robot Animasyon Kontrolü

Robot üzerindeki LED ve servo animasyonları şu parametrelerle kontrol edilebilir:

```python
# LED ışık animasyonları
animations = ["RAINBOW", "WAVE", "FIRE", "GRADIENT", "RANDOM_BLINK", "ALTERNATING", "STACKED_BARS"]

# Servo motor animasyonları
servo_animations = ["HEAD_NOD", "LOOK_UP", "WAVE_HAND", "CENTER"]
```

### Donanım İstek Spesifikasyonları

DeskGUI en iyi performans için aşağıdaki minimum gereksinimleri tavsiye eder:
- **İşlemci:** Intel Core i5 (7. nesil ve üstü) veya AMD Ryzen 5
- **RAM:** 8 GB (yüz tanıma ve görüntü işleme yoğun kullanım için 16 GB)
- **GPU:** Basit kullanım için entegre grafik kartı yeterli, görüntü işleme için NVIDIA GPU önerilir
- **İşletim Sistemi:** Windows 10/11, Ubuntu 20.04+ veya MacOS 10.15+
- **Bağlantı:** Ethernet veya güçlü WiFi bağlantısı (video akışı için)

## Robot İletişim Protokolü

DeskGUI, komut göndermek ve robot durumunu izlemek için SentryBOT'a TCP soketleri üzerinden JSON formatında mesajlar gönderir:

### Temel Komut Formatı

```json
{
  "command": "KOMUT_ADI",
  "params": {
    "param1": "değer1",
    "param2": "değer2"
  }
}
```

### Sık Kullanılan Komutlar

- **animate**: Robot üzerinde animasyon başlatır
  ```json
  {"command": "animate", "params": {"animation": "RAINBOW", "repeat": 1}}
  ```

- **servo**: Servo motorları kontrol eder
  ```json
  {"command": "servo", "params": {"id": 0, "position": 90}}
  ```

- **speech**: Robotun konuşmasını tetikler
  ```json
  {"command": "speech", "params": {"text": "Merhaba dünya"}}
  ```

- **eye_color**: Robot göz rengini değiştirir
  ```json
  {"command": "eye_color", "params": {"r": 255, "g": 0, "b": 0}}
  ```

## Config Dosyaları

DeskGUI aşağıdaki config dosyalarını kullanır:

1. **personalities.json**: Robot kişiliklerini ve LLM başlatma komutlarını tanımlar
   ```json
   {
     "KişilikAdı": {
       "description": "Kişilik açıklaması",
       "startup_prompt": "LLM için sistem komutu"
     }
   }
   ```

2. **priority_animations.json**: Belirli kişiler algılandığında çalışacak animasyonları tanımlar
   ```json
   {
     "KişiAdı": "ANİMASYON_ADI"
   }
   ```

## Sorun Giderme

### Bağlantı Sorunları

1. **Robot bağlanamıyor:**
   - Robot IP adresinin doğru olduğunu kontrol edin
   - Bilgisayarınız ile robot aynı ağda olmalıdır
   - Firewall ayarlarını kontrol edin, gerekli portların açık olduğundan emin olun

2. **Video akışı yok:**
   - `--video-port` parametresi robot tarafındaki port ile eşleşmelidir
   - OpenCV kütüphanesinin doğru kurulduğundan emin olun

3. **Ses sorunları:**
   - `--list-devices` parametresini kullanarak doğru mikrofon cihazını tespit edin
   - Bluetooth sunucusunun çalıştığını ve erişilebilir olduğunu kontrol edin

### TTS Sorunları

1. **Piper çalışmıyor:**
   - Varsayılan olarak PyTTSx3'e geçiş yapılacaktır
   - Piper modellerinizin doğru dizinde olduğunu kontrol edin

2. **XTTS API bağlantı hatası:**
   - API sunucusunun çalıştığını kontrol edin (port 5002)
   - Ses örneği WAV dosyasının doğru konumda olduğunu kontrol edin

### Görüntü İşleme Sorunları

1. **Yüz tanıma çalışmıyor:**
   - `encodings.pickle` dosyasının mevcut olduğunu kontrol edin
   - face_recognition kütüphanesinin doğru kurulduğundan emin olun
   - Dizinde `haarcascade_frontalface_default.xml` olduğunu doğrulayın

2. **Nesne tanıma hatası:**
   - `MODELS_DIR` dizin yolunu yapılandırma için `modules/vision/__init__.py` dosyasını kontrol edin
   - YOLO modelinin doğru konumda olduğunu doğrulayın

## Katkı ve Lisans

Katkıda bulunmak için pull request gönderebilir veya issue açabilirsiniz. Lisans bilgisi için lütfen ana dizindeki LICENSE dosyasını inceleyin.

---

SentryBOT ve DeskGUI ile ilgili daha fazla bilgi için [ana proje sayfasını](https://github.com/) ziyaret edebilirsiniz.
