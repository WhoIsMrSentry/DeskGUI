# speech_input.py

import time
import sounddevice as sd
# import speech_recognition as sr # __init__ içine taşındı

class SpeechInput:
    def __init__(self, protocol="direct", bluetooth_server=None, language="tr-TR", auto_discover=True):
        self._listening = False
        self._audio = None # sr.Recognizer instance
        self.TARGET_MIC_INDEX = 8 # <<< BURAYI DENEMEK İÇİN 24 YAPALIM VEYA 25
        # self._listener = None # sr.Microphone instance -> start içinde oluşturulacak
        self.language = language
        self.protocol = protocol
        print(f"SpeechInput __init__: Kullanılacak hedef mikrofon index'i: {self.TARGET_MIC_INDEX}")
        self.bluetooth_server = bluetooth_server
        self.auto_discover = auto_discover
        self.stop_listener_func = None

        try:
            import speech_recognition as sr
            self.sr = sr
            print("SpeechInput __init__: speech_recognition kütüphanesi başarıyla yüklendi.")
            # Mikrofon listesini başlangıçta bir kere loglayalım
            try:
                print("SpeechInput __init__: Mevcut mikrofonlar (speech_recognition):")
                mic_names = self.sr.Microphone.list_microphone_names()
                for i, name in enumerate(mic_names):
                    print(f"  Index {i}: {name}")
            except Exception as e_mic_list:
                print(f"SpeechInput __init__: Mikrofon listesi alınırken hata: {e_mic_list}")
        except ImportError:
            self.sr = None
            print("HATA: SpeechInput __init__: SpeechRecognition kütüphanesi yüklenemedi. Lütfen `pip install SpeechRecognition` ile kurun.")
        except Exception as e:
            self.sr = None
            print(f"HATA: SpeechInput __init__: speech_recognition yüklenirken beklenmedik hata: {e}")

    def start(self):
        """Konuşma tanımayı başlat. Başarı durumunu döndürür (bool, mesaj)."""
        print("SpeechInput.start() çağrıldı.")

        if not self.sr:
            print("HATA: SpeechInput.start(): speech_recognition kütüphanesi yüklenemediği için başlatılamıyor.")
            return False, "speech_recognition kütüphanesi yok"

        try:
            if self._listening:
                print("SpeechInput: Konuşma tanıma zaten aktif.")
                return True, "Zaten aktif"

            print("SpeechInput: sr.Recognizer() oluşturuluyor...")
            self._audio = self.sr.Recognizer()
            self._audio.dynamic_energy_threshold = True # <<< DİNAMİK EŞİĞİ AKTİF EDELİM
            # self._audio.energy_threshold = 3000 # Dinamik açıkken bu genellikle gereksiz
            print(f"SpeechInput: Recognizer ayarlandı: dynamic_energy_threshold={self._audio.dynamic_energy_threshold}")


            print(f"SpeechInput: sr.Microphone(device_index={self.TARGET_MIC_INDEX}) oluşturuluyor...")
            try:
                self._listener = self.sr.Microphone(device_index=self.TARGET_MIC_INDEX)
                active_mic_name = "Bilinmiyor"
                if self.TARGET_MIC_INDEX is not None:
                    try:
                        # sr.Microphone.list_microphone_names() bir liste döndürür
                        all_mics = self.sr.Microphone.list_microphone_names()
                        if 0 <= self.TARGET_MIC_INDEX < len(all_mics):
                            active_mic_name = all_mics[self.TARGET_MIC_INDEX]
                        else:
                            active_mic_name = f"Geçersiz Index ({self.TARGET_MIC_INDEX})"
                    except Exception as e_mic_name:
                        active_mic_name = f"Ad alınırken hata: {e_mic_name}"
                else:
                    active_mic_name = "Varsayılan sistem mikrofonu"
                print(f"SpeechInput: sr.Microphone() oluşturuldu. Kullanılan mikrofon: {active_mic_name}")

            except Exception as mic_err:
                print(f"HATA: sr.Microphone(index={self.TARGET_MIC_INDEX}) oluşturulurken kritik hata: {mic_err}")
                self._listening = False
                return False

            # adjust_for_ambient_noise'i listen_in_background öncesi çağırmak yerine
            # Recognizer'ın dynamic_energy_threshold ayarına güvenelim.
            # Eğer ses kalitesi çok düşükse, bu satırı tekrar aktif edip test edebilirsiniz:
            # print("SpeechInput: Ortam gürültüsü ayarlanıyor...")
            # with self._listener as source_for_adjust: # Bu satır sorun çıkarabilir!
            #    self._audio.adjust_for_ambient_noise(source_for_adjust, duration=0.5)
            # print("SpeechInput: Gürültü ayarı tamamlandı.")

            def callback(recognizer, audio_data):
                print(f"SpeechInput Callback: Ses verisi alındı (uzunluk: {len(audio_data.get_raw_data()) if audio_data else 0} bytes). Tanıma deneniyor...")
                try:
                    text = recognizer.recognize_google(audio_data, language=self.language)
                    if text:
                        print(f"SpeechInput Callback: Tanınan metin: [{text}]")
                        from pubsub import pub
                        pub.sendMessage('speech', text=text)
                    else:
                        print("SpeechInput Callback: recognize_google boş metin döndürdü.")
                except self.sr.UnknownValueError:
                    print("SpeechInput Callback: Google Speech Recognition sesi anlayamadı.")
                except self.sr.RequestError as e:
                    print(f"SpeechInput Callback: Google Speech Recognition servisinden sonuç istenemedi; {e}")
                except Exception as e_rec:
                    print(f"SpeechInput Callback: Konuşma tanıma callback hatası: {e_rec}")
                    import traceback
                    traceback.print_exc()

            print(f"SpeechInput: Arka planda dinleme '{self.language}' diliyle başlatılıyor...")
            self.stop_listener_func = self._audio.listen_in_background(self._listener, callback, phrase_time_limit=15)
            self._listening = True
            print("SpeechInput: Arka planda dinleme başarıyla başlatıldı.")
            return True, "Başarıyla başlatıldı"
        except Exception as e:
            print(f"HATA: SpeechInput.start(): Başlatılırken genel hata: {e}")
            import traceback
            traceback.print_exc()
            self._listening = False
            return False, f"Hata: {e}"

    def stop_listening(self, wait_for_stop=True):
        print("SpeechInput.stop_listening() çağrıldı.")
        if not self._listening:
            print("SpeechInput: Konuşma tanıma zaten kapalı.")
            return True

        print("SpeechInput: Konuşma tanıma durduruluyor...")
        try:
            if self.stop_listener_func:
                self.stop_listener_func(wait_for_stop=wait_for_stop)
                print("SpeechInput: Arka plan dinleyicisi durduruldu.")
                self.stop_listener_func = None
            else:
                print("SpeechInput: Durdurulacak aktif bir dinleyici fonksiyonu bulunamadı.")

            self._listening = False
            # self._listener = None # Microphone nesnesi start içinde yeniden oluşturulacak
            # self._audio = None    # Recognizer nesnesi start içinde yeniden oluşturulacak
            print("SpeechInput: Konuşma tanıma başarıyla durduruldu.")
            return True
        except Exception as e:
            print(f"HATA: SpeechInput: Konuşma tanıma durdurulurken hata: {e}")
            import traceback
            traceback.print_exc()
            self._listening = False
            return False