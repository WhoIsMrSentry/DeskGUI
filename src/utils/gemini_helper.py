# gemini helper.py: # Google Gemini (genai) API entegrasyonu için yardımcı modül
# https://github.com/googleapis/python-genai

import os
import google.generativeai as genai
# Genai types'ı kullanmak için import edin (genai.types.Blob için gerekli)
from google.generativeai import types
from PIL import Image
import io

class GeminiHelper:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest"): # Default modeli güncelleyelim
        if not api_key or not isinstance(api_key, str) or api_key.strip() == "":
            raise ValueError("Gemini API anahtarı eksik veya geçersiz! Lütfen geçerli bir API anahtarı girin.")
        self.api_key = api_key
        self.model = model
        try:
            genai.configure(api_key=api_key)
            # Modelin varlığını kontrol etmek iyi olabilir, ancak bu init sırasında ekstra istek yapar.
            # Hata generate_content sırasında da yakalanabilir.
            self.gemini = genai.GenerativeModel(model)
            print(f"GeminiHelper: Model '{model}' başlatıldı.") # Log ekleyelim
        except Exception as e:
            print(f"GeminiHelper: Model başlatılırken hata oluştu: {e}")
            raise # Hatayı yeniden fırlat ki GUI bu durumu anlayabilsin

        self.temperature = None
        self.top_k = None
        self.top_p = None
        self.safety_settings = None
        self.system_instruction = None

    def set_parameters(self, temperature=None, top_k=None, top_p=None, safety_settings=None, system_instruction=None):
        """Gemini API için ayarları güncelle"""
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.safety_settings = safety_settings
        self.system_instruction = system_instruction
        # Parametreler ayarlandığında logla
        print(f"GeminiHelper: Parametreler ayarlandı/güncellendi - Temp:{self.temperature}, TopK:{self.top_k}, TopP:{self.top_p}, Safety set:{self.safety_settings is not None}, SysInstr set:{self.system_instruction is not None}")

    def _build_kwargs(self):
        """generate_content metoduna uygun keyword argümanları hazırlar."""
        kwargs = {}
        generation_config_params = {} # generation_config için ayrı bir dict

        if self.temperature is not None:
            generation_config_params['temperature'] = self.temperature
        if self.top_k is not None:
            generation_config_params['top_k'] = self.top_k
        if self.top_p is not None:
            generation_config_params['top_p'] = self.top_p
        # self.max_output_tokens gibi diğer generation_config parametreleri de buraya eklenebilir

        if generation_config_params:
            # types.GenerationConfig kullanmak daha doğru
            if hasattr(types, 'GenerationConfig'):
                 kwargs['generation_config'] = types.GenerationConfig(**generation_config_params)
            else: # Fallback, eğer types import edilemediyse veya eski versiyon
                 kwargs['generation_config'] = generation_config_params
            print(f"GeminiHelper: generation_config oluşturuldu: {kwargs['generation_config']}")

        # safety_settings doğrudan kwargs'a eklenir
        if self.safety_settings is not None:
            if isinstance(self.safety_settings, (list, dict)):
                kwargs['safety_settings'] = self.safety_settings
                print(f"GeminiHelper: safety_settings eklendi: {self.safety_settings}") # Debug logu
            else:
                print(f"GeminiHelper: Geçersiz safety_settings formatı ({type(self.safety_settings)}). Atlanıyor.")

        print(f"GeminiHelper: Final kwargs: {kwargs}") # Debug logu
        return kwargs

    def send_prompt(self, prompt: str, images: list = None, **kwargs):
        """
        Gemini'ye metin ve opsiyonel görsel gönderir.
        images: PIL.Image.Image veya bytes listesi
        """
        print(f"GeminiHelper.send_prompt çağrıldı. Prompt (ilk 30 char): '{prompt[:30]}...'") # Debug logu

        contents = []
        # Önce system_instruction varsa ekle
        if self.system_instruction and self.system_instruction.strip():
            contents.append({
                "role": "model",  # Gemini API gereği 'model' olmalı
                "parts": [{"text": self.system_instruction}]
            })
        # Promptu user olarak ekle
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        # Görselleri eklemek istersen, parts içine ekleyebilirsin (isteğe bağlı)
        # Şu an prompt metni olarak gönderiliyor, istersen burada genişletebilirsin.

        merged_kwargs = self._build_kwargs()
        merged_kwargs.update(kwargs) # Dışarıdan gelen ek kwargs'ları ekle

        print(f"GeminiHelper: generate_content çağrılıyor contents={contents}, kwargs={merged_kwargs}") # Debug logu
        try:
            response = self.gemini.generate_content(contents=contents, **merged_kwargs)
            print(f"GeminiHelper: generate_content yanıtı alındı.") # Debug logu

            # Yanıtı kontrol et ve metni döndür
            if hasattr(response, 'text') and response.text:
                return response.text
            elif hasattr(response, 'parts') and response.parts:
                all_text_parts = [part.text for part in response.parts if hasattr(part, 'text')]
                if all_text_parts:
                    return "".join(all_text_parts)
            elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.safety_ratings:
                return response
            print(f"GeminiHelper: Yanıtta .text veya .parts bulunamadı veya boş. Yanıt objesi: {response}")
            return ""
        except Exception as e:
            print(f"GeminiHelper: generate_content çağrısında hata: {e}")
            raise e

    def send_prompt_stream(self, prompt: str, images: list = None, **kwargs):
        """
        Streaming yanıt almak için (generator döner)
        """
        print(f"GeminiHelper.send_prompt_stream çağrıldı. Prompt (ilk 30 char): '{prompt[:30]}...'") # Debug logu
        parts = [prompt]
        if images:
            print(f"GeminiHelper: {len(images)} görsel eklendi (stream için).") # Debug logu
            for i, img in enumerate(images):
                 try:
                    if isinstance(img, Image.Image):
                        img_bytes = io.BytesIO()
                        img.save(img_bytes, format='PNG')
                        img_bytes.seek(0)
                        parts.append(types.Blob("image/png", img_bytes.read())) # Use imported types
                    elif isinstance(img, bytes):
                         parts.append(types.Blob("image/png", img)) # Use imported types
                    else:
                        print(f"GeminiHelper: Geçersiz görsel tipi atlandı (stream): {type(img)}") # Debug logu
                 except Exception as img_err:
                    print(f"GeminiHelper: Görsel işlenirken hata (stream, index {i}): {img_err}") # Debug logu

        merged_kwargs = self._build_kwargs()
        merged_kwargs.update(kwargs)

        print(f"GeminiHelper: generate_content (stream) çağrılıyor parts={parts}, kwargs={merged_kwargs}") # Debug logu
        try:
            # Stream=True olarak ayarlanmış
            for chunk in self.gemini.generate_content(parts, stream=True, **merged_kwargs):
                 yield chunk.text if hasattr(chunk, 'text') else str(chunk) # Chunk'ın metni veya string temsili
            print("GeminiHelper: Stream tamamlandı.") # Debug logu
        except Exception as e:
            print(f"GeminiHelper: generate_content (stream) çağrısında hata: {e}") # Debug logu
            # Generator hatayı yakalamalı veya dışarıdan fırlatılmalı.
            # raise e # Generator içinde raise kullanmak genellikle beklenmez.
            # Bunun yerine hata durumunu döndürebilir veya loglayabiliriz.
            # Şimdilik sadece loglayalım. _process_llm_request stream kullanmadığı için bu kısım etkilenmiyor.
            pass # Hata durumunda generator biter.