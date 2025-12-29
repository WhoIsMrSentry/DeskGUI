
import requests

class TranslateHelper:
    API_URL = "https://libretranslate.de/translate"

    @staticmethod
    def translate(text, source_lang, target_lang):
        if not text or source_lang == target_lang:
            return text
        try:
            response = requests.post(
                TranslateHelper.API_URL,
                data={
                    "q": text,
                    "source": source_lang,
                    "target": target_lang,
                    "format": "text"
                },
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("translatedText", text)
            else:
                return text
        except Exception:
            return text