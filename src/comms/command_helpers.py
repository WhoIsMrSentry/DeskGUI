import logging
import os
import json
import time
import traceback
from pubsub import pub

class VisionCommands:
    """Görüntü işleme modülleri için komut formatları ve yardımcıları"""
    
    @staticmethod
    def format_gesture_command(command_name, params=None):
        """El hareketlerinden gelen komutları robot için formatlayın"""
        base_params = {"command": command_name}
        if params:
            base_params.update(params)
        return base_params
    
    @staticmethod
    def format_emotion_command(emotion, intensity=1.0):
        """Duygu tespiti komutlarını robot için formatlayın"""
        return {
            "emotion": emotion,
            "intensity": intensity
        }
    
    @staticmethod
    def format_object_detection_command(object_label, confidence, bbox=None):
        """Nesne tespiti komutlarını robot için formatlayın"""
        command = {
            "label": object_label,
            "confidence": confidence
        }
        if bbox:
            command["bbox"] = bbox
        return command
    
    @staticmethod
    def format_tracking_command(tracking_data):
        """İzleme verilerini robot için formatlayın"""
        return tracking_data
    
    @staticmethod
    def log_command(command_type, command_data, logger=None):
        """Komutları logla"""
        log_message = f"VISION CMD - {command_type}: {command_data}"
        if logger:
            logger.debug(log_message)
        else:
            logging.debug(log_message)

class CommandHelpers:
    """Robot kontrol ve etkileşim komutları için yardımcı fonksiyonlar."""
    
    @staticmethod
    def get_available_animations():
        """Kullanılabilir animasyonların listesini döndürür."""
        neopixel_animations = [
            "RAINBOW", "RAINBOW_CYCLE", "SPINNER", "BREATHE", 
            "METEOR", "FIRE", "COMET", "WAVE", "PULSE", 
            "TWINKLE", "COLOR_WIPE", "RANDOM_BLINK", 
            "THEATER_CHASE", "SNOW", "ALTERNATING", 
            "GRADIENT", "BOUNCING_BALL", "RUNNING_LIGHTS", 
            "STACKED_BARS"
        ]
        # Servo animasyonlarını genişlet
        servo_animations = [
            "HEAD_NOD", "LOOK_UP", "WAVE_HAND", "CENTER",
            "HEAD_LEFT", "HEAD_RIGHT", "BOUNCE", "CELEBRATE", "HEAD_NOD_ABS", "HEAD_SHAKE", "HEAD_SHAKE_ABS", "LOOK_DOWN", "RAISED", "SCAN", "SIT", "SLEEP"
        ]
        return {
            "neopixel": neopixel_animations,
            "servo": servo_animations
        }
    
    @staticmethod
    def get_available_colors():
        """Önceden tanımlanmış ESP32-Arduino renk listesini döndürür."""
        return [
            "RED", "GREEN", "BLUE", "YELLOW", "PURPLE", 
            "CYAN", "WHITE", "ORANGE", "PINK", "GOLD", 
            "TEAL", "MAGENTA", "LIME", "SKY_BLUE", "NAVY", 
            "MAROON", "AQUA", "VIOLET", "CORAL", "TURQUOISE"
        ]
    
    @staticmethod
    def get_animation_info(animation_name):
        """Belirli bir animasyon hakkında bilgi döndürür."""
        info = {
            "RAINBOW": {"description": "Tüm renkler aynı anda değişir", "color_needed": False, "color2_needed": False},
            "RAINBOW_CYCLE": {"description": "Renkler tek tek değişir", "color_needed": False, "color2_needed": False},
            "SPINNER": {"description": "İlerleme çubuğu şeklinde dönen animasyon", "color_needed": True, "color2_needed": False},
            "BREATHE": {"description": "Rengin parlaklığı nefes alır gibi değişir", "color_needed": True, "color2_needed": False},
            "METEOR": {"description": "Meteor yağmuru animasyonu", "color_needed": True, "color2_needed": False},
            "FIRE": {"description": "Ateş alevi animasyonu", "color_needed": False, "color2_needed": False},
            "COMET": {"description": "Kuyruklu yıldız animasyonu", "color_needed": True, "color2_needed": False},
            "WAVE": {"description": "Dalga animasyonu", "color_needed": False, "color2_needed": False},
            "PULSE": {"description": "Nabız şeklinde yanıp sönme", "color_needed": True, "color2_needed": False},
            "TWINKLE": {"description": "Yıldız parıltısı efekti", "color_needed": True, "color2_needed": False},
            "COLOR_WIPE": {"description": "Renk silme efekti", "color_needed": True, "color2_needed": False},
            "RANDOM_BLINK": {"description": "Rastgele yanıp sönme", "color_needed": False, "color2_needed": False},
            "THEATER_CHASE": {"description": "Tiyatro takip efekti", "color_needed": True, "color2_needed": False},
            "SNOW": {"description": "Kar yağışı efekti", "color_needed": True, "color2_needed": False},
            "ALTERNATING": {"description": "İki renk arasında geçiş", "color_needed": True, "color2_needed": True},
            "GRADIENT": {"description": "Renk geçişi animasyonu", "color_needed": False, "color2_needed": False},
            "BOUNCING_BALL": {"description": "Zıplayan top efekti", "color_needed": True, "color2_needed": False},
            "RUNNING_LIGHTS": {"description": "Koşan ışıklar efekti", "color_needed": True, "color2_needed": False},
            "STACKED_BARS": {"description": "Yığılmış çubuklar efekti", "color_needed": False, "color2_needed": False},
            
            # Servo Animasyonları
            "HEAD_NOD": {"description": "Başını sallar", "servo": True},
            "LOOK_UP": {"description": "Yukarı bakar", "servo": True},
            "WAVE_HAND": {"description": "Elini sallar", "servo": True},
            "CENTER": {"description": "Servoları merkeze alır", "servo": True},
            "BOUNCE": {"description": "Kısa zıplama hareketi yapar", "servo": True},
            "CELEBRATE": {"description": "Kutlama hareketi yapar", "servo": True},
            "HEAD_LEFT": {"description": "Başı sola çevirir", "servo": True},
            "HEAD_NOD_ABS": {"description": "Başını mutlak açıyla sallar", "servo": True},
            "HEAD_RIGHT": {"description": "Başı sağa çevirir", "servo": True},
            "HEAD_SHAKE": {"description": "Başı hayır anlamında sallar", "servo": True},
            "HEAD_SHAKE_ABS": {"description": "Başı mutlak açıyla hayır anlamında sallar", "servo": True},
            "LOOK_DOWN": {"description": "Aşağı bakar", "servo": True},
            "RAISED": {"description": "Başını kaldırır", "servo": True},
            "SCAN": {"description": "Çevreyi tarar", "servo": True},
            "SIT": {"description": "Oturma pozisyonuna geçer", "servo": True},
            "SLEEP": {"description": "Uyuma pozisyonuna geçer", "servo": True}
        }
        
        return info.get(animation_name.upper(), {"description": "Bilinmeyen animasyon", "color_needed": True, "color2_needed": False})
    
    @staticmethod
    def send_animation(command_sender, animation_name, color=None, color2=None, repeat=1):
        """
        Animasyon komutunu uygun şekilde gönderir.
        Servo animasyonları için 'servo_move', diğerleri için 'send_animation' kullanılır.
        """
        animations = CommandHelpers.get_available_animations()
        servo_animations = [a.upper() for a in animations["servo"]]
        animation_name_upper = animation_name.upper()
        if animation_name_upper in servo_animations:
            # Servo animasyonu
            params = {"animation": animation_name, "repeat": repeat}
            command_sender.send_command("servo_move", params)
        else:
            # NeoPixel veya diğer animasyonlar
            params = {"animation": animation_name, "repeat": repeat}
            if color:
                params["color"] = color
            if color2:
                params["color2"] = color2
            command_sender.send_command("send_animation", params)
    
    @staticmethod
    def format_servo_command(identifier, percentage, absolute=False):
        """Servo komutu için parametreleri düzenler."""
        return {
            'identifier': identifier,
            'percentage': int(percentage),
            'absolute': absolute
        }
    
    @staticmethod
    def get_best_voice_for_language(tts_voices, language_code):
        """Belirli bir dil için en uygun sesi bulur."""
        if not language_code or not tts_voices:
            return None
        
        # Tam eşleşme dene
        if language_code in tts_voices and tts_voices[language_code]:
            return tts_voices[language_code][0]['id']
        
        # Dil öneki eşleşmesi dene
        language_prefix = language_code.split('-')[0].lower()
        for code, voices in tts_voices.items():
            if code.lower().startswith(language_prefix) and voices:
                return voices[0]['id']
        
        return None
    
    @staticmethod
    def hex_to_rgb(hex_color):
        """Hex renk kodunu RGB'ye dönüştürür."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        
        return {
            'r': int(hex_color[0:2], 16),
            'g': int(hex_color[2:4], 16),
            'b': int(hex_color[4:6], 16)
        }
    
    @staticmethod
    def get_loaded_model_paths():
        """Yüklenen model dosyalarının listesini döndürür."""
        from src.config import get_model_path, MODELS_DIR
        
        if not os.path.exists(MODELS_DIR):
            return {"status": "error", "message": f"Model dizini bulunamadı: {MODELS_DIR}", "models": []}
        
        models = []
        for filename in os.listdir(MODELS_DIR):
            file_path = os.path.join(MODELS_DIR, filename)
            if os.path.isfile(file_path):
                models.append({
                    "name": filename,
                    "path": file_path,
                    "size": os.path.getsize(file_path),
                    "modified": time.ctime(os.path.getmtime(file_path))
                })
        
        return {
            "status": "ok", 
            "message": f"{len(models)} model dosyası bulundu", 
            "models": models,
            "models_dir": MODELS_DIR
        }
