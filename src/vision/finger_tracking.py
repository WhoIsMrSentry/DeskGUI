"""Finger Tracking (opsiyonel).

Bu modül cvzone + mediapipe bağımlılıklarına dayanır. mediapipe Windows ortamlarında
DLL yükleme hatası verebileceği için import işlemleri güvence altına alınmıştır.
Import başarısız olursa modül gracefully devre dışı kalır ve GUI çalışmaya devam eder.
"""

import time
import random
from pubsub import pub

try:  # Riskli bağımlılıkları izole et
    from cvzone.HandTrackingModule import HandDetector  # type: ignore
    _HAND_OK = True
    _HAND_ERR = None
except Exception as _e:  # noqa: BLE001
    HandDetector = None  # type: ignore
    _HAND_OK = False
    _HAND_ERR = _e


class FingerTracking:  # Basitleştirilmiş dayanıklı sürüm
    FINGERTIPS = [4, 8, 12, 16, 20]

    def __init__(self, command_sender=None):
        self.command_sender = command_sender
        self.detector = None
        self.processing_active = False
        self.last_published_command = None
        self.min_command_interval = 0.5
        self.last_command_time = 0
        self.motto_list = [
            "Başarı", "Azim", "Odak", "Sabır", "Güç", "İnanç", "Cesaret", "Bilgi", "Deneyim", "Yaratıcılık"
        ]
        self.argo_list = ["lan", "moruk", "kaptan", "reis", "dayı"]
        self.hand_map = self._load_hand_map()
        if _HAND_OK and HandDetector is not None:
            try:
                self.detector = HandDetector(detectionCon=0.8, maxHands=2)
                self.log("HandDetector yüklendi.")
            except Exception as e:  # noqa: BLE001
                self.log(f"HandDetector başlatma hatası: {e}")
        else:
            self.log(f"cvzone/mediapipe yüklenemedi: {_HAND_ERR}")

    def _load_hand_map(self):  # Jest -> komut
        return {
            "Right": {
                "00000": "stop_tts",
                "10000": "servo_zero",
                "01000": "wave_animation",
                "00100": "argo_sentence_deniz",
                "00010": "say_motto_evolution",
                "00001": "breathe_anim",
                "01111": "play_music",
                "11111": "head_right",
            },
            "Left": {
                "00000": "stop_tts",
                "10000": "servo_ninety",
                "01000": "stacked_bars_animation",
                "00100": "argo_sentence_mali",
                "00010": "say_motto_death",
                "00001": "breathe_anim",
                "01111": "play_music",
                "11111": "head_left",
            },
            "Same": {"00000_00000": "double_reset"},
            "Combined": {"01000_01000": "introduce"},
        }

    def print_all_combinations(self):  # Debug
        for side, combos in self.hand_map.items():
            self.log(f"{side}: {len(combos)} jest")

    def start(self):
        if not _HAND_OK or self.detector is None:
            self.log("Başlatılamıyor (HandDetector yok)")
            return False
        self.processing_active = True
        self.log("FingerTracking başlatıldı")
        return True

    def stop(self):
        """İşlemeyi durdur."""
        self.processing_active = False
        self.log("FingerTracking durduruldu")

    def process_frame(self, frame):
        if not self.processing_active or self.detector is None or frame is None:
            return frame, None
        try:
            hands, img = self.detector.findHands(frame, draw=False)
            hand_types = []
            hand_states = []
            for h in hands[:2]:
                lm = h['lmList']
                htype = h['type']
                hand_types.append(htype)
                states = [0] * 5
                if htype == "Right":  # Başparmak
                    states[0] = 1 if lm[self.FINGERTIPS[0]][0] > lm[self.FINGERTIPS[0]-1][0] else 0
                else:  # Sol başparmak geometrisi ters
                    states[0] = 1 if lm[self.FINGERTIPS[0]][0] < lm[self.FINGERTIPS[0]-1][0] else 0
                for i in range(1,5):
                    states[i] = 1 if lm[self.FINGERTIPS[i]][1] < lm[self.FINGERTIPS[i]-2][1] else 0
                hand_states.append(''.join(map(str, states)))
            cmd = self._determine_command(len(hand_states), hand_types, hand_states)
            now = time.time()
            if cmd and (now - self.last_command_time) > self.min_command_interval and cmd != self.last_published_command:
                self.publish_command(cmd)
                self.last_published_command = cmd
                self.last_command_time = now
            return img, {"command": self.last_published_command}
        except Exception as e:  # noqa: BLE001
            self.log(f"İşleme hatası: {e}")
            return frame, None

    def _determine_command(self, num_hands, hand_types, hands_data):  # Basitleştirilmiş
        if num_hands == 1:
            return self.hand_map.get(hand_types[0], {}).get(hands_data[0])
        if num_hands == 2 and hand_types[0] == hand_types[1] and hands_data[0] == hands_data[1]:
            return self.hand_map.get("Same", {}).get(f"{hands_data[0]}_{hands_data[1]}")
        if num_hands == 2:
            # Sol & sağ kombinasyonu
            try:
                l_idx = hand_types.index('Left')
                r_idx = hand_types.index('Right')
                return self.hand_map.get("Combined", {}).get(f"{hands_data[l_idx]}_{hands_data[r_idx]}")
            except ValueError:
                return None
        return None

    def publish_command(self, command):
        if not command:
            return
        self.log(f"Parmak komutu: {command}")
        pub.sendMessage('gesture_command', command=command)
        if not (self.command_sender and getattr(self.command_sender, 'connected', False)):
            return
        try:
            params = {}
            cmd_type = command
            if command.startswith('say_motto'):
                params = {'text': random.choice(self.motto_list)}
                cmd_type = 'tts'
            elif command.startswith('argo_sentence'):
                params = {'text': f"{command.split('_')[-1].capitalize()} {random.choice(self.argo_list)}"}
                cmd_type = 'tts'
            elif command in ('servo_zero', 'servo_ninety'):
                pct = 0 if command.endswith('zero') else 90
                params = {'identifier': 'pan', 'percentage': pct, 'absolute': True}
                cmd_type = 'servo_move'
            elif command in ('wave_animation','stacked_bars_animation','breathe_anim','head_right','head_left'):
                anim_map = {
                    'wave_animation': 'WAVE_HAND',
                    'stacked_bars_animation': 'STACKED_BARS',
                    'breathe_anim': 'BREATHE',
                    'head_right': 'HEAD_RIGHT',
                    'head_left': 'HEAD_LEFT'
                }
                params = {'animation': anim_map[command]}
                if command == 'breathe_anim':
                    params['color'] = 'BLUE'
                cmd_type = 'send_animation'
            self.command_sender.send_command(cmd_type, params)
        except Exception as e:  # noqa: BLE001
            self.log(f"Komut gönderme hatası: {e}")

    def log(self, message):
        pub.sendMessage('log', msg=f"[FingerTracking] {message}")
