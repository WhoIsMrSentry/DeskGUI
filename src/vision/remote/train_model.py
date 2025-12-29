# ----- modules/vision/remote/train_model.py -----

from imutils import paths
import face_recognition
import pickle
import cv2
import os
import shutil
# pubsub import'unu kaldırabiliriz veya loglama için saklayabiliriz, ama sinyal daha iyi.
# from pubsub import pub
import time
import json
import traceback
import numpy as np

# PyQt sinyalini burada tanımlayamayız, dışarıdan alacağız.

import src.config as config

class TrainModel:
    # __init__ metoduna log_signal parametresi ekle
    def __init__(self, log_signal=None, **kwargs): # log_signal eklendi
        self.path = kwargs.get('dataset', 'dataset')
        self.output = kwargs.get('output', str(config.ENCODINGS_FILE))
        self.model = kwargs.get('detection_method', 'hog')
        self.jitters = kwargs.get('jitters', 1)
        self.progress_callback = kwargs.get('progress_callback', None)
        self.log_signal = log_signal # Sinyali sakla

    # Loglama için helper metot
    def _log(self, message):
        # Eğer sinyal varsa onu kullan, yoksa print et
        if self.log_signal:
            try:
                # Sinyalin emit metodunu çağırmayı dene
                # Emin olmak için signal nesnesinin varlığını ve callable olup olmadığını kontrol et
                if hasattr(self.log_signal, 'emit') and callable(self.log_signal.emit):
                     self.log_signal.emit(f'[TrainModel] {message}')
                else:
                     print(f'[TrainModel-Fallback] {message}') # Sinyal emit edilemiyorsa
            except Exception as e:
                 print(f'[TrainModel-SignalError] {message} (Log Signal Error: {e})')
        else:
            print(f'[TrainModel-Print] {message}') # Sinyal hiç verilmediyse

    def ensure_dataset_exists(self):
        """Dataset klasörünün varlığını kontrol et ve yoksa oluştur"""
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            self._log(f'Created dataset directory: {self.path}') # _log kullan

    def train(self):
        """Yüz tanıma modelini eğit"""
        self.ensure_dataset_exists()

        imagePaths = list(paths.list_images(self.path))
        if len(imagePaths) < 1:
            self._log('Nothing to process - no images found in dataset') # _log kullan
            return 0

        self._log(f'Start processing {len(imagePaths)} images...') # _log kullan
        knownEncodings = []
        knownNames = []
        success_count = 0
        fail_count = 0
        start_train_time = time.time() # Başlangıç zamanı

        for (i, imagePath) in enumerate(imagePaths):
            if '.AppleDouble' in imagePath:
                continue

            if self.progress_callback:
                progress = int(((i+1) / len(imagePaths)) * 100) # i+1 kullan
                # Progress callback'in GUI'yi güncellemediğinden emin olunmalı!
                # Sadece print veya log için kullanılmalı.
                # Eğer GUI güncellemesi yapıyorsa, sinyal ile yapılmalı.
                # Şimdilik callback'i olduğu gibi bırakıyoruz ama riskli olabilir.
                try:
                    self.progress_callback(progress, imagePath)
                except Exception as cb_err:
                    self._log(f"Error in progress callback: {cb_err}")

            self._log(f'Processing image {i + 1}/{len(imagePaths)} - {os.path.basename(imagePath)}') # Sadece dosya adı

            name = imagePath.split(os.path.sep)[-2]

            image = cv2.imread(imagePath)
            if image is None:
                self._log(f'Warning: Could not read image {imagePath}') # _log kullan
                fail_count += 1
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                # Yüz bulma
                self._log(f'  Finding faces in {os.path.basename(imagePath)} using {self.model}...') # Ek log
                loc_start = time.time()
                boxes = face_recognition.face_locations(rgb, model=self.model)
                loc_end = time.time()
                self._log(f'  Found {len(boxes)} face(s) in {loc_end - loc_start:.2f}s') # Ek log

                if len(boxes) == 0:
                    self._log(f'Warning: No face detected in {imagePath}') # _log kullan
                    fail_count += 1
                    continue

                # Encoding çıkarma
                self._log(f'  Encoding {len(boxes)} face(s) with jitters={self.jitters}...') # Ek log
                enc_start = time.time()
                encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=self.jitters)
                enc_end = time.time()
                self._log(f'  Encoding done in {enc_end - enc_start:.2f}s') # Ek log

                valid_encodings = []
                for encoding in encodings:
                    if np.isnan(encoding).any():
                        self._log(f'Warning: NaN values in encoding for {imagePath}') # _log kullan
                        continue
                    if len(encoding) != 128:
                        self._log(f'Warning: Incorrect encoding dimension ({len(encoding)}) for {imagePath}') # _log kullan
                        continue
                    # Değer aralığı kontrolü çok katı olabilir, şimdilik kaldıralım veya gevşetelim.
                    # if np.min(encoding) < -1.1 or np.max(encoding) > 1.1:
                    #    self._log(f'Warning: Encoding values out of range for {imagePath}')
                    #    continue
                    valid_encodings.append(encoding)

                if not valid_encodings:
                    self._log(f'Warning: No valid encodings found for {imagePath}') # _log kullan
                    fail_count += 1
                    continue

                for encoding in valid_encodings:
                    knownEncodings.append(encoding)
                    knownNames.append(name)
                success_count += len(valid_encodings) # Başarılı encoding sayısı kadar artır

            except Exception as e:
                self._log(f'Error processing {imagePath}: {e}') # _log kullan
                self._log(traceback.format_exc()) # Traceback'i de logla
                fail_count += 1
                continue

        end_train_time = time.time() # Bitiş zamanı
        self._log(f"Image processing finished in {end_train_time - start_train_time:.2f}s. Success: {success_count}, Fail: {fail_count}")

        if len(knownEncodings) == 0:
            self._log('No valid face encodings generated from any images.') # _log kullan
            return 0 # 0 döndürmek önemli

        # Kaydetme
        self._log(f'Serializing {len(knownEncodings)} encodings...') # _log kullan
        data = {"encodings": knownEncodings, "names": knownNames}
        temp_output = f"{self.output}.tmp"

        try:
            with open(temp_output, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL) # pickle.dump kullanmak daha standart
            self._log(f'Successfully wrote to temporary file {temp_output}') # Ek log

            # Taşıma/Yedekleme
            backup = f"{self.output}.bak_{int(time.time())}" # Daha belirgin backup adı
            if os.path.exists(self.output):
                try:
                    # Önce yedekle, sonra sil, sonra taşı (daha güvenli)
                    shutil.copy2(self.output, backup)
                    os.remove(self.output)
                    self._log(f'Backed up old model to {backup} and removed original.') # Ek log
                except Exception as e:
                    self._log(f'Warning: Could not backup/remove old model: {e}')

            try:
                 shutil.move(temp_output, self.output)
                 self._log(f'Saved encodings to {self.output}') # _log kullan
            except Exception as move_err:
                 self._log(f'!!! CRITICAL ERROR: Failed to move {temp_output} to {self.output}: {move_err}')
                 # Eğer taşıma başarısız olursa, en azından geçici dosya kalır
                 return 0 # Başarısızlık olarak işaretle

        except Exception as pickle_err:
             self._log(f'!!! CRITICAL ERROR during pickling/saving: {pickle_err}')
             self._log(traceback.format_exc())
             # Geçici dosyayı silmeye çalışalım (varsa)
             if os.path.exists(temp_output):
                 try: os.remove(temp_output)
                 except: pass
             return 0 # Başarısızlık

        unique_names = set(knownNames)
        self._log(f'Model contains {len(unique_names)} unique persons: {", ".join(sorted(list(unique_names)))}') # _log kullan

        return len(knownEncodings) # Başarılı encoding sayısını döndür