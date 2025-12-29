import dlib
import os
import shutil
import face_recognition_models
import tempfile

# ASCII Temp Path
temp_dir = tempfile.gettempdir() # C:\Users\emirh\AppData\Local\Temp
ascii_model_path = os.path.join(temp_dir, "shape_predictor_68_face_landmarks.dat")

original_path = face_recognition_models.pose_predictor_model_location()

print(f"Original Path (Unicode?): {original_path}")
print(f"ASCII Safe Path: {ascii_model_path}")

try:
    shutil.copy2(original_path, ascii_model_path)
    print("Copied to ASCII path.")
except Exception as e:
    print(f"Copy failed: {e}")

try:
    print("Attempting to load from ASCII path...")
    predictor = dlib.shape_predictor(ascii_model_path)
    print("SUCCESS: Loaded from ASCII path!")
except Exception as e:
    print(f"FAILURE: Could not load from ASCII path: {e}")

# If this works, we know the issue is the path encoding.
