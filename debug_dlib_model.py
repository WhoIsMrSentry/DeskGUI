import dlib
import os
import shutil
import face_recognition_models

# Original path
model_path = face_recognition_models.pose_predictor_model_location()
print(f"Original model path: {model_path}")

# Check file existence and size
if os.path.exists(model_path):
    print(f"File exists. Size: {os.path.getsize(model_path)} bytes")
else:
    print("File does NOT exist at original path!")

# Try to copy to a simple path (C:\Temp or current dir) to rule out long paths/permissions
local_copy = "temp_shape_predictor.dat"
print(f"Copying to local file: {local_copy}...")
try:
    shutil.copy2(model_path, local_copy)
    print("Copy successful.")
except Exception as e:
    print(f"Copy failed: {e}")

# Try loading from local copy
print("Attempting to load from local copy...")
try:
    predictor = dlib.shape_predictor(local_copy)
    print("SUCCESS: Model loaded successfully from local copy!")
except Exception as e:
    print(f"FAILURE: Could not load from local copy: {e}")

# Try loading from original path
print("Attempting to load from original path...")
try:
    predictor = dlib.shape_predictor(model_path)
    print("SUCCESS: Model loaded successfully from original path!")
except Exception as e:
    print(f"FAILURE: Could not load from original path: {e}")
