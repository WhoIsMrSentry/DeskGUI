#!/usr/bin/env python3
"""debug_imports.py

Amaç:
    Ortamınızdaki paketlerin import edilebilirliğini test eder ve özellikle
    'DLL load failed while importing _framework_bindings' gibi hataların kök
    nedenini bulmaya yardımcı olacak ek teşhis çıktıları üretir.

Özellikler:
    - Gerekli & opsiyonel modülleri listeleyip sürümünü gösterir
    - Hata olursa kısaltılmış traceback + çözüm önerisi verir
    - desk_gui import'unu aşamalı test eder (yüksek riskli bağımlılıkları önce dener)
    - Çıkış kodu: 0 (başarılı), 1 (kritik hata)
"""

from __future__ import annotations
import importlib
import sys
import traceback
import platform
import os
from typing import Optional, Dict, Any


def short_tb(exc: BaseException, limit: int = 6) -> str:
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    if len(tb_lines) > limit:
        tb_lines = tb_lines[-limit:]
    return ''.join(tb_lines)


def try_import(name: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"name": name, "ok": False, "version": None, "error": None, "hint": None}
    try:
        module = importlib.import_module(name)
        info["ok"] = True
        # Versiyon alma stratejisi
        ver = getattr(module, "__version__", None)
        if ver is None:
            # Bazı modüller farklı attribute kullanır
            for attr in ("VERSION", "version", "__VERSION__"):
                if hasattr(module, attr):
                    cand = getattr(module, attr)
                    if isinstance(cand, (str, tuple)):
                        ver = cand
                        break
        info["version"] = ver
    except BaseException as e:  # noqa: BLE001 - SystemError dahil
        info["error"] = short_tb(e)
        info["hint"] = suggest_fix(name, e)
    return info


def suggest_fix(name: str, exc: BaseException) -> Optional[str]:
    msg = str(exc).lower()
    if "_framework_bindings" in msg or "dll load failed" in msg:
        # Genelde mediapipe / protobuf / görsel C++ runtime problemi
        base = (
            "DLL yükleme hatası. Olası nedenler: (1) mediapipe sürüm uyumsuzluğu, (2) Visual C++ Redistributable eksik, "
            "(3) Python'u Microsoft Store yerine python.org'dan kurmanız gerekebilir, (4) AVX/CPU uyumsuz binary."
        )
        if name.startswith("mediapipe") or "mediapipe" in msg:
            return base + " Öneri: 'pip install --upgrade --force-reinstall mediapipe==0.10.14' deneyin."
        if name == "desk_gui":
            return base + " Önce riskli bağımlılıkları (cvzone, mediapipe, face_recognition) tek tek test edin."
        return base
    if name == "face_recognition" and ("dlib" in msg or "cmake" in msg):
        return "face_recognition için dlib derleme sorunu. 'pip install cmake' ve Visual Studio C++ Build Tools kurulu olmalı."
    return None


def print_section(title: str):
    print("\n" + title)
    print("-" * len(title))


def diagnose_desk_gui():
    print_section("desk_gui modülünü aşamalı teşhis")
    # Adım 1: Sadece PyQt5
    for step_name, import_list in [
        ("PyQt5 temel", ["PyQt5"]),
        ("GUI alt modülleri", ["PyQt5.QtWidgets", "PyQt5.QtCore", "PyQt5.QtGui"]),
        ("Yüksek riskli bağımlılıklar", ["cv2", "cvzone.HandTrackingModule", "mediapipe", "face_recognition"]),
    ]:
        print(f"[Adım] {step_name}")
        for mod in import_list:
            r = try_import(mod)
            status = "✅" if r["ok"] else "❌"
            ver = f" - {r['version']}" if r["ok"] and r["version"] else ""
            print(f"  {status} {mod}{ver}")
            if not r["ok"]:
                if r["hint"]:
                    print(f"    Öneri: {r['hint']}")
                print("    Hata kısaltılmış traceback:\n" + indent(r["error"], 8))
                print("  --> Bu noktada desk_gui import'u başarısız olacaktır. Diğer adımlar atlanıyor.\n")
                return
    print("Ön adımlar başarıyla yüklendi, şimdi 'desk_gui' import deneniyor...")
    r = try_import("desk_gui")
    if r["ok"]:
        print("✅ desk_gui modülü başarıyla import edildi.")
    else:
        print("❌ desk_gui import başarısız")
        if r["hint"]:
            print(f"Öneri: {r['hint']}")
        print("Kısaltılmış traceback:\n" + indent(r["error"], 2))


def indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return ''.join(pad + line if line.strip() else line for line in text.splitlines(True))


def main():  # noqa: D401
    print("Debug Imports - Ortam Teşhisi")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Working directory: {os.getcwd()}")
    print("Sys.path:")
    for p in sys.path:
        print(f"  - {p}")

    required = [
        "PyQt5", "cv2", "numpy", "face_recognition", "json", "pickle", "requests",
        "threading", "pubsub", "time", "socket"
    ]
    optional = ["fastapi", "uvicorn", "pyttsx3", "speech_recognition", "onnxruntime", "pydub", "pygame"]

    print_section("Gerekli Modüller")
    numpy_version = None
    for name in required:
        r = try_import(name)
        status = "✅" if r["ok"] else "❌"
        version = f" - {r['version']}" if r["ok"] and r["version"] else ""
        print(f"{status} {name}{version}")
        if name == "numpy" and r["ok"]:
            numpy_version = r["version"]
        if not r["ok"]:
            if r["hint"]:
                print(f"  Öneri: {r['hint']}")
            print(indent(r["error"], 4))

    # requirements.txt içindeki numpy kısıtını kontrol et
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if numpy_version and os.path.isfile(req_path):
        try:
            with open(req_path, 'r', encoding='utf-8') as fh:
                lines = fh.readlines()
            for line in lines:
                if line.strip().startswith('numpy') and '<2' in line:
                    major = str(numpy_version).split('.')[0]
                    if major and major.isdigit() and int(major) >= 2:
                        print(f"\n⚠️  UYARI: Installed NumPy {numpy_version} requirements.txt içindeki '<2' kısıtını ihlal ediyor. 'pip install \"numpy<2\" --force-reinstall' önerilir.")
                    break
        except OSError:
            pass

    print_section("Opsiyonel Modüller")
    for name in optional:
        r = try_import(name)
        status = "✅" if r["ok"] else "❌"
        version = f" - {r['version']}" if r["ok"] and r["version"] else ""
        print(f"{status} {name}{version}")
        if not r["ok"]:
            if r["hint"]:
                print(f"  Öneri: {r['hint']}")
            print(indent(r["error"], 4))
            # NumPy 2.x uyumsuzluk özel uyarısı
            if name == "onnxruntime" and numpy_version and str(numpy_version).startswith("2"):
                print("  Ek Uyarı: onnxruntime + NumPy 2.x ikili uyumsuzluğu. Çözüm: 'pip install numpy<2' veya 'pip install --upgrade onnxruntime' (NumPy 2 desteği olan sürüme).")

    diagnose_desk_gui()
    print("\nTeşhis tamamlandı. Sorun devam ederse önerilen adımlar:"
          "\n  1) Yeni sanal ortam: python -m venv venv && venv\\Scripts\\activate"
          "\n  2) pip install --upgrade pip wheel setuptools"
          "\n  3) Minimum paketler: pip install PyQt5 opencv-python numpy requests pypubsub face_recognition cvzone mediapipe"
          "\n  4) Ardından diğer opsiyonelleri ekleyin.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

