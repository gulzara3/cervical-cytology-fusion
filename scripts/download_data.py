"""Download the public SIPaKMeD, Herlev, and UCI clinical datasets.

Run once before training:  python scripts/download_data.py
All datasets are public; source URLs are listed in the paper's Data Availability.
"""
import os, subprocess, urllib.request, zipfile, glob, shutil
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ---- SIPaKMeD ----
print("="*70)
print("� DOWNLOADING SIPaKMeD DATASET")
print("   Source: University of Ioannina, Greece")
print("   URL: https://www.cs.uoi.gr/~marina/sipakmed.html")
print("="*70)

SIPAKMED_BASE = "https://www.cs.uoi.gr/~marina/SIPAKMED"
SIPAKMED_FILES = [
    ("im_Superficial-Intermediate.7z", "Superficial-Intermediate", 0),
    ("im_Parabasal.7z", "Parabasal", 0),
    ("im_Metaplastic.7z", "Metaplastic", 0),
    ("im_Koilocytotic.7z", "Koilocytotic", 1),
    ("im_Dyskeratotic.7z", "Dyskeratotic", 1),
]

sipakmed_paths = []
sipakmed_labels = []

for filename, classname, label in SIPAKMED_FILES:
    url = f"{SIPAKMED_BASE}/{filename}"
    local_7z = f"data/sipakmed/{filename}"
    extract_dir = f"data/sipakmed/{classname}"

    existing_images = glob.glob(f"{extract_dir}/**/*.bmp", recursive=True) + \
                      glob.glob(f"{extract_dir}/**/*.BMP", recursive=True)

    if len(existing_images) > 0:
        print(f"    {classname}: {len(existing_images)} images (already exists)")
    else:
        print(f"   � Downloading {classname}...")
        try:
            urllib.request.urlretrieve(url, local_7z)
            print(f"      Downloaded: {os.path.getsize(local_7z)/1e6:.1f} MB")
            os.makedirs(extract_dir, exist_ok=True)
            result = subprocess.run(["7z", "x", "-y", f"-o{extract_dir}", local_7z],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                print(f"       Extracted")
                os.remove(local_7z)
            else:
                print(f"       Extraction issue")
        except Exception as e:
            print(f"       Error: {e}")

    for ext in ['*.bmp', '*.BMP', '*.png', '*.jpg']:
        for img_path in glob.glob(f"{extract_dir}/**/{ext}", recursive=True):
            sipakmed_paths.append(img_path)
            sipakmed_labels.append(label)

print(f"\n SIPaKMeD Total: {len(sipakmed_paths)} images")
if len(sipakmed_paths) > 0:
    print(f"   Normal (0): {sipakmed_labels.count(0)}")
    print(f"   Abnormal (1): {sipakmed_labels.count(1)}")

# ---- Herlev ----
print("\n" + "="*70)
print("� DOWNLOADING HERLEV DATASET")
print("   Source: MDE Lab, University of the Aegean / DTU")
print("="*70)

HERLEV_URL = "http://mde-lab.aegean.gr/images/stories/docs/smear2005.zip"
local_zip = "data/herlev/smear2005.zip"

# Check if already extracted - FIXED: search all subdirectories
existing_herlev = []
for ext in ['*.bmp', '*.BMP', '*.png', '*.jpg', '*.JPG']:
    existing_herlev.extend(glob.glob(f"data/herlev/**/{ext}", recursive=True))

if len(existing_herlev) > 100:
    print(f"    Already exists: {len(existing_herlev)} images")
else:
    print("   � Downloading smear2005.zip (85 MB)...")
    try:
        # Try wget first (more reliable)
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", local_zip, HERLEV_URL],
            capture_output=False
        )
        if not os.path.exists(local_zip) or os.path.getsize(local_zip) < 1000000:
            urllib.request.urlretrieve(HERLEV_URL, local_zip)
        print(f"      Downloaded: {os.path.getsize(local_zip)/1e6:.1f} MB")
    except Exception as e:
        print(f"      Trying urllib: {e}")
        urllib.request.urlretrieve(HERLEV_URL, local_zip)

    # Extract
    print("   � Extracting...")
    try:
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall("data/herlev")
        print("       Extracted!")
    except Exception as e:
        print(f"      Trying unzip command: {e}")
        subprocess.run(["unzip", "-o", "-q", local_zip, "-d", "data/herlev"])

# FIXED: Comprehensive path detection for Herlev
# Herlev has 7 classes in folders, we map to binary
HERLEV_CLASS_MAP = {
    # Normal classes (0)
    'superficial': 0, 'intermediate': 0, 'columnar': 0,
    'normal': 0, 'superficiel': 0,
    # Abnormal classes (1)
    'mild': 1, 'moderate': 1, 'severe': 1, 'carcinoma': 1,
    'dysplasia': 1, 'light': 1, 'cancer': 1, 'situ': 1
}

herlev_paths = []
herlev_labels = []

# Search all possible image locations
print("\n   � Scanning for Herlev images...")
all_herlev_images = []
for ext in ['*.bmp', '*.BMP', '*.png', '*.jpg', '*.JPG', '*.jpeg']:
    all_herlev_images.extend(glob.glob(f"data/herlev/**/{ext}", recursive=True))

print(f"      Found {len(all_herlev_images)} total image files")

# Classify each image based on path
for img_path in all_herlev_images:
    path_lower = img_path.lower()
    label = None

    for keyword, lbl in HERLEV_CLASS_MAP.items():
        if keyword in path_lower:
            label = lbl
            break

    if label is not None:
        herlev_paths.append(img_path)
        herlev_labels.append(label)

print(f"\n Herlev Total: {len(herlev_paths)} images")
if len(herlev_paths) > 0:
    print(f"   Normal (0): {herlev_labels.count(0)}")
    print(f"   Abnormal (1): {herlev_labels.count(1)}")
else:
    # Debug: show directory structure
    print("    No labeled images found. Directory structure:")
    for root, dirs, files in os.walk("data/herlev"):
        level = root.replace("data/herlev", "").count(os.sep)
        indent = "   " * (level + 1)
        print(f"{indent}{os.path.basename(root)}/")
        if level < 2:
            for f in files[:3]:
                print(f"{indent}  {f}")
            if len(files) > 3:
                print(f"{indent}  ... and {len(files)-3} more")

# ---- UCI Clinical ----
print("\n" + "="*70)
print("� DOWNLOADING UCI CLINICAL DATA")
print("="*70)

clinical_path = "data/clinical/cervical_cancer.csv"

if os.path.exists(clinical_path):
    print("    Already exists!")
else:
    UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"
    try:
        urllib.request.urlretrieve(UCI_URL, clinical_path)
        print("    Downloaded!")
    except Exception as e:
        print(f"    UCI download failed: {e}")

if os.path.exists(clinical_path):
    clinical_df = pd.read_csv(clinical_path)
    print(f"    Shape: {clinical_df.shape}")
