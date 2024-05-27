import os
import pydicom
from PIL import Image

def convert_dcm_to_jpg(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".DCM") or file.endswith(".dcm"):
                print("Converting", file)
                dcm_path = os.path.join(root, file)
                dcm_image = pydicom.dcmread(dcm_path)
                jpg_path = os.path.splitext(dcm_path)[0] + ".jpg"
                jpg_image = Image.fromarray(dcm_image.pixel_array)
                jpg_image.save(jpg_path)

# Usage example
folder_path = "/disk8t/jialiangfan/trained_models/dataset/medical_data/2018/信 729047/1"
convert_dcm_to_jpg(folder_path)
