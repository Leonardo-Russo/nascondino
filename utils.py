from pycocotools.coco import COCO
import requests
from PIL import ImageFile
import os
import zipfile

def download_and_extract(coco_path, url, target_dir):
    """Downloads a zip file and extracts its contents."""
    filename = url.split("/")[-1]
    filepath = os.path.join(coco_path, filename)

    if not os.path.exists(target_dir):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(coco_path)
        os.remove(filepath)

def download_coco_dataset(coco_path, op_sys="windows"):

    if op_sys == "windows" or op_sys == "shit":

        base_url = "http://images.cocodataset.org/zips/"
        annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

        # Download and extract train2017 images
        if not os.path.exists(f"{coco_path}/train2017"):
            download_and_extract(coco_path, f"{base_url}train2017.zip", f"{coco_path}/train2017")

        # Download and extract val2017 images
        if not os.path.exists(f"{coco_path}/val2017"):
            download_and_extract(coco_path, f"{base_url}val2017.zip", f"{coco_path}/val2017")

        # Download and extract annotations
        if not os.path.exists(f"{coco_path}/annotations"):
            download_and_extract(coco_path, annotations_url, f"{coco_path}/annotations")

    elif op_sys == "unix":
        
        base_url = "http://images.cocodataset.org/zips/"
        annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

        # Download and extract train2017 images
        if not os.path.exists(f"{coco_path}/train2017"):
            print("Downloading train2017 images...")
            os.system(f"wget {base_url}train2017.zip -P {coco_path}")
            os.system(f"unzip -q {coco_path}/train2017.zip -d {coco_path}")

        # Download and extract val2017 images
        if not os.path.exists(f"{coco_path}/val2017"):
            print("Downloading val2017 images...")
            os.system(f"wget {base_url}val2017.zip -P {coco_path}")
            os.system(f"unzip -q {coco_path}/val2017.zip -d {coco_path}")

        # Download and extract annotations
        if not os.path.exists(f"{coco_path}/annotations"):
            print("Downloading annotations...")
            os.system(f"wget {annotations_url} -P {coco_path}")
            os.system(f"unzip -q {coco_path}/annotations_trainval2017.zip -d {coco_path}")

    