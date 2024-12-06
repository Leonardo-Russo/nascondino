from pycocotools.coco import COCO
import requests
import os
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms

class COCOSegmentation(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.coco = COCO(annFile)
        self.root = root
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # Create segmentation mask
        mask = np.zeros((img_info['height'], img_info['width']))
        for i in range(len(anns)):
            print(f"mask: {mask}")
            print(f"anns[i]: {anns[i]}")
            print(f"anns[i]['category_id']: {anns[i]['category_id']}")
            mask = np.maximum(mask, coco.annToMask(anns[i]) * anns[i]['category_id'])  # Use category ID for pixel value

        if self.transform:
            img = self.transform(img)
            mask_transform = transforms.Compose([
                transforms.Resize((img.shape[-1], img.shape[-2]), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
            mask = mask_transform(Image.fromarray(mask))

        return img, mask

    def __len__(self):
        return len(self.ids)
    
def visualize_segmentation(image, mask):
    """
    Visualizes the segmentation mask as an overlay on the original image.

    Args:
      image: The original image as a PyTorch tensor (C, H, W).
      mask: The segmentation mask as a PyTorch tensor (C, H, W).
    """

    # Convert image and mask to NumPy arrays
    image_np = image.permute(1, 2, 0).cpu().detach().numpy()
    mask_np = mask.squeeze(0).cpu().detach().numpy().squeeze()

    # Get the unique class IDs in the mask (excluding background, usually 0)
    class_ids = np.unique(mask_np)
    class_ids = class_ids[class_ids != 0]

    # Create a colormap with a distinct color for each class
    cmap = cm.get_cmap('viridis', len(class_ids))  # 'viridis' or any other colormap

    # Create a colored mask
    colored_mask = np.zeros_like(image_np)
    for i, class_id in enumerate(class_ids):
        colored_mask[mask_np == class_id] = cmap(i)[:3]  # Take only RGB values

    # Overlay the colored mask on the original image with some transparency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    ax1.imshow(image_np)
    ax1.set_title("Original Image")
    ax1.axis('off')
    # ax2.imshow(image_np)
    ax2.imshow(colored_mask, alpha=1)
    ax2.set_title("Segmentation Mask")
    ax2.axis('off')
    plt.show()

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


def collate_fn(batch):
    """Custom collate function to handle variable-sized annotations."""
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)  # Append annotations as they are (list of dicts)
    return torch.stack(images), targets