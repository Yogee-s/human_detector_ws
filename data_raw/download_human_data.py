import os
import json
import requests
import random
from tqdm import tqdm
from pycocotools.coco import COCO
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from shutil import copyfile

# Set random seed for reproducibility
random.seed(42)

# Define paths
BASE_DIR = '../data_training/data_person'
IMG_DIR_TRAIN = os.path.join(BASE_DIR, 'images/train')
IMG_DIR_VAL = os.path.join(BASE_DIR, 'images/val')
LBL_DIR_TRAIN = os.path.join(BASE_DIR, 'labels/train')
LBL_DIR_VAL = os.path.join(BASE_DIR, 'labels/val')

for d in [IMG_DIR_TRAIN, IMG_DIR_VAL, LBL_DIR_TRAIN, LBL_DIR_VAL]:
    os.makedirs(d, exist_ok=True)

# Load COCO
coco = COCO('../data_training/data_person/annotations/instances_train2017.json')
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)

# Split train/val
random.shuffle(images)
split_idx = int(0.9 * len(images))
train_images = images[:split_idx]
val_images = images[split_idx:]

# Retry session for downloads
session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Helper to convert COCO bbox to YOLO
def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    return [x_center, y_center, w / img_w, h / img_h]

# Process dataset
def process_dataset(images, img_dir, lbl_dir):
    for img in tqdm(images, desc=f"Processing {img_dir}"):
        file_name = img['file_name']
        img_path = os.path.join(img_dir, file_name)
        label_path = os.path.join(lbl_dir, file_name.replace('.jpg', '.txt'))

        # Download image if not exists
        if not os.path.isfile(img_path):
            try:
                img_data = session.get(img['coco_url']).content
                with open(img_path, 'wb') as f:
                    f.write(img_data)
            except Exception as e:
                print(f"Failed to download {file_name}: {e}")
                continue

        # Get annotations
        annIds = coco.getAnnIds(imgIds=[img['id']], catIds=catIds, iscrowd=False)
        anns = coco.loadAnns(annIds)

        if not anns:
            continue

        with open(label_path, 'w') as f:
            for ann in anns:
                bbox = coco_to_yolo_bbox(ann['bbox'], img['width'], img['height'])
                f.write(f"0 {' '.join([f'{x:.6f}' for x in bbox])}\n")

# Process both sets
process_dataset(train_images, IMG_DIR_TRAIN, LBL_DIR_TRAIN)
process_dataset(val_images, IMG_DIR_VAL, LBL_DIR_VAL)
