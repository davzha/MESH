#%%
import numpy as np
import json
import argparse
from pprint import pprint
import pycocotools._mask as _mask
import cv2
from pathlib import Path
import h5py
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from math import ceil
from einops import rearrange, repeat


ROOT = Path("~/data/object_discovery/clevrtex_full").expanduser()
RESOLUTION = (64, 64)
CENTER_CROP = True
json_files = list(ROOT.glob("**/*.json"))
OUTPUT_FILE = Path(f"~/data/object_discovery/clevrtex{RESOLUTION[0]}.h5").expanduser()
NUM_EXAMPLES = len(json_files)
MAX_OBJECTS = 10
SKIP_BACKGROUND = True

db = h5py.File(OUTPUT_FILE, 'w')
# num_examples, num_frames, num_channels, height, width
db.create_dataset('images', (NUM_EXAMPLES, 3, *RESOLUTION), dtype=np.uint8)
# num_examples, num_frames, num_channels, height, width
# db.create_dataset('depth', (NUM_EXAMPLES, 3, *RESOLUTION), dtype=np.uint8)
# num_examples, num_frames, num_objects, num_channels, height, width
db.create_dataset('masks', (NUM_EXAMPLES, MAX_OBJECTS, 1, *RESOLUTION), dtype=np.uint8)
# num_examples, num_frames
db.create_dataset('num_objects', (NUM_EXAMPLES,), dtype=np.uint8)
# db.create_dataset('num_examples', (1,), dtype=np.long)
# db.create_dataset('attributes', (NUM_EXAMPLES, MAX_OBJECTS, 3), dtype=np.uint8)

for j in tqdm(json_files):
    with open(j, 'r') as f:
        data = json.load(f)
    idx = data['image_index']
    db['num_objects'][idx] = data['num_objects']

    img = cv2.imread(str(j.parent / data['image_filename']))
    mask = cv2.imread(str(j.parent / data['mask_filename']))
    
    if CENTER_CROP:
        height = img.shape[0]
        width = img.shape[1]
        crop_size = int(0.8 * float(min(width, height)))
        img = img[(height - crop_size) // 2:(height + crop_size) // 2, 
                    (width - crop_size) // 2:(width + crop_size) // 2]
        mask = mask[(height - crop_size) // 2:(height + crop_size) // 2, 
                    (width - crop_size) // 2:(width + crop_size) // 2]

    # save image
    img = cv2.resize(img, RESOLUTION, interpolation=cv2.INTER_LINEAR)
    db['images'][idx] = img.transpose(2, 0, 1) # BGR -> RGB

    # save masks
    mask = cv2.resize(mask, RESOLUTION, interpolation=cv2.INTER_NEAREST)
    object_ids = np.unique(mask.reshape(-1,3), axis=0)
    object_masks = (
        repeat(object_ids, 'n c -> n 1 1 c') == 
        repeat(mask, 'h w c -> 1 h w c')).all(-1)
    if SKIP_BACKGROUND:
        object_masks = object_masks[1:]
    db['masks'][idx, :len(object_masks), 0] = object_masks.astype(np.uint8)

    # save depth
    # depth = cv2.imread(str(j.parent / data['depth_filename']))
    # depth = cv2.resize(depth, RESOLUTION, interpolation=cv2.INTER_LINEAR)
    # db['depth'][idx] = depth.transpose(2, 0, 1)  # doesn't matter since it's grayscale

db.close()

# %%
