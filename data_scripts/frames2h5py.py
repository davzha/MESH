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

def decode(rleObjs):
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:,:,0]

COLORS = {
    'cyan': 1,
    'gray': 2,
    'brown': 3,
    'green': 4,
    'blue': 5,
    'yellow': 6,
    'red': 7,
    'purple': 8,
}

MATERIALS = {
    'metal': 1,
    'rubber': 2,
}

SHAPES = {
    'sphere': 1,
    'cylinder': 2,
    'cube': 3,
}

ROOT_MASKS = Path("~/data/object_discovery/CLEVERER/processed_proposals").expanduser()
ROOT_FRAMES = Path("~/data/object_discovery/CLEVERER/video_frames").expanduser()

RESOLUTION = (64, 64)
KEEP_FRAME_IDS = [0, 16]
OUTPUT_FILE = Path(f"~/data/object_discovery/clevrer_f{KEEP_FRAME_IDS[0]}_f{KEEP_FRAME_IDS[1]}.h5").expanduser()


#%%
db = h5py.File(OUTPUT_FILE, 'w')
# num_examples, num_frames, num_channels, height, width
db.create_dataset('images', (20000, len(KEEP_FRAME_IDS), 3, 64, 64), dtype=np.uint8)
# num_examples, num_frames, num_objects, num_channels, height, width
db.create_dataset('masks', (20000, len(KEEP_FRAME_IDS), 7, 1, 64, 64), dtype=np.uint8)
# num_examples, num_frames
db.create_dataset('num_objects', (20000, len(KEEP_FRAME_IDS)), dtype=np.uint8)
db.create_dataset('num_examples', (1,), dtype=np.long)
db.create_dataset('attributes', (20000, len(KEEP_FRAME_IDS), 7, 3), dtype=np.uint8)

db['num_examples'][0] = 20000

for video_id in tqdm(range(20000)):
    with (ROOT_MASKS / f"sim_{video_id:05d}.json").open("r") as f:
        data = json.load(f)

    if len(data['frames']) < 128:
        print(f"Skipping video {video_id} with {len(data['frames'])} frames")
        # 7648: 115 frames in json
        # 9777: 104 frames in json
        # 10800: 114 frames in json
        continue

    keep_frames = [data['frames'][id] for id in KEEP_FRAME_IDS]
    for frame in keep_frames:
        frame_id = frame['frame_index']
        frame_file = ROOT_FRAMES / f"video_{video_id:05d}" / f"{frame_id}.png"

        masks = []
        db['num_objects'][video_id, KEEP_FRAME_IDS.index(frame_id)] = len(frame['objects'])
        img = cv2.imread(str(frame_file))
        img = cv2.resize(img, RESOLUTION, interpolation=cv2.INTER_LINEAR)
        db['images'][video_id, KEEP_FRAME_IDS.index(frame_id)] = img.transpose(2, 0, 1) # BGR -> RGB
        
        masks = []
        attributes = []
        for obj_id, obj in enumerate(frame['objects']):
            try:
                mask = decode(obj['mask'])
                mask = cv2.resize(mask, RESOLUTION, interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                print(f"Error reading mask for video {video_id} frame {frame_id} object {obj_id}")
            masks.append(mask)

            attributes.append(np.array(
                (COLORS[obj['color']], 
                MATERIALS[obj['material']], 
                SHAPES[obj['shape']])))
                
        masks = np.stack(masks, axis=0)
        db['masks'][video_id, KEEP_FRAME_IDS.index(frame_id), :masks.shape[0], 0] = masks

        attributes = np.stack(attributes, axis=0)
        db['attributes'][video_id, KEEP_FRAME_IDS.index(frame_id), :attributes.shape[0]] = attributes

db.close()

def plot_video(video_id, subsample=2, root=ROOT_FRAMES):
    frames = list((root / f"video_{video_id:05d}").glob("*.png"))
    frames = sorted(frames, key=lambda x: int(x.stem))
    frames = frames[::subsample]

    fig, axes = plt.subplots(ceil(len(frames) / 4), 4, figsize=(10, 6))
    for frame, ax in zip(frames, axes.flatten()):
        ax.imshow(cv2.imread(str(frame)))
        ax.axis('off')
    fig.tight_layout()
    plt.show()
