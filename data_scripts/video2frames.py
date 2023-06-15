import cv2
from pathlib import Path
from tqdm import tqdm

# directory containing the mp4 files
dir_in = Path('~/data/object_discovery/CLEVERER').expanduser()
# directory to save the png frames
dir_out = Path('~/data/object_discovery/CLEVERER/video_frames').expanduser()

# create the output directory if it doesn't exist
dir_out.mkdir(parents=True, exist_ok=True)

# loop through all the mp4 files in the input directory
for file in tqdm(Path(dir_in).glob('**/*.mp4')):
    # open the video file
    video = cv2.VideoCapture(str(file))

    # create a directory for the frames of this video
    video_dir = Path(dir_out) / file.stem
    video_dir.mkdir(parents=True, exist_ok=True)

    # loop through all the frames in the video
    frame_count = 0
    success = True
    while success:
        # read the next frame
        success, frame = video.read()

        if success:
            # save the frame to a file
            (video_dir / f'{frame_count}.png').write_bytes(cv2.imencode('.png', frame)[1])
            frame_count += 1

    # close the video file
    video.release()
