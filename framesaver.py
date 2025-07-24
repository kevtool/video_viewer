import subprocess
import numpy as np
import cv2
import re
import yaml, json
import os
import imageio.v3 as iio
from pathlib import Path

class Y4MVideoReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.width, self.height, self.fps = self._parse_y4m_header()
        self.frame_size = self.width * self.height * 3  # RGB format
        self.pipe = self._start_ffmpeg_process()

    def _parse_y4m_header(self):
        with open(self.file_path, 'rb') as f:
            header = ''
            while not header.endswith('\n'):
                byte = f.read(1).decode('utf-8', errors='ignore')
                if not byte:
                    raise ValueError("Could not read Y4M header")
                header += byte

            match = re.match(r'YUV4MPEG2 W(\d+) H(\d+) F(\d+):(\d+) .*$', header)
            if not match:
                raise ValueError("Invalid Y4M header")

            width = int(match.group(1))
            height = int(match.group(2))
            fps_numerator = int(match.group(3))
            fps_denominator = int(match.group(4))
            fps = fps_numerator / fps_denominator if fps_denominator else 30

            return width, height, fps

    def _start_ffmpeg_process(self, start_time_sec=None):
        command = [
            'ffmpeg',
            '-i', self.file_path,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo',
            '-'
        ]
        if start_time_sec is not None:
            command.insert(1, '-ss')
            command.insert(2, str(start_time_sec))
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def read_frames(self):
        """Generator that yields each frame as a NumPy array (BGR)"""
        while True:
            raw_frame = self.pipe.stdout.read(self.frame_size)
            if len(raw_frame) != self.frame_size:
                break
            yield self._process_raw_frame(raw_frame)

    def _process_raw_frame(self, raw_frame):
        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 3)).copy()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    def get_frame_count(self):
        """
        Returns the total number of frames in the Y4M video using ffprobe.
        """
        import subprocess
        import re

        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 'stream=nb_read_packets',
            '-of', 'csv=p=0',
            self.file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        output = result.stdout.strip()
        if not output or not re.match(r'^\d+$', output):
            # Fallback: count 'FRAME' headers manually
            return self._count_frames_fallback()

        return int(output)

    def play(self, resize_factor=0.3, start_frame_num=0):
        try:
            # Recalculate start time in seconds
            start_time = start_frame_num / self.fps if self.fps > 0 else 0

            # Restart ffmpeg process from desired time
            self.pipe = self._start_ffmpeg_process(start_time)

            # Re-initialize the frame reader using the new pipe
            frames = self.read_frames()

            # Re-read frames now starting from the new offset
            for idx, frame in enumerate(frames, start=start_frame_num + 1):
                cv2.putText(frame, f"Frame: {idx}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                resized = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
                cv2.imshow('Y4M Player', resized)
                key = cv2.waitKey(int(1000 / self.fps)) & 0xFF
                if key == ord('q'):
                    break
        finally:
            self.close()
            cv2.destroyAllWindows()

    def extract_roi(self, frame_num, roi_rect, zoom_factor=2, save_path=None, show=True):
        x, y, w, h = roi_rect
        try:
            # Calculate time offset
            start_time = frame_num / self.fps if self.fps > 0 else 0

            # Restart ffmpeg with seek
            self.pipe = self._start_ffmpeg_process(start_time)

            # Get the first frame after seeking
            frames = self.read_frames()
            for idx, frame in enumerate(frames, start=frame_num):  # assume we get frame_num directly
                # Display ROI
                roi = frame[y:y+h, x:x+w]

                if show:
                    resized = cv2.resize(roi, None, fx=zoom_factor, fy=zoom_factor)
                    cv2.imshow(f"ROI - Frame {frame_num}", resized)
                    print(f"Displayed ROI of frame {frame_num}")
                    cv2.waitKey(0)

                if save_path:
                    cv2.imwrite(save_path, roi)
                    print(f"Saved ROI frame {frame_num} to: {save_path}")
                
                break
        finally:
            self.close()
            cv2.destroyAllWindows()

    def visualize_roi_on_frame(self, frame_num, roi_rect, save_path=None, show=True):
        x, y, w, h = roi_rect
        try:
            # Calculate time offset for seeking
            start_time = frame_num / self.fps if self.fps > 0 else 0

            # Restart ffmpeg with seek to target frame
            self.pipe = self._start_ffmpeg_process(start_time)

            # Read the first frame after seeking
            frames = self.read_frames()
            for idx, frame in enumerate(frames, start=frame_num):
                # Draw ROI rectangle on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box, thickness 2
                cv2.putText(frame, f"Frame: {idx}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if show:
                    cv2.imshow(f"Full Frame with ROI - Frame {frame_num}", frame)
                    print(f"Displayed full frame with ROI boundary for frame {frame_num}")
                    cv2.waitKey(0)  # Wait until keypress

                if save_path:
                    cv2.imwrite(save_path, frame)
                    print(f"Saved full frame with ROI to: {save_path}")

                break  # Only process one frame
        finally:
            self.close()
            cv2.destroyAllWindows()

    def close(self):
        self.pipe.stdout.close()
        self.pipe.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def yml_save_frames(yml_file):
    with open(yml_file, 'r') as file:
        frames = yaml.safe_load(file)

        for frame in frames:
            video_path = "./videos/" + frame['video'] + ".y4m"

            if os.path.isfile(video_path):
                id, frame_num, x, y, w, h = frame['id'], frame['frame_number'], frame['roi_left'], frame['roi_top'], frame['roi_width'], frame['roi_height']

                with Y4MVideoReader(video_path) as reader:
                    reader.extract_roi(frame_num=frame_num, roi_rect=(x, y, w, h), save_path=f"dataset/motion_blur/{id}.png", show=False)

def jsonl_save_frames(category):
    jsonl_file = f'dataset_files/{category}.jsonl'

    with open(jsonl_file, 'r') as file:
        for id, line in enumerate(file):
            frame = json.loads(line)
            video_path = "./videos/" + frame['video'] + ".y4m"

            if os.path.isfile(video_path):
                frame_num, x, y, w, h = frame['frame_number'], frame['left'], frame['top'], frame['width'], frame['height']

                with Y4MVideoReader(video_path) as reader:
                    reader.extract_roi(frame_num=frame_num, roi_rect=(x, y, w, h), save_path=f"dataset/{category}/{id}.png", show=False)

def find_last_number(s):
    numbers = re.findall(r'\d+', s)  # Find all sequences of digits
    return int(numbers[-1]) if numbers else None  # Convert last to int

def images_to_y4m(folder_path, output_y4m, fps=30, width=1920, height=1080):
    fpath = Path(folder_path)
    image_paths = [f'{folder_path}/{f.name}' for f in fpath.iterdir() if f.is_file() and f.name.startswith('box') and f.name.endswith('png')]

    start_number = find_last_number(image_paths[0])
    print(image_paths)
    print(start_number)

    cmd = [
        'ffmpeg',
        '-start_number', str(start_number),
        '-r', str(fps),                   # Input frame rate
        '-i', f'{folder_path}/box%d.png',              # Input images
        '-f', 'rawvideo',                 # Output as raw video stream
        '-pix_fmt', 'yuv420p',            # Required pixel format
        '-s', f'{width}x{height}',        # Must specify resolution
        '-f', 'yuv4mpegpipe',             # Output format: YUV4MPEG
        output_y4m
    ]

    print(cmd)

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully created {output_y4m}")
    except subprocess.CalledProcessError as e:
        print("FFmpeg error:", e.stderr.decode())
        raise
    except FileNotFoundError:
        print("'ffmpeg' not found. Install ffmpeg and make sure it's in your PATH.")
        raise
        
def save_roi_contexts(category, neighboring_frames=5, force_regen=False, generate_video=False):
    jsonl_file = f'dataset_files/{category}.jsonl'

    with open(jsonl_file, 'r') as file:
        for id, line in enumerate(file):
            frame = json.loads(line)
            video_path = "./videos/" + frame['video'] + ".y4m"

            # execute only if current video is downloaded
            if os.path.isfile(video_path):
                frame_num, x, y, w, h = frame['frame_number'], frame['left'], frame['top'], frame['width'], frame['height']


                folder_path = f'dataset/{category}/{id}'
                with Y4MVideoReader(video_path) as reader:
                    start_frame = max(0, frame_num - neighboring_frames)
                    end_frame = min(reader.get_frame_count() - 1, frame_num + neighboring_frames + 1)

                    if (not os.path.exists(folder_path)) or force_regen:
                        if not os.path.exists(folder_path): os.makedirs(folder_path)
                    
                        for fnum in range(start_frame, end_frame):
                            reader.extract_roi(frame_num=fnum, roi_rect=(x, y, w, h), save_path=folder_path+f"/roi{fnum}.png", show=False)
                            reader.visualize_roi_on_frame(frame_num=fnum, roi_rect=(x, y, w, h), save_path=folder_path+f"/box{fnum}.png", show=False)

                        if generate_video:
                            # fpath = Path(folder_path)
                            # image_paths = [f'{folder_path}/{f.name}' for f in fpath.iterdir() if f.is_file() and f.name.startswith('box')]
                            # print(image_paths)
                            # frames = [iio.imread(path) for path in image_paths]

                            files_to_remove = Path(folder_path).glob(f'*x.y4m')
                            for file_path in files_to_remove:
                                if file_path.is_file():  # Ensure it's a file
                                    file_path.unlink()
                                    print(f"Removed: {file_path}")

                            video = f"dataset/{category}/{id}/box.y4m"
                            # iio.imwrite(
                            #     video,
                            #     frames,
                            #     extension=".y4m",
                            #     plugin="FFMPEG",
                            #     format="yuv4mpeg",
                            #     fps=5,
                            #     pixelformat="yuv420p"
                            # )
                            # print(f"Saved frames as video {video}")
                            images_to_y4m(folder_path, video)

                    else:
                        print(f'WARNING: {category} {id} folder already exists, skipping instance')

if __name__ == "__main__":
    file = "./videos/netflix_aerial.y4m"

    # play the video
    # with Y4MVideoReader(file) as reader:
    #     reader.play(resize_factor=0.3, start_frame_num=450)

    # entire frame, zoomed out
    # with Y4MVideoReader(file) as reader:
    #     reader.extract_roi(frame_num=483, roi_rect=(0, 0, 4096, 3072), zoom_factor=0.3)

    # roi, zoomed in
    # with Y4MVideoReader(file) as reader:
    #     reader.extract_roi(frame_num=483, roi_rect=(3300, 900, 796, 800), zoom_factor=1.1)


    # jsonl_save_frames(category='texture_loss_static')

    # with Y4MVideoReader(file) as reader:
    #     reader.visualize_roi_on_frame(frame_num=100, roi_rect=(3300, 900, 600, 600), save_path="test.png")
    save_roi_contexts(category='texture_loss_static', neighboring_frames=10, force_regen=True, generate_video=True)