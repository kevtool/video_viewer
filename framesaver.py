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
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")

        self.width, self.height, self.fps = self._get_video_info()  # Uses ffprobe
        self.frame_size = self.width * self.height * 3
        self.pipe = None  # Delayed creation

    def _get_video_info(self):
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate',
            '-of', 'csv=p=0', self.file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout.strip()

        width_str, height_str, fps_ratio = output.split(',')
        width = int(width_str)
        height = int(height_str)

        # Parse "30000/1000" → 30.0
        if '/' in fps_ratio:
            num, den = map(int, fps_ratio.split('/'))
            fps = num / den if den else 30.0
        else:
            fps = float(fps_ratio)

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

    def read_frames(self, start_frame_num=0):
        start_time_sec = start_frame_num / self.fps if self.fps > 0 else 0
        self.pipe = self._start_ffmpeg_process(start_time_sec)

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
        Returns the total number of frames in the video using ffprobe.
        
        This method first tries to count the number of video packets using ffprobe,
        which gives an accurate frame count for most formats.
        
        If that fails (e.g., due to format limitations), it falls back to estimating
        the frame count using the video's duration and frame rate.
        
        Returns:
            int: Total number of frames in the video.
        
        Raises:
            ValueError: If neither packet count nor duration can be retrieved.
        """
        # Step 1: Try to get exact frame count using packet counting
        # This is the most accurate method when supported by the container.
        cmd_packet_count = [
            'ffprobe',
            '-v', 'error',                           # Suppress verbose output
            '-select_streams', 'v:0',                # Select the first video stream
            '-count_packets',                        # Count all packets in the stream
            '-show_entries', 'stream=nb_read_packets',  # Output only the packet count
            '-of', 'csv=p=0',                        # Output as raw value (no headers)
            self.file_path                           # Input file path
        ]

        result = subprocess.run(
            cmd_packet_count,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Return strings, not bytes
        )

        output = result.stdout.strip()

        # Check if output is a valid integer string
        if output.isdigit():
            return int(output)

        # If packet count failed, log warning and fall back to duration-based estimate
        print("Warning: Could not get exact frame count via packet counting. "
            "Falling back to duration-based estimation.")

        # Step 2: Fallback — Estimate frame count using duration and FPS
        cmd_duration = [
            'ffprobe',
            '-v', 'error',                   # Suppress unnecessary output
            '-show_entries', 'format=duration',  # Get the total duration
            '-of', 'csv=p=0',               # Output only the value
            self.file_path                  # Input file
        ]

        result = subprocess.run(
            cmd_duration,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        duration_output = result.stdout.strip()

        try:
            duration = float(duration_output)
            estimated_frame_count = int(duration * self.fps)
            return estimated_frame_count
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Could not determine video duration or calculate frame count for '{self.file_path}'. "
                "Ensure the file is a valid video and ffprobe is installed."
            ) from e

    def play(self, resize_factor=0.3, start_frame_num=0):
        """
        Play the video starting from a specified frame number.
        
        This method uses ffmpeg to decode frames starting from the approximate time
        corresponding to the given frame number. It displays each frame using OpenCV,
        with a frame counter overlaid. Playback speed matches the original FPS.
        
        Press 'q' to quit playback early.
        
        Args:
            resize_factor (float): Scaling factor to resize the video window (e.g., 0.3 = 30% size).
            start_frame_num (int): Frame number to start playback from (0-indexed).
        
        Raises:
            RuntimeError: If video cannot be decoded or display fails.
        """
        # Validate input
        if resize_factor <= 0:
            raise ValueError("resize_factor must be greater than 0")
        if start_frame_num < 0:
            raise ValueError("start_frame_num must be non-negative")

        try:
            # Calculate start time in seconds based on frame number and FPS
            if self.fps > 0:
                start_time_sec = start_frame_num / self.fps
            else:
                start_time_sec = 0.0  # Default to beginning if FPS is invalid

            # OpenCV window for playback
            cv2.namedWindow('Video Player', cv2.WINDOW_AUTOSIZE)

            # Use the improved read_frames method which handles seeking internally
            frame_generator = self.read_frames(start_frame_num=start_frame_num)

            # Start playing frames
            for frame_index, frame in enumerate(frame_generator, start=start_frame_num + 1):
                # Add frame counter text overlay
                cv2.putText(
                    img=frame,
                    text=f"Frame: {frame_index}",
                    org=(10, 30),  # Position: (x, y)
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 255, 0),  # Green
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

                # Resize frame for display (optional, for smaller windows)
                resized_frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)

                # Show the frame
                cv2.imshow('Video Player', resized_frame)

                # Wait approximately the correct time between frames (in milliseconds)
                delay_ms = int(1000 / self.fps)  # E.g., 33ms for 30fps
                if delay_ms < 1:
                    delay_ms = 1  # Avoid zero or negative delay

                # Wait for keypress; check if 'q' was pressed to exit
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord('q'):
                    print("Playback stopped by user (pressed 'q').")
                    break

        except Exception as e:
            raise RuntimeError(f"Error during video playback: {e}") from e

        finally:
            # Ensure resources are cleaned up even if an error occurs
            self.close()
            cv2.destroyAllWindows()

    def extract_roi(self, frame_num, roi_rect, zoom_factor=2, save_path=None, show=True):
        """
        Extract and optionally display or save a Region of Interest (ROI) from a specific frame.
        
        Args:
            frame_num (int): The frame number (0-indexed) to extract.
            roi_rect (tuple): Region of interest as (x, y, width, height).
            zoom_factor (float): Zoom multiplier for displaying the ROI (e.g., 2 = 2x size).
            save_path (str or None): If provided, save the ROI image to this path.
            show (bool): If True, display the ROI in a window until a key is pressed.
        
        Raises:
            ValueError: If ROI is out of bounds or invalid.
            RuntimeError: If frame could not be read.
        """
        x, y, w, h = roi_rect

        # Input validation
        if not all(isinstance(v, int) and v >= 0 for v in [x, y, w, h]):
            raise ValueError("ROI coordinates and dimensions must be non-negative integers")
        if x + w > self.width or y + h > self.height:
            raise ValueError(f"ROI {roi_rect} exceeds video resolution {self.width}x{self.height}")
        if zoom_factor <= 0:
            raise ValueError("zoom_factor must be positive")

        try:
            # Calculate start time for seeking
            start_time_sec = frame_num / self.fps if self.fps > 0 else 0.0

            # Use read_frames with seeking to jump close to target frame
            frame_generator = self.read_frames(start_frame_num=frame_num)

            # Read only the first frame (should be the target frame or very close)
            try:
                frame = next(frame_generator)
            except StopIteration:
                raise RuntimeError(f"Could not read frame {frame_num}. Possibly out of range.")

            # Extract the ROI (region of interest)
            roi = frame[y:y+h, x:x+w].copy()  # Use .copy() to ensure contiguous array

            # Optionally display the ROI (zoomed in)
            if show:
                zoomed_width = int(w * zoom_factor)
                zoomed_height = int(h * zoom_factor)
                resized_roi = cv2.resize(roi, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)
                window_title = f"ROI - Frame {frame_num}"
                cv2.imshow(window_title, resized_roi)
                print(f"Displayed ROI of frame {frame_num}: {w}x{h} pixels (zoomed {zoom_factor}x)")
                cv2.waitKey(0)  # Wait until any key is pressed
                cv2.destroyWindow(window_title)

            # Optionally save ROI to disk
            if save_path is not None:
                try:
                    success = cv2.imwrite(save_path, roi)
                    if success:
                        print(f"Saved ROI from frame {frame_num} to: {save_path}")
                    else:
                        raise RuntimeError(f"Failed to save image to {save_path}")
                except Exception as e:
                    raise RuntimeError(f"Error saving ROI image: {e}") from e

        except Exception as e:
            raise RuntimeError(f"Error in extract_roi: {e}") from e

        finally:
            # Always clean up
            self.close()
            if not show:
                cv2.destroyAllWindows()

    def visualize_roi_on_frame(self, frame_num, roi_rect, save_path=None, show=True):
        """
        Display the full frame with a rectangle drawn around the specified Region of Interest (ROI).
        
        This helps visualize where the ROI is located in the context of the entire frame.
        
        Args:
            frame_num (int): The frame number (0-indexed) to visualize.
            roi_rect (tuple): Region of interest as (x, y, width, height).
            save_path (str or None): If provided, save the annotated frame to this path.
            show (bool): If True, display the annotated frame in a window.
        
        Raises:
            ValueError: If ROI is invalid or out of bounds.
            RuntimeError: If frame cannot be read or saved.
        """
        x, y, w, h = roi_rect

        # Input validation
        if not all(isinstance(v, int) and v >= 0 for v in [x, y, w, h]):
            raise ValueError("ROI coordinates and dimensions must be non-negative integers")
        if x + w > self.width or y + h > self.height:
            raise ValueError(f"ROI {roi_rect} exceeds video resolution {self.width}x{self.height}")

        try:
            # Calculate start time for seeking
            start_time_sec = frame_num / self.fps if self.fps > 0 else 0.0

            # Use read_frames with seeking
            frame_generator = self.read_frames(start_frame_num=frame_num)

            # Get the target frame
            try:
                frame = next(frame_generator)
            except StopIteration:
                raise RuntimeError(f"Could not read frame {frame_num}. Possibly out of range.")

            # Draw green rectangle around ROI
            cv2.rectangle(
                img=frame,
                pt1=(x, y),
                pt2=(x + w, y + h),
                color=(0, 255, 0),  # Green in BGR
                thickness=2
            )

            # Overlay frame number
            cv2.putText(
                img=frame,
                text=f"Frame: {frame_num}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA
            )

            # Optionally display the full annotated frame
            if show:
                window_title = f"Full Frame with ROI - Frame {frame_num}"
                cv2.imshow(window_title, frame)
                print(f"Displayed full frame with ROI boundary for frame {frame_num}")
                cv2.waitKey(0)  # Wait for keypress
                cv2.destroyWindow(window_title)

            # Optionally save the annotated frame
            if save_path is not None:
                try:
                    success = cv2.imwrite(save_path, frame)
                    if success:
                        print(f"Saved full frame with ROI to: {save_path}")
                    else:
                        raise RuntimeError(f"Failed to save annotated frame to {save_path}")
                except Exception as e:
                    raise RuntimeError(f"Error saving annotated frame: {e}") from e

        except Exception as e:
            raise RuntimeError(f"Error in visualize_roi_on_frame: {e}") from e

        finally:
            # Always clean up
            self.close()
            if not show:
                cv2.destroyAllWindows()

    def close(self):
        """
        Safely terminate the ffmpeg subprocess and release all resources.
        
        This method ensures that:
        - The stdout pipe is closed.
        - The ffmpeg process is terminated.
        - The process is waited for to prevent zombie processes.
        - The pipe reference is cleared.
        
        Can be called multiple times safely.
        """
        if self.pipe is None:
            return  # Already closed or never started

        try:
            # Close the stdout pipe to signal EOF
            if self.pipe.stdout:
                self.pipe.stdout.close()
        except (OSError, AttributeError):
            pass  # Ignore errors during close

        try:
            # Terminate the ffmpeg process
            if self.pipe.poll() is None:  # Still running
                self.pipe.terminate()
                try:
                    # Wait up to 5 seconds for clean exit
                    self.pipe.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't respond
                    print("Warning: ffmpeg did not terminate gracefully, forcing kill.")
                    self.pipe.kill()
                    self.pipe.wait()  # Final wait after kill
        except (OSError, subprocess.TimeoutExpired):
            pass  # Ignore errors during termination
        finally:
            self.pipe = None  # Clear reference

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def yml_save_frames(yml_file):
    with open(yml_file, 'r') as file:
        frames = yaml.safe_load(file)

        for frame in frames:
            video_path = "./videos/" + frame['video']

            if os.path.isfile(video_path):
                id, frame_num, x, y, w, h = frame['id'], frame['frame_number'], frame['roi_left'], frame['roi_top'], frame['roi_width'], frame['roi_height']

                with Y4MVideoReader(video_path) as reader:
                    reader.extract_roi(frame_num=frame_num, roi_rect=(x, y, w, h), save_path=f"dataset/motion_blur/{id}.png", show=False)

def jsonl_save_frames(category):
    jsonl_file = f'dataset_files/{category}.jsonl'

    with open(jsonl_file, 'r') as file:
        for id, line in enumerate(file):
            frame = json.loads(line)
            video_path = "./videos/" + frame['video']

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
            video_path = "./videos/" + frame['video']

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

def save_frames(video_path, start_frame_num, end_frame_num):
    pass

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
    save_roi_contexts(category='motion_blur', neighboring_frames=10, force_regen=True, generate_video=True)