import subprocess
import numpy as np
import cv2
import re

# Path to your .y4m file
y4m_file = './videos/netflix_aerial.y4m'

# FFmpeg command to output raw RGB frames
command = [
    'ffmpeg',
    '-i', y4m_file,
    '-f', 'image2pipe',
    '-pix_fmt', 'rgb24',
    '-vcodec', 'rawvideo',
    '-'
]

# Start FFmpeg subprocess
pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Read and parse the Y4M header manually
with open(y4m_file, 'rb') as f:
    header = ''
    while not header.endswith('\n'):
        byte = f.read(1).decode('utf-8', errors='ignore')
        if not byte:
            break
        header += byte

    match = re.match(r'YUV4MPEG2 W(\d+) H(\d+) F(\d+):(\d+) .*$', header)
    if not match:
        raise ValueError("Invalid Y4M header")

    width = int(match.group(1))
    height = int(match.group(2))
    fps_numerator = int(match.group(3))
    fps_denominator = int(match.group(4))
    fps = fps_numerator / fps_denominator if fps_denominator else 30

frame_size = width * height * 3
target_frame_number = 100  # <-- Change this to the desired frame
frame_count = 0

# Define the region of interest (ROI): x, y, w, h
roi_x, roi_y, roi_w, roi_h = 500, 300, 1000, 1000  # Example: 200x200 px starting at (500,300)

while True:
    raw_frame = pipe.stdout.read(frame_size)
    if len(raw_frame) != frame_size:
        print("Incomplete frame read or end of stream")
        break

    frame_count += 1

    if frame_count == target_frame_number:
        # Convert raw data to image
        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        frame = frame.reshape((height, width, 3)).copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Extract ROI
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        # Optional: Resize for better viewing
        resized_roi = cv2.resize(roi, None, fx=2, fy=2)  # Zoom in 2x

        # Display ROI
        cv2.imshow(f"ROI of Frame {frame_count}", resized_roi)
        print(f"Displayed ROI of frame {frame_count}")
        cv2.waitKey(0)  # Wait indefinitely until key press
        break

# Cleanup
pipe.stdout.close()
pipe.terminate()
cv2.destroyAllWindows()