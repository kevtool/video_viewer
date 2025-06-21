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
    '-f', 'image2pipe',     # Output to pipe
    '-pix_fmt', 'rgb24',    # Output RGB format
    '-vcodec', 'rawvideo',  # Raw video output
    '-'
]

# Start FFmpeg subprocess
pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Read and parse the Y4M header manually to extract width/height/fps
# This avoids relying on FFmpeg output for metadata
with open(y4m_file, 'rb') as f:
    header = ''
    while not header.endswith('\n'):
        byte = f.read(1).decode('utf-8', errors='ignore')
        if not byte:
            break
        header += byte

    # Parse header
    match = re.match(r'YUV4MPEG2 W(\d+) H(\d+) F(\d+):(\d+) .*$', header)
    if not match:
        raise ValueError("Invalid Y4M header")

    width = int(match.group(1))
    height = int(match.group(2))
    fps_numerator = int(match.group(3))
    fps_denominator = int(match.group(4))
    fps = fps_numerator / fps_denominator if fps_denominator else 30

frame_size = width * height * 3  # RGB frames are 3 bytes per pixel
frame_count = 0

while True:
    raw_frame = pipe.stdout.read(frame_size)
    if len(raw_frame) != frame_size:
        print("Incomplete frame read or end of stream")
        break

    # Convert raw binary data to numpy array and reshape
    frame = np.frombuffer(raw_frame, dtype=np.uint8)
    frame = frame.reshape((height, width, 3)).copy()  # Writable copy

    # Convert RGB -> BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Overlay frame number
    frame_count += 1
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize if needed
    resized_frame = cv2.resize(frame, None, fx=0.3, fy=0.3)

    # Show frame
    cv2.imshow('Y4M Player', resized_frame)

    # Exit on 'q' key
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Cleanup
pipe.stdout.close()
pipe.terminate()
cv2.destroyAllWindows()