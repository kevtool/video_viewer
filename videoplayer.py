import subprocess
import numpy as np
import cv2
import re

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

    def _start_ffmpeg_process(self):
        command = [
            'ffmpeg',
            '-i', self.file_path,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo',
            '-'
        ]
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

    def play(self, resize_factor=0.3):
        """Play the video interactively using OpenCV"""
        try:
            for idx, frame in enumerate(self.read_frames(), start=1):
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

    def extract_roi(self, frame_num, roi_rect, zoom_factor=2):
        """Extract and display ROI from a specific frame"""
        x, y, w, h = roi_rect
        try:
            for idx, frame in enumerate(self.read_frames(), start=1):
                if idx == frame_num:
                    roi = frame[y:y+h, x:x+w]
                    resized = cv2.resize(roi, None, fx=zoom_factor, fy=zoom_factor)
                    cv2.imshow(f"ROI - Frame {frame_num}", resized)
                    print(f"Displayed ROI of frame {frame_num}")
                    cv2.waitKey(0)
                    break
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

if __name__ == "__main__":
    # with Y4MVideoReader("./videos/netflix_aerial.y4m") as reader:
    #     reader.play(resize_factor=0.3)

    with Y4MVideoReader("./videos/netflix_aerial.y4m") as reader:
        reader.extract_roi(frame_num=100, roi_rect=(500, 300, 200, 200), zoom_factor=2)