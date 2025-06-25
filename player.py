import sys
import av
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog
)
import json

class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.original_width = 4096
        self.original_height = 2160

        # ROI corners
        self.topleft = None
        self.bottomright = None

    def set_original_size(self, width, height):
        """Call this after decoding a new frame"""
        self.original_width = width
        self.original_height = height

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pixmap = self.pixmap()
            if not pixmap:
                return

            label_size = self.size()
            pixmap_size = pixmap.size()

            # Calculate scale ratios
            width_ratio = self.original_width / pixmap_size.width()
            height_ratio = self.original_height / pixmap_size.height()

            # Determine how much padding there is (aspect ratio scaling)
            w, h = pixmap_size.width(), pixmap_size.height()
            l_w, l_h = label_size.width(), label_size.height()

            if (w / h) > (l_w / l_h):
                # Video width fills label, vertical padding
                display_height = l_w * (h / w)
                pad_top = (l_h - display_height) / 2
                x_click = event.x()
                y_click = event.y()

                if not (pad_top < y_click < (pad_top + display_height)):
                    print("Click outside video area (top/bottom padding)")
                    return

                # Adjust y for padding
                y_click -= pad_top
            else:
                # Video height fills label, horizontal padding
                display_width = l_h * (w / h)
                pad_left = (l_w - display_width) / 2
                x_click = event.x()
                y_click = event.y()

                if not (pad_left < x_click < (pad_left + display_width)):
                    print("Click outside video area (left/right padding)")
                    return

                # Adjust x for padding
                x_click -= pad_left

            # Convert clicked position to original pixel coordinates
            x_click_orig = int(x_click * width_ratio)
            y_click_orig = int(y_click * height_ratio)

            print(f"Clicked at original pixel: ({x_click_orig}, {y_click_orig})")
            self.topleft, self.bottomright = self.bottomright, (x_click_orig, y_click_orig)

    def getROI(self):
        return self.topleft, self.bottomright

class Y4MPlayer(QWidget):
    def __init__(self, savefile=None):
        super().__init__()
        self.setWindowTitle("Y4M Video Player - PyAV")
        self.resize(1600, 1000)

        # UI Elements
        self.video_label = VideoLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.frame_counter_label = QLabel("Frame: 0", self)

        self.play_pause_btn = QPushButton("Play")
        self.load_btn = QPushButton("Load .y4m File")
        self.rewind_btn = QPushButton("<< Rewind 1s")
        self.forward_btn = QPushButton("Forward 1s >>")
        self.record_roi_btn = QPushButton("Record ROI")

        # Layouts
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Video display area
        self.video_label = VideoLabel(self)
        self.video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_label, stretch=10)  # 10x space

        # Control panel (smaller area at the bottom)
        control_panel = QWidget()
        control_panel.setFixedHeight(60)  # Small fixed height
        control_panel.setStyleSheet("background-color: #f0f0f0;")

        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 5, 10, 5)

        # Add controls to control panel
        control_layout.addWidget(self.frame_counter_label)
        control_layout.addStretch()
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.rewind_btn)
        control_layout.addWidget(self.play_pause_btn)
        control_layout.addWidget(self.forward_btn)
        control_layout.addWidget(self.record_roi_btn)

        main_layout.addWidget(control_panel, stretch=1)  # 1x space

        self.setLayout(main_layout)
        
        # Video state
        self.container = None
        self.video_stream = None
        self.frame_iter = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_next_frame)
        self.is_playing = False
        self.fps = 25
        self.frame_interval = int(1000 / self.fps)
        self.global_frame_number = 0
        self.current_time_sec = 0.0  # Track time in seconds
        self.time_base = None

        # filename
        self.filename = None
        self.savefile = savefile

        # Connect buttons
        self.load_btn.clicked.connect(self.load_video)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.rewind_btn.clicked.connect(lambda: self.seek_video(-1))
        self.forward_btn.clicked.connect(lambda: self.seek_video(1))
        self.record_roi_btn.clicked.connect(self.recordROI)

    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Y4M File", "", "Y4M Files (*.y4m)")
        if filename:
            try:
                self.filename = filename.replace('C:/Users/kevin/Documents/MCL/video_distortion/videos/', '').replace('.y4m', '')
                self.container = av.open(filename)
                self.video_stream = self.container.streams.video[0]
                self.video_stream.thread_type = "AUTO"
                self.fps = float(self.video_stream.average_rate)
                self.frame_interval = int(1000 / self.fps)
                self.time_base = float(self.video_stream.time_base)
                self.frame_iter = self.container.decode(video=0)
                self.is_playing = True
                self.global_frame_number = 0
                self.current_time_sec = 0.0
                self.frame_counter_label.setText(f"Frame: {self.global_frame_number}")
                self.timer.start(self.frame_interval)
                self.play_pause_btn.setText("Pause")
            except Exception as e:
                print("Error loading file:", e)

    def toggle_play_pause(self):
        if self.container is None:
            return
        if self.is_playing:
            self.timer.stop()
            self.play_pause_btn.setText("Play")
        else:
            self.timer.start(self.frame_interval)
            self.play_pause_btn.setText("Pause")
        self.is_playing = not self.is_playing

    def show_next_frame(self):
        try:
            frame = next(self.frame_iter)
            self.current_time_sec = frame.time
            self.global_frame_number = int(round(frame.time * self.fps))  # Estimate global frame number
            img = frame.to_image()
            data = img.tobytes("raw", "RGB")
            qimg = QImage(data, img.width, img.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
            self.frame_counter_label.setText(f"Frame: {self.global_frame_number} | Time: {self.current_time_sec:.2f}s")

        except StopIteration:
            self.timer.stop()
            self.play_pause_btn.setText("Play")
            self.is_playing = False
            print("End of video.")

    def update_display_frame(self, frame):
        img = frame.to_image()
        data = img.tobytes("raw", "RGB")
        qimg = QImage(data, img.width, img.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

        self.global_frame_number = int(round(frame.time * self.fps))
        self.frame_counter_label.setText(f"Frame: {self.global_frame_number} | Time: {frame.time:.2f}s")

    def seek_video(self, seconds):
        if self.container is None:
            return

        # Stop playback
        was_playing = self.is_playing
        self.timer.stop()

        # Compute target timestamp
        target_seconds = max(0.0, self.current_time_sec + seconds)
        target_pts = int(target_seconds / self.time_base)

        # Seek
        self.container.seek(target_pts, stream=self.video_stream)
        self.frame_iter = self.container.decode(video=0)

        # Try to get the first frame's PTS after seek
        try:
            first_frame = next(self.frame_iter)
            self.current_time_sec = first_frame.time
            self.global_frame_number = int(round(first_frame.time * self.fps))
            self.update_display_frame(first_frame)
        except StopIteration:
            print("No frames found after seeking.")
            return

        # Restart playback if needed
        if was_playing:
            self.timer.start(self.frame_interval)

        print(f"Jumped to: {self.current_time_sec:.2f}s | Frame: {self.global_frame_number}")

    def recordROI(self):
        if not self.savefile:
            print("No savefile specified.")
            return
        
        topleft, bottomright = self.video_label.getROI()

        if self.filename and topleft and bottomright:
            left, top = topleft
            right, bottom = bottomright

            new_entry = {
                "video": self.filename,
                "frame_number": self.global_frame_number,
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top
            }

            with open(self.savefile, 'a') as file:
                file.write(json.dumps(new_entry) + '\n')

            print(f'ROI {topleft}, {bottomright} recorded to {self.savefile}')

    def closeEvent(self, event):
        if self.container:
            self.container.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    player = Y4MPlayer(savefile='dataset_files/motion_blur.jsonl')
    player.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()