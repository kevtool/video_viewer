import sys
import av
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QScrollArea
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

            w, h = pixmap_size.width(), pixmap_size.height()
            l_w, l_h = label_size.width(), label_size.height()

            if (w / h) > (l_w / l_h):
                display_height = l_w * (h / w)
                pad_top = (l_h - display_height) / 2
                x_click = event.x()
                y_click = event.y()
                if not (pad_top < y_click < (pad_top + display_height)):
                    print("Click outside video area (top/bottom padding)")
                    return
                y_click -= pad_top
            else:
                display_width = l_h * (w / h)
                pad_left = (l_w - display_width) / 2
                x_click = event.x()
                y_click = event.y()
                if not (pad_left < x_click < (pad_left + display_width)):
                    print("Click outside video area (left/right padding)")
                    return
                x_click -= pad_left

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
        
        self.zoom_factor = 1.0  # Zoom state

        # UI Elements
        self.frame_counter_label = QLabel("Frame: 0", self)
        self.play_pause_btn = QPushButton("Play")
        self.load_btn = QPushButton("Load .y4m File")
        self.rewind_btn = QPushButton("<< Rewind 1s")
        self.forward_btn = QPushButton("Forward 1s >>")
        self.record_roi_btn = QPushButton("Record ROI")
        self.zoom_in_btn = QPushButton("Zoom In (+)")
        self.zoom_out_btn = QPushButton("Zoom Out (-)")
        self.reset_zoom_btn = QPushButton("Reset Zoom")

        # Scroll Area for video label
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)  # Keeps content centered

        # Video Label (inside scroll area)
        self.video_label = VideoLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.scroll_area.setWidget(self.video_label)

        # Layouts
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Fixed-size video container
        video_container = QWidget()
        video_container.setFixedHeight(800)  # Adjust height as needed
        video_layout = QHBoxLayout(video_container)
        video_layout.addWidget(self.scroll_area)
        main_layout.addWidget(video_container, stretch=0)

        # Control Panel
        control_panel = QWidget()
        control_panel.setFixedHeight(60)
        control_panel.setStyleSheet("background-color: #f0f0f0;")
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 5, 10, 5)

        # Add controls
        control_layout.addWidget(self.frame_counter_label)
        control_layout.addStretch()
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.rewind_btn)
        control_layout.addWidget(self.play_pause_btn)
        control_layout.addWidget(self.forward_btn)
        control_layout.addWidget(self.zoom_in_btn)
        control_layout.addWidget(self.zoom_out_btn)
        control_layout.addWidget(self.reset_zoom_btn)
        control_layout.addWidget(self.record_roi_btn)

        main_layout.addWidget(control_panel, stretch=0)

        self.setLayout(main_layout)

        # Connect buttons
        self.load_btn.clicked.connect(self.load_video)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.rewind_btn.clicked.connect(lambda: self.seek_video(-1))
        self.forward_btn.clicked.connect(lambda: self.seek_video(1))
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        self.record_roi_btn.clicked.connect(self.recordROI)

        # timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_next_frame) 

        # savefile
        self.savefile = savefile

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
                # Set original size for correct mapping
                first_frame = next(self.frame_iter)
                self.video_label.set_original_size(first_frame.width, first_frame.height)
                self.update_display_frame(first_frame)
                self.frame_iter = self.container.decode(video=0)  # Reset iterator
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
            self.update_display_frame(frame)
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

        scaled_size = pixmap.size() * self.zoom_factor
        scaled_pixmap = pixmap.scaled(
            scaled_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        self.global_frame_number = int(round(frame.time * self.fps))
        self.frame_counter_label.setText(f"Frame: {self.global_frame_number} | Time: {frame.time:.2f}s")

    def seek_video(self, seconds):
        if self.container is None:
            return
        was_playing = self.is_playing
        self.timer.stop()
        target_seconds = max(0.0, self.current_time_sec + seconds)
        target_pts = int(target_seconds / self.time_base)
        self.container.seek(target_pts, stream=self.video_stream)
        self.frame_iter = self.container.decode(video=0)
        try:
            first_frame = next(self.frame_iter)
            self.current_time_sec = first_frame.time
            self.global_frame_number = int(round(first_frame.time * self.fps))
            self.update_display_frame(first_frame)
        except StopIteration:
            print("No frames found after seeking.")
            return
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

    def zoom_in(self):
        self.zoom_factor *= 1.25
        print(f"Zoomed in: {self.zoom_factor:.2f}x")
        self.show_next_frame()

    def zoom_out(self):
        self.zoom_factor /= 1.25
        self.zoom_factor = max(0.1, self.zoom_factor)
        print(f"Zoomed out: {self.zoom_factor:.2f}x")
        self.show_next_frame()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.show_next_frame()

    def closeEvent(self, event):
        if self.container:
            self.container.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    player = Y4MPlayer(savefile='dataset_files/color_distortion.jsonl')
    player.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()