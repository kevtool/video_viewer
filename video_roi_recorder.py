# import sys
# import json
# import cv2
# import av
# import numpy as np
# from pathlib import Path
# from PyQt5.QtCore import Qt, QTimer, QRect, QSize, QPoint
# from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QLabel, QPushButton,
#     QVBoxLayout, QHBoxLayout, QFileDialog, QRubberBand,
#     QMessageBox, QSizePolicy
# )

# class VideoLabel(QLabel):
#     """QLabel subclass that lets the user draw an ROI and maps widget coords back to original video pixels."""
#     def __init__(self, player):
#         super().__init__(player)
#         self.player = player
#         self.setAlignment(Qt.AlignCenter)
#         self.original_width = 1
#         self.original_height = 1
#         self.topleft = None
#         self.bottomright = None
#         self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
#         self.origin = QPoint()

#     def set_original_size(self, w, h):
#         """Remember the dimensions of the video frames."""
#         self.original_width = w
#         self.original_height = h

#     def map_to_original(self, pos):
#         """
#         Convert a QPoint in widget coordinates into (x, y) in the original video frame,
#         accounting for aspect-fit scaling and centering padding.
#         """
#         lw, lh = self.width(), self.height()
#         scale = min(lw / self.original_width, lh / self.original_height)
#         disp_w = int(self.original_width * scale)
#         disp_h = int(self.original_height * scale)
#         pad_x = (lw - disp_w) / 2
#         pad_y = (lh - disp_h) / 2

#         x = (pos.x() - pad_x) / scale
#         y = (pos.y() - pad_y) / scale
#         x = int(np.clip(x, 0, self.original_width))
#         y = int(np.clip(y, 0, self.original_height))
#         return x, y

#     def mousePressEvent(self, event):
#         if self.player.roi_mode and event.button() == Qt.LeftButton:
#             self.origin = event.pos()
#             self.rubberBand.setGeometry(QRect(self.origin, QSize()))
#             self.rubberBand.show()
#         else:
#             super().mousePressEvent(event)

#     def mouseMoveEvent(self, event):
#         if self.player.roi_mode and not self.origin.isNull():
#             self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())
#         else:
#             super().mouseMoveEvent(event)

#     def mouseReleaseEvent(self, event):
#         if self.player.roi_mode and event.button() == Qt.LeftButton:
#             self.rubberBand.hide()
#             p1, p2 = self.origin, event.pos()
#             o1 = self.map_to_original(p1)
#             o2 = self.map_to_original(p2)
#             if o1 and o2:
#                 x0, y0 = o1
#                 x1, y1 = o2
#                 self.topleft = (min(x0, x1), min(y0, y1))
#                 self.bottomright = (max(x0, x1), max(y0, y1))
#                 print(f"[INFO] ROI set to {self.topleft} → {self.bottomright}")
#             self.player.roi_mode = False
#             self.player.update_display_frame(self.player.current_frame)
#         else:
#             super().mouseReleaseEvent(event)


# class Y4MPlayer(QWidget):
#     """Main application widget: load video, play/pause, draw ROI, zoom, record clipped ROI."""
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Video ROI Recorder")

#         # Playback and state variables
#         self.container = None
#         self.frame_iter = None
#         self.video_stream = None
#         self.fps = 25.0
#         self.frame_interval = 40
#         self.current_frame = None
#         self.global_frame_number = 0
#         self.is_playing = False
#         self.roi_mode = False
#         self.recording = False
#         self.writer = None
#         self.record_start_frame = None
#         self.record_end_frame = None

#         # -- Video display widgets --

#         # Main video widget: smaller minimum size
#         self.video_label = VideoLabel(self)
#         self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.video_label.setMinimumSize(640, 360)

#         # Zoom inset widget: larger minimum size
#         self.zoom_label = QLabel(self)
#         self.zoom_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.zoom_label.setMinimumSize(960, 540)
#         self.zoom_label.setAlignment(Qt.AlignCenter)
#         self.zoom_label.setStyleSheet("background-color: black;")

#         # Layout both side by side, with zoom twice as wide as main
#         video_area = QHBoxLayout()
#         video_area.addWidget(self.video_label)
#         video_area.addWidget(self.zoom_label)
#         video_area.setStretch(0, 1)  # main video
#         video_area.setStretch(1, 2)  # zoom inset

#         # -- Control buttons --
#         self.frame_label   = QLabel("Frame: 0")
#         self.load_btn      = QPushButton("Load Video")
#         self.play_btn      = QPushButton("Play")
#         self.pause_btn     = QPushButton("Pause")
#         self.roi_btn       = QPushButton("Record ROI")
#         self.zoom_btn      = QPushButton("Show ROI Zoom")  # still available to jump focus
#         self.reset_btn     = QPushButton("Reset Zoom")
#         self.start_rec_btn = QPushButton("Start Recording")
#         self.stop_rec_btn  = QPushButton("Stop Recording")

#         for btn in (
#             self.play_btn, self.pause_btn, self.roi_btn,
#             self.zoom_btn, self.reset_btn,
#             self.start_rec_btn, self.stop_rec_btn
#         ):
#             btn.setEnabled(False)

#         ctrl = QHBoxLayout()
#         for w in (
#             self.frame_label, self.load_btn, self.play_btn, self.pause_btn,
#             self.roi_btn, self.zoom_btn, self.reset_btn,
#             self.start_rec_btn, self.stop_rec_btn
#         ):
#             ctrl.addWidget(w)

#         # Assemble main layout
#         layout = QVBoxLayout(self)
#         layout.addLayout(video_area)
#         layout.addLayout(ctrl)

#         # Connect signals
#         self.load_btn.clicked.connect(self.load_video)
#         self.play_btn.clicked.connect(self.play)
#         self.pause_btn.clicked.connect(self.pause)
#         self.roi_btn.clicked.connect(self.enable_roi)
#         self.zoom_btn.clicked.connect(self.show_zoom_subwindow)
#         self.reset_btn.clicked.connect(self.clear_zoom)
#         self.start_rec_btn.clicked.connect(self.start_recording)
#         self.stop_rec_btn.clicked.connect(self.stop_recording)

#         # Playback timer
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.next_frame)

#     def load_video(self):
#         """Prompt file dialog, open video, initialize playback and UI."""
#         path, _ = QFileDialog.getOpenFileName(
#             self, "Open Video", "", "Videos (*.y4m *.mp4 *.avi *.mkv)"
#         )
#         if not path:
#             return

#         self.container = av.open(path)
#         self.video_stream = self.container.streams.video[0]
#         self.fps = float(self.video_stream.average_rate)
#         self.frame_interval = int(1000 / self.fps)
#         self.frame_iter = self.container.decode(video=0)

#         # grab first frame to set sizes
#         first = next(self.frame_iter)
#         self.video_label.set_original_size(first.width, first.height)
#         self.current_frame = first
#         self.global_frame_number = 0
#         self.update_display_frame(first)

#         # reset iterator and enable controls
#         self.frame_iter = self.container.decode(video=0)
#         for btn in (
#             self.play_btn, self.pause_btn, self.roi_btn,
#             self.zoom_btn, self.reset_btn, self.start_rec_btn
#         ):
#             btn.setEnabled(True)
#         self.stop_rec_btn.setEnabled(False)

#     def play(self):
#         """Start playback."""
#         if self.container:
#             self.is_playing = True
#             self.timer.start(self.frame_interval)

#     def pause(self):
#         """Pause playback."""
#         self.is_playing = False
#         self.timer.stop()

#     def next_frame(self):
#         """Advance one frame, record if active, then render update."""
#         try:
#             frame = next(self.frame_iter)
#         except StopIteration:
#             self.pause()
#             return

#         self.current_frame = frame
#         self.global_frame_number = int(round(frame.time * self.fps))

#         # If recording, write the cropped ROI region
#         if self.recording and self.video_label.topleft:
#             bgr = frame.to_ndarray(format='bgr24')
#             x0, y0 = self.video_label.topleft
#             x1, y1 = self.video_label.bottomright
#             crop = bgr[y0:y1, x0:x1]
#             self.writer.write(crop)

#         self.update_display_frame(frame)

#     def update_display_frame(self, frame):
#         """
#         Render the current frame into the main video label (with ROI overlay),
#         and continuously update the zoom inset with the cropped ROI video.
#         """
#         arr = frame.to_ndarray(format='rgb24')
#         h, w, _ = arr.shape

#         # compute scale/padding for main video
#         lw, lh = self.video_label.width(), self.video_label.height()
#         scale = min(lw / w, lh / h)
#         disp_w, disp_h = int(w * scale), int(h * scale)

#         # convert to QPixmap
#         bytes_per_line = w * 3
#         qimg = QImage(arr.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
#         pix = QPixmap.fromImage(qimg).scaled(
#             disp_w, disp_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
#         )

#         # draw ROI rect on main video
#         if self.video_label.topleft and self.video_label.bottomright:
#             x0, y0 = self.video_label.topleft
#             x1, y1 = self.video_label.bottomright
#             rx = int(x0 * scale)
#             ry = int(y0 * scale)
#             rw = int((x1 - x0) * scale)
#             rh = int((y1 - y0) * scale)
#             painter = QPainter(pix)
#             painter.setPen(QPen(Qt.green, 2))
#             painter.drawRect(rx, ry, rw, rh)
#             painter.end()

#         self.video_label.setPixmap(pix)
#         self.frame_label.setText(f"Frame: {self.global_frame_number}")

#         # continuously update zoom inset if ROI set
#         if self.video_label.topleft and self.video_label.bottomright:
#             x0, y0 = self.video_label.topleft
#             x1, y1 = self.video_label.bottomright
#             crop = arr[y0:y1, x0:x1]
#             ch, cw, _ = crop.shape
#             bytes_pl = cw * 3
#             qz = QImage(crop.tobytes(), cw, ch, bytes_pl, QImage.Format_RGB888)
#             zp = QPixmap.fromImage(qz).scaled(
#                 self.zoom_label.width(),
#                 self.zoom_label.height(),
#                 Qt.KeepAspectRatio,
#                 Qt.SmoothTransformation
#             )
#             self.zoom_label.setPixmap(zp)

#     def enable_roi(self):
#         """Enable ROI selection mode."""
#         self.roi_mode = True

#     def show_zoom_subwindow(self):
#         """Manually refocus the zoom window on the ROI (still available)."""
#         # no change needed, continuous update now handles playback

#         if not self.video_label.topleft:
#             QMessageBox.warning(self, "No ROI", "Draw an ROI first.")
#             return

#     def clear_zoom(self):
#         """Clear zoom inset."""
#         self.zoom_label.clear()

#     def start_recording(self):
#         """Begin recording cropped ROI video."""
#         if not self.video_label.topleft:
#             QMessageBox.warning(self, "No ROI", "Draw an ROI before recording.")
#             return

#         base = Path(self.container.name).stem
#         out = f"{base}_clip.mp4"
#         x0, y0 = self.video_label.topleft
#         x1, y1 = self.video_label.bottomright
#         w_rect, h_rect = x1 - x0, y1 - y0

#         self.writer = cv2.VideoWriter(
#             out,
#             cv2.VideoWriter_fourcc(*'mp4v'),
#             self.fps,
#             (w_rect, h_rect)
#         )
#         self.recording = True
#         self.record_start_frame = self.global_frame_number
#         print(f"[INFO] Recording to {out} from frame {self.record_start_frame}")
#         self.start_rec_btn.setEnabled(False)
#         self.stop_rec_btn.setEnabled(True)

#     def stop_recording(self):
#         """Stop recording and save metadata JSON."""
#         self.recording = False
#         self.writer.release()
#         self.record_end_frame = self.global_frame_number

#         meta = {
#             'video_file': self.container.name,
#             'start_frame': self.record_start_frame,
#             'end_frame': self.record_end_frame,
#             'topleft': self.video_label.topleft,
#             'bottomright': self.video_label.bottomright
#         }
#         jf = f"{Path(self.container.name).stem}_clip.json"
#         with open(jf, 'w') as f:
#             json.dump(meta, f, indent=2)

#         print(f"[INFO] Saved metadata to {jf}")
#         self.start_rec_btn.setEnabled(True)
#         self.stop_rec_btn.setEnabled(False)

#     def closeEvent(self, event):
#         """Cleanup on exit."""
#         if getattr(self, 'writer', None):
#             self.writer.release()
#         if self.container:
#             self.container.close()
#         event.accept()


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     player = Y4MPlayer()
#     player.show()  # keep the overall window size unchanged
#     sys.exit(app.exec_())


import sys
import json
import cv2
import av
import numpy as np
from pathlib import Path
from PyQt5.QtCore import Qt, QTimer, QRect, QSize, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QRubberBand,
    QMessageBox, QSizePolicy
)

class VideoLabel(QLabel):
    """QLabel subclass that lets the user draw an ROI and maps widget coords back to original video pixels."""
    def __init__(self, player):
        super().__init__(player)
        self.player = player
        self.setAlignment(Qt.AlignCenter)
        self.original_width = 1
        self.original_height = 1
        self.topleft = None
        self.bottomright = None
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()

    def set_original_size(self, w, h):
        """Remember the dimensions of the video frames."""
        self.original_width = w
        self.original_height = h

    def map_to_original(self, pos):
        """
        Convert a QPoint in widget coordinates into (x, y) in the original video frame,
        accounting for aspect-fit scaling and centering padding.
        """
        lw, lh = self.width(), self.height()
        scale = min(lw / self.original_width, lh / self.original_height)
        disp_w = int(self.original_width * scale)
        disp_h = int(self.original_height * scale)
        pad_x = (lw - disp_w) / 2
        pad_y = (lh - disp_h) / 2

        x = (pos.x() - pad_x) / scale
        y = (pos.y() - pad_y) / scale
        x = int(np.clip(x, 0, self.original_width))
        y = int(np.clip(y, 0, self.original_height))
        return x, y

    def mousePressEvent(self, event):
        if self.player.roi_mode and event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.player.roi_mode and not self.origin.isNull():
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.player.roi_mode and event.button() == Qt.LeftButton:
            self.rubberBand.hide()
            p1, p2 = self.origin, event.pos()
            o1 = self.map_to_original(p1)
            o2 = self.map_to_original(p2)
            if o1 and o2:
                x0, y0 = o1
                x1, y1 = o2
                self.topleft = (min(x0, x1), min(y0, y1))
                self.bottomright = (max(x0, x1), max(y0, y1))
                print(f"[INFO] ROI set to {self.topleft} → {self.bottomright}")
            self.player.roi_mode = False
            self.player.update_display_frame(self.player.current_frame)
        else:
            super().mouseReleaseEvent(event)


class Y4MPlayer(QWidget):
    """Main application widget: load video, play/pause, draw ROI, zoom, record clipped ROI."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video ROI Recorder")

        # Playback and state variables
        self.container = None
        self.frame_iter = None
        self.video_stream = None
        self.fps = 25.0
        self.frame_interval = 40
        self.current_frame = None
        self.global_frame_number = 0
        self.is_playing = False
        self.roi_mode = False
        self.recording = False
        self.writer = None
        self.record_start_frame = None
        self.record_end_frame = None

        # -- Video display widgets --

        # Main video widget: narrower minimum size
        self.video_label = VideoLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(560, 315)  # 16:9 aspect

        # Zoom inset widget: also narrower, but larger than main
        self.zoom_label = QLabel(self)
        self.zoom_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.zoom_label.setMinimumSize(840, 472)   # same 16:9 ratio ×1.5
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setStyleSheet("background-color: black;")

        # Layout both side by side, with zoom twice the horizontal stretch of main
        video_area = QHBoxLayout()
        video_area.addWidget(self.video_label)
        video_area.addWidget(self.zoom_label)
        video_area.setStretch(0, 1)  # main video
        video_area.setStretch(1, 2)  # zoom inset

        # -- Control buttons --
        self.frame_label   = QLabel("Frame: 0")
        self.load_btn      = QPushButton("Load Video")
        self.play_btn      = QPushButton("Play")
        self.pause_btn     = QPushButton("Pause")
        self.roi_btn       = QPushButton("Record ROI")
        self.zoom_btn      = QPushButton("Show ROI Zoom")
        self.reset_btn     = QPushButton("Reset Zoom")
        self.start_rec_btn = QPushButton("Start Recording")
        self.stop_rec_btn  = QPushButton("Stop Recording")

        for btn in (
            self.play_btn, self.pause_btn, self.roi_btn,
            self.zoom_btn, self.reset_btn,
            self.start_rec_btn, self.stop_rec_btn
        ):
            btn.setEnabled(False)

        ctrl = QHBoxLayout()
        for w in (
            self.frame_label, self.load_btn, self.play_btn, self.pause_btn,
            self.roi_btn, self.zoom_btn, self.reset_btn,
            self.start_rec_btn, self.stop_rec_btn
        ):
            ctrl.addWidget(w)

        # Assemble main layout
        layout = QVBoxLayout(self)
        layout.addLayout(video_area)
        layout.addLayout(ctrl)

        # Connect signals
        self.load_btn.clicked.connect(self.load_video)
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.roi_btn.clicked.connect(self.enable_roi)
        self.zoom_btn.clicked.connect(self.show_zoom_subwindow)
        self.reset_btn.clicked.connect(self.clear_zoom)
        self.start_rec_btn.clicked.connect(self.start_recording)
        self.stop_rec_btn.clicked.connect(self.stop_recording)

        # Playback timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    def load_video(self):
        """Prompt file dialog, open video, initialize playback and UI."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Videos (*.y4m *.mp4 *.avi *.mkv)"
        )
        if not path:
            return

        self.container = av.open(path)
        self.video_stream = self.container.streams.video[0]
        self.fps = float(self.video_stream.average_rate)
        self.frame_interval = int(1000 / self.fps)
        self.frame_iter = self.container.decode(video=0)

        # grab first frame to set sizes
        first = next(self.frame_iter)
        self.video_label.set_original_size(first.width, first.height)
        self.current_frame = first
        self.global_frame_number = 0
        self.update_display_frame(first)

        # reset iterator and enable controls
        self.frame_iter = self.container.decode(video=0)
        for btn in (
            self.play_btn, self.pause_btn, self.roi_btn,
            self.zoom_btn, self.reset_btn, self.start_rec_btn
        ):
            btn.setEnabled(True)
        self.stop_rec_btn.setEnabled(False)

    def play(self):
        """Start playback."""
        if self.container:
            self.is_playing = True
            self.timer.start(self.frame_interval)

    def pause(self):
        """Pause playback."""
        self.is_playing = False
        self.timer.stop()

    def next_frame(self):
        """Advance one frame, record if active, then render update."""
        try:
            frame = next(self.frame_iter)
        except StopIteration:
            self.pause()
            return

        self.current_frame = frame
        self.global_frame_number = int(round(frame.time * self.fps))

        # If recording, write the cropped ROI region
        if self.recording and self.video_label.topleft:
            bgr = frame.to_ndarray(format='bgr24')
            x0, y0 = self.video_label.topleft
            x1, y1 = self.video_label.bottomright
            crop = bgr[y0:y1, x0:x1]
            self.writer.write(crop)

        self.update_display_frame(frame)

    def update_display_frame(self, frame):
        """
        Render the current frame into the main video label (with ROI overlay),
        and continuously update the zoom inset with the cropped ROI video.
        """
        arr = frame.to_ndarray(format='rgb24')
        h, w, _ = arr.shape

        # compute scale/padding for main video
        lw, lh = self.video_label.width(), self.video_label.height()
        scale = min(lw / w, lh / h)
        disp_w, disp_h = int(w * scale), int(h * scale)

        # convert to QPixmap
        bytes_per_line = w * 3
        qimg = QImage(arr.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            disp_w, disp_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # draw ROI rect on main video
        if self.video_label.topleft and self.video_label.bottomright:
            x0, y0 = self.video_label.topleft
            x1, y1 = self.video_label.bottomright
            rx = int(x0 * scale)
            ry = int(y0 * scale)
            rw = int((x1 - x0) * scale)
            rh = int((y1 - y0) * scale)
            painter = QPainter(pix)
            painter.setPen(QPen(Qt.green, 2))
            painter.drawRect(rx, ry, rw, rh)
            painter.end()

        self.video_label.setPixmap(pix)
        self.frame_label.setText(f"Frame: {self.global_frame_number}")

        # continuously update zoom inset if ROI set
        if self.video_label.topleft and self.video_label.bottomright:
            x0, y0 = self.video_label.topleft
            x1, y1 = self.video_label.bottomright
            crop = arr[y0:y1, x0:x1]
            ch, cw, _ = crop.shape
            bytes_pl = cw * 3
            qz = QImage(crop.tobytes(), cw, ch, bytes_pl, QImage.Format_RGB888)
            zp = QPixmap.fromImage(qz).scaled(
                self.zoom_label.width(),
                self.zoom_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.zoom_label.setPixmap(zp)

    def enable_roi(self):
        """Enable ROI selection mode."""
        self.roi_mode = True

    def show_zoom_subwindow(self):
        """Manually refocus the zoom window on the ROI (now updated continuously)."""
        if not self.video_label.topleft:
            QMessageBox.warning(self, "No ROI", "Draw an ROI first.")
            return

    def clear_zoom(self):
        """Clear zoom inset."""
        self.zoom_label.clear()

    def start_recording(self):
        """Begin recording cropped ROI video."""
        if not self.video_label.topleft:
            QMessageBox.warning(self, "No ROI", "Draw an ROI before recording.")
            return

        base = Path(self.container.name).stem
        out = f"{base}_clip.mp4"
        x0, y0 = self.video_label.topleft
        x1, y1 = self.video_label.bottomright
        w_rect, h_rect = x1 - x0, y1 - y0

        self.writer = cv2.VideoWriter(
            out,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (w_rect, h_rect)
        )
        self.recording = True
        self.record_start_frame = self.global_frame_number
        print(f"[INFO] Recording to {out} from frame {self.record_start_frame}")
        self.start_rec_btn.setEnabled(False)
        self.stop_rec_btn.setEnabled(True)

    def stop_recording(self):
        """Stop recording and save metadata JSON."""
        self.recording = False
        self.writer.release()
        self.record_end_frame = self.global_frame_number

        meta = {
            'video_file': self.container.name,
            'start_frame': self.record_start_frame,
            'end_frame': self.record_end_frame,
            'topleft': self.video_label.topleft,
            'bottomright': self.video_label.bottomright
        }
        jf = f"{Path(self.container.name).stem}_clip.json"
        with open(jf, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"[INFO] Saved metadata to {jf}")
        self.start_rec_btn.setEnabled(True)
        self.stop_rec_btn.setEnabled(False)

    def closeEvent(self, event):
        """Cleanup on exit."""
        if getattr(self, 'writer', None):
            self.writer.release()
        if self.container:
            self.container.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = Y4MPlayer()
    player.show()  # keep the overall window size unchanged
    sys.exit(app.exec_())
