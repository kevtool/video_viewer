import sys, os, time
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

NUM_MODES = 4
ORIGINAL_MODE = 0
Y_DIFF_MODE = 1
U_DIFF_MODE = 2
V_DIFF_MODE = 3
mode_txt = [
    'Original',
    'Y-Diff',
    'U-Diff',
    'V-Diff'
]

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
    def __init__(self, save_folder=''):
        super().__init__()
        self.setWindowTitle("Video ROI Recorder")

        # save folder
        self.save_folder = save_folder
        if not os.path.exists(save_folder): 
            os.makedirs(save_folder)

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
        self.writer_full = None
        self.record_start_frame = None
        self.record_end_frame = None
        self.show_mode = ORIGINAL_MODE
        self.next_frame_buffer = None

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
        # self.pause_btn     = QPushButton("Pause")
        self.rewind_btn  = QPushButton("<< Rewind")
        self.forward_btn = QPushButton("Forward >>")
        self.roi_btn       = QPushButton("Record ROI")
        self.zoom_btn      = QPushButton("Show ROI Zoom")
        self.reset_btn     = QPushButton("Reset Zoom")
        self.start_rec_btn = QPushButton("Start Recording")
        self.stop_rec_btn  = QPushButton("Stop Recording")
        self.show_mode_btn = QPushButton(mode_txt[self.show_mode])

        for btn in (
            self.play_btn, self.rewind_btn, self.forward_btn,
            self.roi_btn, self.zoom_btn, self.reset_btn,
            self.start_rec_btn, self.stop_rec_btn, self.show_mode_btn
        ):
            btn.setEnabled(False)

        ctrl = QHBoxLayout()
        for w in (
            self.frame_label, self.load_btn, self.play_btn, self.rewind_btn, self.forward_btn,
            self.roi_btn, self.zoom_btn, self.reset_btn,
            self.start_rec_btn, self.stop_rec_btn, self.show_mode_btn
        ):
            ctrl.addWidget(w)

        # Assemble main layout
        layout = QVBoxLayout(self)
        layout.addLayout(video_area)
        layout.addLayout(ctrl)

        # Connect signals
        self.load_btn.clicked.connect(self.load_video)
        self.play_btn.clicked.connect(self.toggle_play_pause)
        # self.pause_btn.clicked.connect(self.pause)
        self.rewind_btn.clicked.connect(lambda: self.jump_frames(-30))   # ~1 sec backward if fps=30
        self.forward_btn.clicked.connect(lambda: self.jump_frames(30))
        self.roi_btn.clicked.connect(self.enable_roi)
        self.zoom_btn.clicked.connect(self.show_zoom_subwindow)
        self.reset_btn.clicked.connect(self.clear_zoom)
        self.start_rec_btn.clicked.connect(self.start_recording)
        self.stop_rec_btn.clicked.connect(self.stop_recording)
        self.show_mode_btn.clicked.connect(self.toggle_y_view)

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
        self.next_frame_buffer = next(self.frame_iter)  # prefetch next frame
        self.global_frame_number = 0
        self.update_display_frame(first)

        # reset iterator and enable controls
        self.frame_iter = self.container.decode(video=0)
        for btn in (
            self.play_btn, self.rewind_btn, self.forward_btn, 
            self.roi_btn, self.zoom_btn, self.reset_btn, self.start_rec_btn,
            self.show_mode_btn
        ):
            btn.setEnabled(True)
        self.stop_rec_btn.setEnabled(False)

    def toggle_play_pause(self):
        """Toggle between playing and pausing the video."""
        if not self.container:
            return

        if self.is_playing:
            # Pause
            self.is_playing = False
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            # Play
            self.is_playing = True
            self.timer.start(self.frame_interval)
            self.play_btn.setText("Pause")

    def jump_frames(self, offset):
        """
        Jump forward or backward in the video by `offset` frames.
        Positive = forward, Negative = backward.
        """
        if not self.container or self.current_frame is None:
            return

        target_frame = max(0, self.global_frame_number + offset)
        target_pts = int(target_frame / self.fps * av.time_base)  # convert to pts

        # Seek close to the target
        self.container.seek(int(target_frame / self.fps * self.video_stream.time_base.denominator),
                            any_frame=False, backward=offset < 0, stream=self.video_stream)

        # Restart decoding from seek point
        self.frame_iter = self.container.decode(video=0)

        # Skip until target_frame is reached
        for frame in self.frame_iter:
            self.global_frame_number = int(round(frame.time * self.fps))
            if self.global_frame_number >= target_frame:
                self.current_frame = frame
                try:
                    self.next_frame_buffer = next(self.frame_iter)
                except StopIteration:
                    self.next_frame_buffer = None
                self.update_display_frame(self.current_frame)
                break

    def next_frame(self):
        """Advance one frame, record if active, then render update."""
        if self.next_frame_buffer is None:
            self.toggle_play_pause()
            return
    
        self.current_frame = self.next_frame_buffer
        try:
            self.next_frame_buffer = next(self.frame_iter)
        except StopIteration:
            self.next_frame_buffer = None

        self.global_frame_number = int(round(self.current_frame.time * self.fps))

        # If recording, write the cropped ROI region
        if self.recording and self.video_label.topleft:
            bgr = self.current_frame.to_ndarray(format='bgr24')
            x0, y0 = self.video_label.topleft
            x1, y1 = self.video_label.bottomright

            # Cropped ROI
            crop = bgr[y0:y1, x0:x1]
            self.writer.write(crop)

            # Full frame with rectangle
            bgr_with_box = bgr.copy()
            cv2.rectangle(bgr_with_box, (x0, y0), (x1, y1), (0, 255, 0), 2)
            self.writer_full.write(bgr_with_box)

        self.update_display_frame(self.current_frame)

    def update_display_frame(self, frame):
        """
        Render the current frame into the main video label (with ROI overlay),
        and continuously update the zoom inset with the cropped ROI video.
        """
        arr = frame.to_ndarray(format='rgb24')
        h, w, _ = arr.shape

        if self.show_mode == Y_DIFF_MODE or (hasattr(self, 'next_frame_buffer') and self.next_frame_buffer is not None):

            # YUV difference
            yuv = frame.to_ndarray(format='yuv420p')
            y = yuv[0:h, :]  # Y channel
            u = yuv[h:h + h // 4, :w // 2]
            v = yuv[h + h // 4:, :w // 2]
            u, v = [cv2.resize(ch, (w, h), interpolation=cv2.INTER_LINEAR) for ch in (u, v)]

            if hasattr(self, 'next_frame_buffer') and self.next_frame_buffer is not None:
                next_yuv = self.next_frame_buffer.to_ndarray(format='yuv420p')
                next_y = next_yuv[0:h, :]
                next_u = next_yuv[h:h + h // 4, :w // 2]
                next_v = next_yuv[h + h // 4:, :w // 2]
                next_u, next_v = [cv2.resize(ch, (w, h), interpolation=cv2.INTER_LINEAR) for ch in (next_u, next_v)]
                # Now you can compute frame diff, optical flow, etc. on y and next_y
                # Example:

                y_diff = cv2.absdiff(y, next_y)
                u_diff = np.abs(u.astype(np.int16) - next_u).astype(np.uint8) * 10
                v_diff = np.abs(v.astype(np.int16) - next_v).astype(np.uint8) * 10


            else:
                next_y = None
                next_u = None
                next_v = None

            if self.show_mode == Y_DIFF_MODE:
                arr = np.stack([y_diff] * 3, axis=-1)
            elif self.show_mode == U_DIFF_MODE:
                arr = np.stack([u_diff] * 3, axis=-1)
            elif self.show_mode == V_DIFF_MODE:
                arr = np.stack([v_diff] * 3, axis=-1)
            
        else:
            # Regular RGB display
            
            if self.next_frame_buffer is not None:
                next_arr = self.next_frame_buffer.to_ndarray(format='rgb24')
                # Now you have both `arr` and `next_arr` as numpy arrays (RGB)
                # Do whatever comparison, optical flow, prediction, etc. you need
            else:
                next_arr = None
            

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
        self.basename = f"{base}_{int(time.time())}"
        out = os.path.join(self.save_folder, f"{self.basename}_roi.mp4")
        x0, y0 = self.video_label.topleft
        x1, y1 = self.video_label.bottomright
        w_rect, h_rect = x1 - x0, y1 - y0

        self.writer = cv2.VideoWriter(
            out,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (w_rect, h_rect)
        )

        out_full = os.path.join(self.save_folder, f"{self.basename}_context.mp4")
        frame_w = self.current_frame.width
        frame_h = self.current_frame.height

        self.writer_full = cv2.VideoWriter(
            out_full,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (frame_w, frame_h)
        )

        self.recording = True
        self.record_start_frame = self.global_frame_number
        print(f"[INFO] Recording to {out} from frame {self.record_start_frame}")
        self.start_rec_btn.setEnabled(False)
        self.stop_rec_btn.setEnabled(True)

    def stop_recording(self):
        """Stop recording and save metadata JSON."""
        self.recording = False
        if self.writer:
            self.writer.release()
        if self.writer_full:
            self.writer_full.release()
        self.record_end_frame = self.global_frame_number

        meta = {
            'video_file': self.container.name,
            'start_frame': self.record_start_frame,
            'end_frame': self.record_end_frame,
            'topleft': self.video_label.topleft,
            'bottomright': self.video_label.bottomright
        }
        jf = os.path.join(self.save_folder, f"{self.basename}_roi.json")
        with open(jf, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"[INFO] Saved metadata to {jf}")
        self.start_rec_btn.setEnabled(True)
        self.stop_rec_btn.setEnabled(False)

    def toggle_y_view(self):
        """Toggle between RGB and Y (luma) display."""
        self.show_mode = (self.show_mode + 1) % NUM_MODES
        self.show_mode_btn.setText(mode_txt[self.show_mode])
        # Refresh current frame
        if self.current_frame is not None:
            self.update_display_frame(self.current_frame)

    def closeEvent(self, event):
        """Cleanup on exit."""
        if getattr(self, 'writer', None):
            self.writer.release()
        if self.container:
            self.container.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = Y4MPlayer(save_folder='dataset/perceptual_temporal_noise')
    player.show()  # keep the overall window size unchanged
    sys.exit(app.exec_())
