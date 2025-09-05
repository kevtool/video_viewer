import sys, os, time, subprocess
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
    QMessageBox, QSizePolicy, QGridLayout
)

from patch_pca import FramePCA
from utils import save_histogram_from_bins

NUM_MODES = 5
ORIGINAL_MODE = 0
PCA_MODE = 1
PCA_DIFF_1_MODE = 2 # t and t+1
PCA_DIFF_2_MODE = 3 # t and t+2
PCA_DIFF_3_MODE = 4 # t and t+3
mode_txt = [
    'Original',
    'PCA1',
    'PCA1-Diff, t+1',
    'PCA1-Diff, t+2',
    'PCA1-Diff, t+3'
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
                self.player.gethist_btn.setEnabled(True)
            
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
        self.output_format = 'y4m'
        self.show_mode = ORIGINAL_MODE
        # self.next_frame_buffer = None
        self.frames = [None] * 4
        self.eigenvectors = None
        self.frozen_eigenvecs = False

        self.frame_pca = FramePCA()
        self.frame_pca.initialize_histogram()

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

        # -- Control buttons first row --
        self.frame_label   = QLabel("Frame: 0")
        self.load_btn      = QPushButton("Load Video")
        self.play_btn      = QPushButton("Play")
        # self.pause_btn     = QPushButton("Pause")
        self.rewind_btn    = QPushButton("<< Rewind")
        self.forward_btn   = QPushButton("Forward >>")
        self.roi_btn       = QPushButton("Record ROI")
        self.zoom_btn      = QPushButton("Show ROI Zoom")
        self.reset_btn     = QPushButton("Reset Zoom")
        self.rec_btn       = QPushButton("Start Recording")
        self.sw_format_btn = QPushButton("Output: Y4M")

        # -- Control buttons second row --
        self.gethist_btn   = QPushButton("Compute Histogram")
        self.show_mode_btn = QPushButton(mode_txt[self.show_mode])
        self.eigenvec_btn  = QPushButton("Lock Eigenv's")

        for btn in (
            self.play_btn, self.rewind_btn, self.forward_btn,
            self.roi_btn, self.zoom_btn, self.reset_btn,
            self.rec_btn, self.sw_format_btn, self.gethist_btn,
            self.show_mode_btn,
        ):
            btn.setEnabled(False)

        ctrl_layout = QGridLayout()
        ctrl_layout.setSpacing(10)
        ctrl_layout.setContentsMargins(0, 10, 0, 0)

        # === First Row ===
        ctrl_layout.addWidget(self.frame_label,   0, 0)
        ctrl_layout.addWidget(self.load_btn,      0, 1)
        ctrl_layout.addWidget(self.play_btn,      0, 2)
        ctrl_layout.addWidget(self.rewind_btn,    0, 3)
        ctrl_layout.addWidget(self.forward_btn,   0, 4)
        ctrl_layout.addWidget(self.roi_btn,       0, 5)
        ctrl_layout.addWidget(self.zoom_btn,      0, 6)
        ctrl_layout.addWidget(self.reset_btn,     0, 7)
        ctrl_layout.addWidget(self.rec_btn,       0, 8)
        ctrl_layout.addWidget(self.sw_format_btn, 0, 9)

        # === Second Row ===
        ctrl_layout.addWidget(self.gethist_btn,   1, 7)
        ctrl_layout.addWidget(self.show_mode_btn, 1, 8)
        ctrl_layout.addWidget(self.eigenvec_btn,  1, 9)

        # Assemble main layout
        layout = QVBoxLayout(self)
        layout.addLayout(video_area)
        layout.addLayout(ctrl_layout)

        # Connect signals
        self.load_btn.clicked.connect(self.load_video)
        self.play_btn.clicked.connect(self.toggle_play_pause)
        # self.pause_btn.clicked.connect(self.pause)
        self.rewind_btn.clicked.connect(lambda: self.jump_frames(-30))   # ~1 sec backward if fps=30
        self.forward_btn.clicked.connect(lambda: self.jump_frames(30))
        self.roi_btn.clicked.connect(self.enable_roi)
        self.zoom_btn.clicked.connect(self.show_zoom_subwindow)
        self.reset_btn.clicked.connect(self.clear_zoom)
        self.rec_btn.clicked.connect(self.toggle_recording)
        self.sw_format_btn.clicked.connect(self.switch_format)
        self.gethist_btn.clicked.connect(self.compute_histogram)
        self.show_mode_btn.clicked.connect(self.toggle_y_view)
        self.eigenvec_btn.clicked.connect(self.toggle_eigenvecs)

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

        
        for i in range(1, 4):
            try:
                self.frames[i] = next(self.frame_iter)
            except StopIteration:
                self.frames[i] = None

        self.global_frame_number = 0
        self.update_display_frame(first)

        # reset iterator and enable controls
        # self.frame_iter = self.container.decode(video=0)
        for btn in (
            self.play_btn, self.rewind_btn, self.forward_btn, 
            self.roi_btn, self.zoom_btn, self.reset_btn, self.rec_btn,
            self.sw_format_btn, self.show_mode_btn
        ):
            btn.setEnabled(True)

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


                for i in range(1, 4):
                    try:
                        self.frames[i] = next(self.frame_iter)
                    except StopIteration:
                        self.frames[i] = None
                self.update_display_frame(self.current_frame)
                break

    def next_frame(self):
        """Advance one frame, record if active, then render update."""
        if self.frames[1] is None:
            self.toggle_play_pause()
            return
    
        self.current_frame = self.frames[1]
        for i in range(1, 3):
            self.frames[i] = self.frames[i + 1]
        try:
            self.frames[3] = next(self.frame_iter)
        except StopIteration:
            self.frames[3] = None

        self.global_frame_number = int(round(self.current_frame.time * self.fps))

        # If recording, write the cropped ROI region
        if self.recording and self.video_label.topleft:
            bgr = self.get_processed_frame(self.current_frame)
            x0, y0 = self.video_label.topleft
            x1, y1 = self.video_label.bottomright

            # Cropped ROI
            crop = bgr[y0:y1, x0:x1]
            if self.output_format == 'mp4':
                self.writer.write(crop)
            elif self.output_format == 'y4m':
                self.proc_roi.stdin.write(crop.tobytes())

            # Full frame with rectangle
            bgr_with_box = bgr.copy()
            cv2.rectangle(bgr_with_box, (x0, y0), (x1, y1), (0, 255, 0), 2)
            if self.output_format == 'mp4':
                self.writer_full.write(bgr_with_box)
            elif self.output_format == 'y4m':
                self.proc_full.stdin.write(bgr_with_box.tobytes())

        self.update_display_frame(self.current_frame)

    def get_processed_frame(self, frame):
        """
        Returns an RGB np.array that matches the current self.show_mode display.
        """
        arr = frame.to_ndarray(format='rgb24')
        h, w, _ = arr.shape

        # PCA mode
        if self.show_mode == PCA_DIFF_1_MODE and self.video_label.topleft and (self.frames[1] is not None):

            next_arr = self.frames[1].to_ndarray(format='rgb24')
            arr = self.frame_pca.patch_pca(arr, next_arr, self.video_label.topleft, patch_size=128, entire_frame=True)

        elif self.show_mode == PCA_DIFF_2_MODE and self.video_label.topleft and (self.frames[2] is not None):

            next_arr = self.frames[2].to_ndarray(format='rgb24')
            arr = self.frame_pca.patch_pca(arr, next_arr, self.video_label.topleft, patch_size=128, entire_frame=True)

        elif self.show_mode == PCA_DIFF_3_MODE and self.video_label.topleft and (self.frames[3] is not None):

            next_arr = self.frames[3].to_ndarray(format='rgb24')
            arr = self.frame_pca.patch_pca(arr, next_arr, self.video_label.topleft, patch_size=128, entire_frame=True)

            

        elif self.show_mode == PCA_MODE and self.video_label.topleft and self.video_label.bottomright:

            x0, y0 = self.video_label.topleft
            x1, y1 = self.video_label.bottomright
            roi = arr[y0:y1, x0:x1]
            data = roi.reshape(-1, 3).T  # shape (3, N)
            mean_color = np.mean(data, axis=1, keepdims=True)

            # only recalculate eigenvectors if not frozen
            if (not self.frozen_eigenvecs) or self.eigenvectors is None:
                # Center the data
                data_centered = data - mean_color
                covariance_matrix = data_centered @ data_centered.T / (3 - 1)

                # get eigenvectors
                eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)  # eigh for symmetric matrices

                # Sort by eigenvalues in descending order
                idx = np.argsort(eigenvals)[::-1]  # descending
                eigenvals = eigenvals[idx]

                eigenvectors = eigenvecs[:, idx]
                self.eigenvectors = eigenvectors

            else:
                eigenvectors = self.eigenvectors

            eigenvector = eigenvectors[0, :]
                        
            pixels = arr.reshape(-1, 3).T - mean_color
            pc = pixels.T @ eigenvector 
            pc = pc.reshape(h, w) 
            arr = np.stack([pc, pc, pc], axis=-1)
            arr = np.clip(arr, 0, 255)
            roi = arr[y0:y1, x0:x1]
            arr = (arr - roi.min()) / (roi.max() - roi.min()) * 255

            print(arr[y0:y1, x0:x1].min(), arr[y0:y1, x0:x1].max(), arr.min(), arr.max())
            arr = np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)

        return arr

    def update_display_frame(self, frame):
        """
        Render the current frame into the main video label (with ROI overlay),
        and continuously update the zoom inset with the cropped ROI video.
        """
        arr = self.get_processed_frame(frame)
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

    def compute_histogram(self):
        if not self.container or not self.video_label.topleft:
            return
        
        prev_arr = None
        frame_iter = self.container.decode(video=0)
        counts = np.zeros((64,), dtype=np.int32)

        for frame in frame_iter:
            arr = frame.to_ndarray(format='rgb24')

            if prev_arr is not None:
                counts += self.frame_pca.patch_pca(prev_arr, arr, self.video_label.topleft, patch_size=128, entire_frame=False,
                                         get_histogram_data=True)
            
                print(counts)

            prev_arr = arr

        save_histogram_from_bins(counts)

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
        """Begin recording cropped ROI and full context, either as MP4 or Y4M."""
        if not self.video_label.topleft:
            QMessageBox.warning(self, "No ROI", "Draw an ROI before recording.")
            return

        base = Path(self.container.name).stem
        curr_time = int(time.time())
        self.basename = f"{curr_time}/{base}_{curr_time}"

        sample_folder = Path(self.save_folder) / f"{curr_time}"
        if not os.path.exists(sample_folder): 
            os.makedirs(sample_folder)


        x0, y0 = self.video_label.topleft
        x1, y1 = self.video_label.bottomright
        w_rect, h_rect = x1 - x0, y1 - y0

        frame_w = self.current_frame.width
        frame_h = self.current_frame.height

        # Paths
        roi_ext = self.output_format
        full_ext = self.output_format
        out_roi = Path(self.save_folder) / f"{self.basename}_roi.{roi_ext}"
        out_full = Path(self.save_folder) / f"{self.basename}_context.{full_ext}"

        self.out_roi_path = out_roi
        self.out_full_path = out_full

        self.record_start_frame = self.global_frame_number
        self.recording = True

        # --- MP4: Use OpenCV VideoWriter ---
        if self.output_format == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            self.writer = cv2.VideoWriter(out_roi, fourcc, self.fps, (w_rect, h_rect))
            self.writer_full = cv2.VideoWriter(out_full, fourcc, self.fps, (frame_w, frame_h))

            if not self.writer or not self.writer_full:
                QMessageBox.critical(self, "Error", "Failed to initialize MP4 video writers.")
                self.recording = False
                return

        # --- Y4M: Use FFmpeg subprocess ---
        elif self.output_format == 'y4m':
            cmd_roi = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{w_rect}x{h_rect}',
                '-r', str(self.fps),
                '-i', '-',
                '-pix_fmt', 'yuv420p',
                '-f', 'yuv4mpegpipe',
                out_roi
            ]
            self.proc_roi = subprocess.Popen(cmd_roi, stdin=subprocess.PIPE)

            cmd_full = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{frame_w}x{frame_h}',
                '-r', str(self.fps),
                '-i', '-',
                '-pix_fmt', 'yuv420p',
                '-f', 'yuv4mpegpipe',
                out_full
            ]
            self.proc_full = subprocess.Popen(cmd_full, stdin=subprocess.PIPE)

        else:
            QMessageBox.critical(self, "Error", f"Unsupported format: {self.output_format}")
            self.recording = False
            return

        print(f"[INFO] Recording to {out_roi} and {out_full} from frame {self.record_start_frame}")
        self.sw_format_btn.setEnabled(False)
        self.rec_btn.setText("Stop Recording")

    def stop_recording(self):
        """Stop recording and save metadata JSON."""
        self.recording = False
        self.record_end_frame = self.global_frame_number

        # --- MP4: Release OpenCV writers ---
        if self.output_format == 'mp4':
            if hasattr(self, 'writer') and self.writer:
                self.writer.release()
            if hasattr(self, 'writer_full') and self.writer_full:
                self.writer_full.release()

        # --- Y4M: Close FFmpeg stdin and wait ---
        elif self.output_format == 'y4m':
            if hasattr(self, 'proc_roi') and self.proc_roi and self.proc_roi.stdin:
                self.proc_roi.stdin.close()
                self.proc_roi.wait()
            if hasattr(self, 'proc_full') and self.proc_full and self.proc_full.stdin:
                self.proc_full.stdin.close()
                self.proc_full.wait()

        # Save metadata
        meta = {
            'video_file': self.container.name,
            'start_frame': self.record_start_frame,
            'end_frame': self.record_end_frame,
            'topleft': self.video_label.topleft,
            'bottomright': self.video_label.bottomright,
            'roi_video': self.out_roi_path.as_posix(),
            'context_video': self.out_full_path.as_posix(),
            'format': self.output_format
        }

        jf = Path(self.save_folder) / f"{self.basename}_roi.json"
        with open(jf, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"[INFO] Saved metadata to {jf}")
        self.sw_format_btn.setEnabled(True)
        self.rec_btn.setText("Start Recording")

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def toggle_eigenvecs(self):
        if self.frozen_eigenvecs:
            self.frozen_eigenvecs = False
            self.eigenvec_btn.setText("Lock Eigenv's")
            self.frame_pca.set_freeze_eigenvectors(False)
        else:
            self.frozen_eigenvecs = True
            self.eigenvec_btn.setText("Unlock Eigenv's")
            self.frame_pca.set_freeze_eigenvectors(True)
            

    def switch_format(self):
        """Switch output format between Y4M and MP4"""
        if self.output_format == 'mp4':
            self.output_format = 'y4m'
            self.sw_format_btn.setText('Output: Y4M')
        elif self.output_format == 'y4m':
            self.output_format = 'mp4'
            self.sw_format_btn.setText('Output: MP4')
        else:
            raise Exception('ERROR: Invalid output file format')

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
