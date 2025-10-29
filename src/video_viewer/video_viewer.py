import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QSplitter, QSizePolicy
)
from PyQt6.QtCore import QTimer, Qt, QPointF
from PyQt6.QtGui import QImage, QPixmap

# internal imports
from video_viewer.clickable_label import ClickableLabel
from video_viewer.box import DistortionBox, Zoombox
from video_viewer.project_model import ProjectModel
from video_viewer.qtroles import PATH_ROLE, ITEM_TYPE_ROLE, DATA_ROLE
from video_viewer.cov_pca import apply_pca
from video_viewer.channels import *


class VideoViewer(QWidget):
    def __init__(self, project_model: ProjectModel, parent=None):
        super().__init__(parent)
        if parent is None:
            self.setWindowTitle("Video Zoom Tool")

        self.model = project_model
        self.model.file_selected.connect(self.load_video_from_model)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_one_frame_and_render)
        self.paused = False

        # Buffer for current frame (RGB)
        self.current_rgb_frame = None
        self.prev_rgb_frame = None

        # UI Elements
        self.full_view = ClickableLabel(text="Full Video View", parent=self)
        self.zoom_view = QLabel("Zoomed View")
        for lbl in (self.full_view, self.zoom_view):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            lbl.setMinimumSize(10, 10)  # avoid zero-size glitches

        # Splitter between two views
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.full_view)
        self.splitter.addWidget(self.zoom_view)
        self.splitter.setSizes([400, 700])
        # call render_frames (no cap.read) when the splitter is moved
        self.splitter.splitterMoved.connect(lambda pos, idx: self.render_frames())

        # Buttons
        self.play_button = QPushButton("⏸ Pause")
        self.rewind_button = QPushButton("⏪ Rewind")
        self.rewind_1_button = QPushButton("⏪ 1 Frame")
        self.forward_1_button = QPushButton("⏩ 1 Frame")
        self.forward_button = QPushButton("⏩ Forward")
        self.zoom_in_button = QPushButton("➕ Zoom In")
        self.zoom_out_button = QPushButton("➖ Zoom Out")
        self.channel_button = QPushButton("Original")

        self.play_button.clicked.connect(self.toggle_play)
        self.rewind_button.clicked.connect(self.rewind_video)
        self.rewind_1_button.clicked.connect(lambda: self.rewind_video(frame=1))
        self.forward_1_button.clicked.connect(lambda: self.forward_video(frame=1))
        self.forward_button.clicked.connect(self.forward_video)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.channel_button.clicked.connect(self.change_channel)

        self.play_button.setEnabled(False)
        self.rewind_button.setEnabled(False)
        self.rewind_1_button.setEnabled(False)
        self.forward_1_button.setEnabled(False)
        self.forward_button.setEnabled(False)
        self.zoom_in_button.setEnabled(False)
        self.zoom_out_button.setEnabled(False)

        info_layout = QHBoxLayout()
        self.coord_label = QLabel("No box selected")
        self.frame_num_label = QLabel("Frame: 0")
        info_layout.addWidget(self.coord_label)
        info_layout.addWidget(self.frame_num_label)

        button_layout = QHBoxLayout()
        for btn in (self.play_button, self.rewind_button, self.rewind_1_button, self.forward_1_button, self.forward_button,
                    self.zoom_in_button, self.zoom_out_button, self.channel_button):
            button_layout.addWidget(btn)

        # the layout
        layout = QVBoxLayout()
        info_widget = QWidget()
        info_widget.setLayout(info_layout)
        info_widget.setFixedHeight(30)
        layout.addWidget(info_widget)
        layout.addWidget(self.splitter)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.retain_frame_num_on_load = True
        self.zoombox = None  # will be initialized on video open
        self.dragging = False
        self.drag_start_pos = None  # initial mouse position
        self.drag_start_box = None
        self.channel = ORIGINAL

    def load_video_from_model(self, flags):
        if flags["active_video_changed"]:
            if self.model.active_video is None:
                self.clear_video()
            else:
                self.load_video(self.model.active_video.data(0, PATH_ROLE))
        elif flags["boxes_changed"] or flags["selected_box_changed"]:
            self.render_frames()

    def load_video(self, path):
        current_frame = None
        if self.cap:
            if self.retain_frame_num_on_load:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.current_rgb_frame = None
        # start timer for playback
        if not self.paused:
            self.timer.start(30)

        self.read_one_frame()
        if self.current_rgb_frame is None:
            return
        
        self.play_button.setEnabled(True)
        self.rewind_button.setEnabled(True)
        self.rewind_1_button.setEnabled(True)
        self.forward_1_button.setEnabled(True)
        self.forward_button.setEnabled(True)
        self.zoom_in_button.setEnabled(True)
        self.zoom_out_button.setEnabled(True)

        h, w, _ = self.current_rgb_frame.shape
        # Initialize zoombox
        if not self.zoombox:
            self.zoombox = Zoombox(
                zoom_factor=2,
                x=w // 4,
                y=h // 4,
                width=w // 2,
                height=h // 2,
                original_width=w,
                original_height=h,
            )
        else:
            self.zoombox = Zoombox(
                zoom_factor=self.zoombox.zoom_factor,
                x=self.zoombox.x,
                y=self.zoombox.y,
                width=self.zoombox.width,
                height=self.zoombox.height,
                original_width=w,
                original_height=h,
            )

        if self.retain_frame_num_on_load and current_frame:
            try:
                self.reset_prev_frame(current_frame)
            except Exception as e:
                self.reset_prev_frame(0)

        self.render_frames()
        self.model.video_loaded.emit(self.model.active_video.data(0, PATH_ROLE))

    def clear_video(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_rgb_frame = None
        self.timer.stop()
        self.play_button.setEnabled(False)
        self.rewind_button.setEnabled(False)
        self.forward_button.setEnabled(False)
        self.zoom_in_button.setEnabled(False)
        self.zoom_out_button.setEnabled(False)
        self.full_view.setPixmap(QPixmap())
        self.zoom_view.setPixmap(QPixmap())
        self.full_view.setText("Full Video View")
        self.zoom_view.setText("Zoomed View")
        self.model.video_cleared.emit()

    def toggle_play(self):
        # Toggle paused state. If we are switching to play (unpaused)
        # and a video is loaded, ensure the timer is running so
        # update_frame_timer will be called. This fixes the case
        # where the timer was stopped at end-of-video and clicking
        # play did not restart playback.
        if not self.cap:
            return

        if self.paused or not self.timer.isActive():
            # ensure we are in "playing" state and timer is running
            self.paused = False
            self.timer.start(30)
            self.play_button.setText("⏸ Pause")
        else:
            # pause playback
            self.paused = True
            self.timer.stop()
            self.play_button.setText("▶ Play")

    def rewind_video(self, frame: int | None = None):
        """Go back 1 second (adjustable)."""
        seconds = 1
        if not self.cap:
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame:
            new_frame = max(current_frame - 1 - frame, 0)
        else:
            new_frame = max(current_frame - int(seconds * fps) - 1, 0)
        self.reset_prev_frame(new_frame)
        self.render_frames()

    def forward_video(self, frame: int | None = None):
        """Go forward 1 second (adjustable)."""
        seconds = 1
        if not self.cap:
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame:
            new_frame = min(current_frame - 1 + frame, max(total_frames - 1, 0))
        else:
            new_frame = min(current_frame + int(seconds * fps) - 1, max(total_frames - 1, 0))
        self.reset_prev_frame(new_frame)
        self.render_frames()


    def zoom_in(self):
        """Zoom in if a zoombox exists."""
        if not self.zoombox:
            return
        self.zoombox.zoom_in()
        self.render_frames()

    def zoom_out(self):
        """Zoom out if a zoombox exists."""
        if not self.zoombox:
            return
        self.zoombox.zoom_out()
        self.render_frames()

    def start_drag(self, pos: QPointF):
        if self.current_rgb_frame is None:
            return

        self.dragging = True
        self.drag_start_pos = pos
        
        if self.model.selected_box:
            box_data = self.model.selected_box.data(2, DATA_ROLE)
            self.drag_target = "distortion"
            self.drag_start_box = (box_data.x, box_data.y)

        elif self.zoombox:
            self.drag_target = "zoom"
            self.drag_start_box = (self.zoombox.x, self.zoombox.y)
        else:
            self.drag_target = None

    def update_drag(self, pos: QPointF):
        if not self.dragging or self.drag_target is None:
            return

        dx = pos.x() - self.drag_start_pos.x()
        dy = pos.y() - self.drag_start_pos.y()

        # scale from label pixels to frame coordinates
        label_w, label_h = self.full_view.width(), self.full_view.height()
        frame_h, frame_w, _ = self.current_rgb_frame.shape
        scale_x = frame_w / label_w
        scale_y = frame_h / label_h
        dx_frame = dx * scale_x
        dy_frame = dy * scale_y

        if self.drag_target == "zoom" and self.zoombox:
            new_x = min(max(0, self.drag_start_box[0] + dx_frame), frame_w - self.zoombox.width)
            new_y = min(max(0, self.drag_start_box[1] + dy_frame), frame_h - self.zoombox.height)
            self.zoombox.x = new_x
            self.zoombox.y = new_y

        elif self.drag_target == "distortion" and self.model.selected_box is not None:
            box_data = self.model.selected_box.data(2, DATA_ROLE)
            new_x = min(max(0, self.drag_start_box[0] + dx_frame), frame_w - box_data.size)
            new_y = min(max(0, self.drag_start_box[1] + dy_frame), frame_h - box_data.size)

            # update model’s box data in-place
            box_data.x = new_x
            box_data.y = new_y
            self.model.selected_box.setData(2, DATA_ROLE, box_data)

        self.render_frames()

    def update_coord_label(self):
        """Update the coordinate label based on the selected box or zoombox."""
        if self.model.selected_box is not None:
            box_data = self.model.selected_box.data(2, DATA_ROLE)
            x, y = int(box_data.x), int(box_data.y)
            w, h = int(box_data.size), int(box_data.size)
            self.coord_label.setText(f"📦 Box: x={x}, y={y}, w={w}, h={h}")
        elif self.zoombox:
            self.coord_label.setText(
                f"🔍 ZoomBox: x={int(self.zoombox.x)}, y={int(self.zoombox.y)}, "
                f"w={int(self.zoombox.width)}, h={int(self.zoombox.height)}"
            )
        else:
            self.coord_label.setText("No box selected")

    def end_drag(self):
        self.dragging = False
        self.drag_target = None
        self.drag_start_pos = None
        self.drag_start_box = None

    def change_channel(self):
        self.channel += 1
        if self.channel >= NUM_CHANNELS:
            self.channel = 0
        self.channel_button.setText(CHANNEL[self.channel])
        self.render_frames()

    def reset_prev_frame(self, index: int):
        """Reset the previous frame buffer (for residue calculation)."""

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = self.cap.read()
        self.prev_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, frame = self.cap.read()
        self.current_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if MIN_FRAME_NUM[self.channel] <= index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index+1)
            

    def read_one_frame(self):
        """Read a single frame from cap to update the buffer."""
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.handle_end_of_video()
            return
        
        if self.current_rgb_frame is not None:
            # keep a copy for residue calculation
            self.prev_rgb_frame = self.current_rgb_frame.copy()
        else:
            self.prev_rgb_frame = None

        self.current_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

    def read_one_frame_and_render(self):
        """Read a single frame from cap to update the buffer and render."""
        self.read_one_frame()
        self.render_frames()

    def handle_end_of_video(self):
        """Handle end-of-video situation."""
        self.timer.stop()
        self.paused = True
        self.play_button.setText("▶ Play")

    def resizeEvent(self, event):
        """Redraw frame on window resize (and when widgets change size)."""
        # render from buffer (no cap.read)
        self.render_frames()
        super().resizeEvent(event)

    def render_frames(self):
        """Render the buffered frame into both labels with correct scaling.
        This does NOT call cap.read(), so it's safe to call on resizes/splitter moves.
        """
        if self.current_rgb_frame is None:
            return

        if self.channel == ORIGINAL:
            rgb_frame = self.current_rgb_frame.copy()
        elif self.channel == RESIDUE or self.channel == PCA1:
            rgb_frame = self.current_rgb_frame.copy()
            # Apply residue processing here

            if self.prev_rgb_frame is not None:
                residue = cv2.absdiff(self.current_rgb_frame, self.prev_rgb_frame)
                rgb_frame = residue * 4  # amplify for visibility
        
        h, w, _ = rgb_frame.shape
        if h == 0 or w == 0:
            return
        
        boxes_to_draw = []

        # Get zoombox coordinates
        if self.zoombox:
            x1, y1, x2, y2 = self.zoombox.get_coordinates()

        boxes_to_draw.append((x1, y1, x2, y2, (255, 0, 0)))  # red box for zoombox


        zoomed_x1, zoomed_y1 = x1, y1
        zoomed_x2, zoomed_y2 = x2, y2

        if self.model.boxes:
            for box in self.model.boxes:
                box_data = box.data(2, DATA_ROLE)
                x1, y1 = int(box_data.x), int(box_data.y)
                x2, y2 = x1 + int(box_data.size), y1 + int(box_data.size)

                if box == self.model.selected_box:
                    boxes_to_draw.append((x1, y1, x2, y2, (0, 50, 255)))  # blue for selected
                    zoomed_x1, zoomed_y1 = x1, y1
                    zoomed_x2, zoomed_y2 = x2, y2


                    # distortion box must be square. set height equal to width so that zoomed view is also square.
                    h = w
                else:
                    boxes_to_draw.append((x1, y1, x2, y2, (0, 255, 0)))  # green for others

        # If the crop would be invalid for any reason, fall back to whole frame
        if zoomed_x2 <= zoomed_x1 or zoomed_y2 <= zoomed_y1:
            zoomed = rgb_frame
        else:
            zoomed = rgb_frame[zoomed_y1:zoomed_y2, zoomed_x1:zoomed_x2]
            # resize zoomed region up to original frame size for quality before final label scaling
            zoomed = cv2.resize(zoomed, (w, h), interpolation=cv2.INTER_CUBIC)

        if self.channel == PCA1 and self.model.selected_box is not None:
            pc_image = apply_pca(zoomed, component=1)
            pc_norm = cv2.normalize(pc_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            pc_uint8 = np.clip(pc_norm, 0, 255).astype(np.uint8)
            zoomed = np.stack([pc_uint8, pc_uint8, pc_uint8], axis=-1)


        # draw the boxes
        for box in boxes_to_draw:
            x1, y1, x2, y2, color = box
            cv2.rectangle(
                rgb_frame,
                (x1, y1), (x2, y2),
                color=color,
                thickness=2,
            )

        # Helper to draw to a QLabel safely (skip if label size is zero)
        def draw_to_label(label, frame_to_draw):
            lw = max(1, label.width())
            lh = max(1, label.height())
            if lw == 0 or lh == 0:
                return
            # convert frame to QImage / QPixmap
            fh, fw, _ = frame_to_draw.shape
            qimg = QImage(frame_to_draw.data, fw, fh, 3 * fw, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(scaled)

        # Render both views
        draw_to_label(self.full_view, rgb_frame)
        draw_to_label(self.zoom_view, zoomed)

        # if self.model.selected_box:
        #     hist = residue_to_histogram_feature(zoomed)
        #     print(hist)

        # update info bar
        self.update_coord_label()
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_num_label.setText(f"Frame: {current_frame} / {total_frames}")


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     viewer = VideoViewer()
#     viewer.resize(1800, 900)
#     viewer.show()
#     sys.exit(app.exec())