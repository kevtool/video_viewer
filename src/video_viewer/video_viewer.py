import sys
import cv2
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


class VideoViewer(QWidget):
    def __init__(self, project_model: ProjectModel, parent=None):
        super().__init__(parent)
        if parent is None:
            self.setWindowTitle("Video Zoom Tool")

        self.model = project_model
        self.model.video_selected.connect(self.load_video_from_model)
        self.current_video = None

        self.video_path = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame_timer)
        self.paused = False

        # Buffer for current frame (RGB)
        self.current_rgb_frame = None

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
        self.play_button = QPushButton("⏯ Play / Pause")
        self.rewind_button = QPushButton("⏪ Rewind")
        self.forward_button = QPushButton("⏩ Forward")
        self.zoom_in_button = QPushButton("➕ Zoom In")
        self.zoom_out_button = QPushButton("➖ Zoom Out")

        self.play_button.clicked.connect(self.toggle_play)
        self.rewind_button.clicked.connect(self.rewind_video)
        self.forward_button.clicked.connect(self.forward_video)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)

        self.zoom_in_button.setEnabled(False)
        self.zoom_out_button.setEnabled(False)

        # Layouts
        button_layout = QHBoxLayout()
        for btn in (self.play_button, self.rewind_button, self.forward_button,
                    self.zoom_in_button, self.zoom_out_button):
            button_layout.addWidget(btn)

        layout = QVBoxLayout()
        layout.addWidget(self.splitter)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.zoombox = None  # will be initialized on video open
        self.dragging = False
        self.drag_start_pos = None  # initial mouse position
        self.drag_start_box = None

    def load_video_from_model(self, video_data):
        self.current_video = video_data
        self.load_video(video_data["path"])

    def load_video(self, path):
        if self.cap:
            self.cap.release()
        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)
        self.current_rgb_frame = None
        # start timer for playback (will call update_frame_timer)
        self.timer.start(30)

        self.read_one_frame()
        if self.current_rgb_frame is None:
            return
        
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

        self.render_frames()

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
        else:
            # pause playback
            self.paused = True
            self.timer.stop()

    def rewind_video(self):
        """Go back 1 second (adjustable)."""
        seconds = 1
        if not self.cap:
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_frame = max(current_frame - int(seconds * fps), 0)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        # read one frame immediately to update buffer/render
        self.read_one_frame_and_render()

    def forward_video(self):
        """Go forward 1 second (adjustable)."""
        seconds = 1
        if not self.cap:
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_frame = min(current_frame + int(seconds * fps), max(total_frames - 1, 0))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.read_one_frame_and_render()

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
        if not self.zoombox:
            return
        self.dragging = True
        self.drag_start_pos = pos
        self.drag_start_box = (self.zoombox.x, self.zoombox.y)

    def update_drag(self, pos: QPointF):
        if not self.dragging or not self.zoombox:
            return

        # Compute mouse delta in label coordinates
        dx = pos.x() - self.drag_start_pos.x()
        dy = pos.y() - self.drag_start_pos.y()

        # Scale delta to frame coordinates
        label_w, label_h = self.full_view.width(), self.full_view.height()
        frame_h, frame_w, _ = self.current_rgb_frame.shape
        scale_x = frame_w / label_w
        scale_y = frame_h / label_h
        dx_frame = dx * scale_x
        dy_frame = dy * scale_y

        # Move zoombox, clamped to frame boundaries
        new_x = min(max(0, self.drag_start_box[0] + dx_frame), frame_w - self.zoombox.width)
        new_y = min(max(0, self.drag_start_box[1] + dy_frame), frame_h - self.zoombox.height)
        self.zoombox.x = new_x
        self.zoombox.y = new_y

        # Update display
        self.render_frames()

    def end_drag(self):
        self.dragging = False
        self.drag_start_pos = None
        self.drag_start_box = None

    def read_one_frame(self):
        """Read a single frame from cap to update the buffer."""
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read_one_frame_and_render(self):
        """Read a single frame from cap to update the buffer and render."""
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            # stop timer and keep last frame (if any)
            self.timer.stop()
            return
        self.current_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.render_frames()

    def update_frame_timer(self):
        """Called on every timer tick when playing (advances playback)."""
        if not self.cap or self.paused:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
        self.current_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.render_frames()

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

        rgb_frame = self.current_rgb_frame.copy()
        h, w, _ = rgb_frame.shape
        if h == 0 or w == 0:
            return

        # Get zoombox coordinates
        if self.zoombox:
            x1, y1, x2, y2 = self.zoombox.get_coordinates()

        # Draw rectangle on full view to indicate zoomed area
        cv2.rectangle(
            rgb_frame,           # target image
            (x1, y1), (x2, y2),  # top-left, bottom-right
            color=(255, 0, 0),   # red box (RGB)
            thickness=2,         # thickness in pixels
        )

        if self.current_video:
            for box in self.current_video["boxes"]:
                x1, y1 = int(box["x"]), int(box["y"])
                x2, y2 = x1 + int(box["width"]), y1 + int(box["height"])
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # If the crop would be invalid for any reason, fall back to whole frame
        if x2 <= x1 or y2 <= y1:
            zoomed = rgb_frame
        else:
            zoomed = rgb_frame[y1:y2, x1:x2]
            # resize zoomed region up to original frame size for quality before final label scaling
            zoomed = cv2.resize(zoomed, (w, h), interpolation=cv2.INTER_CUBIC)

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


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     viewer = VideoViewer()
#     viewer.resize(1800, 900)
#     viewer.show()
#     sys.exit(app.exec())