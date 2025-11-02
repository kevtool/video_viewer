class Box():
    def __init__(self, x=0, y=0, width=100, height=100):
        self.x = x      # top-left corner x
        self.y = y      # top-left corner y
        self.width = width      # box width
        self.height = height   # box height

    def get_coordinates(self):
        """Return (x1, y1, x2, y2) as integers."""
        x1 = int(self.x)
        y1 = int(self.y)
        x2 = int(self.x + self.width)
        y2 = int(self.y + self.height)
        return x1, y1, x2, y2
    
class DistortionBox(Box):
    def __init__(self, x=0, y=0, width=128, height=128):
        super().__init__(x, y, width, height)



class Zoombox(Box):
    def __init__(self, zoom_factor=2, x=0, y=0, width=200, height=150,
                 original_width=None, original_height=None):
        super().__init__(x, y, width, height)
        self.zoom_factor = zoom_factor
        self.original_width = original_width
        self.original_height = original_height
        self.max_zoom = 7
        self.min_zoom = 1

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor + 1, self.max_zoom)  # max zoom factor
        self._update_size()

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor - 1, self.min_zoom)  # min zoom factor
        self._update_size()

    def _update_size(self):
        """Update width/height based on zoom_factor while keeping center fixed."""
        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2

        # New size inversely proportional to zoom factor
        new_width = max(10, int(self.original_width / self.zoom_factor))
        new_height = max(10, int(self.original_height / self.zoom_factor))

        # Update top-left to keep center same
        self.x = min(max(0, center_x - new_width // 2), self.original_width - new_width)
        self.y = min(max(0, center_y - new_height // 2), self.original_height - new_height)
        self.width = new_width
        self.height = new_height