from pydantic import BaseModel, field_validator

class VideoObject(BaseModel):
    video_path: str
    
    def __str__(self):
        return str(self.model_dump())

class Video(VideoObject):
    setting: int | None = None
    
    def to_dict(self):
        return {
            "video_path": self.video_path,
            "setting": self.setting,
        }

class Distortion(VideoObject):
    name: str
    x: int
    y: int
    size: int
    setting: int | None = None

    predicted_label: int | None  = None  # To be set later
    ground_truth_label: int | None = None  # To be set later

    @field_validator('x', 'y', 'size', mode='before')
    @classmethod
    def round_floats(cls, v):
        if isinstance(v, float):
            return round(v)  # or int(v) for truncation, math.floor(v), etc.
        return v

    def to_dict(self):
        return {
            "name": self.name,
            "video_path": self.video_path,
            "setting": self.setting,
            "x": self.x,
            "y": self.y,
            "size": self.size,
            "predicted_label": self.predicted_label,
            "ground_truth_label": self.ground_truth_label,
        }

    # def __iter__(self):
    #     yield from (self.video_path, self.setting, self.x, self.y, self.size, self.predicted_label, self.ground_truth_label)