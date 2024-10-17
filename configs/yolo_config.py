from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import yaml

class YOLOConfig(BaseModel):  # type: ignore
    hyperparameters: dict[str, str | float | bool] = Field(
    {
        "model_name": 'yolov8n',
        "epochs": 150,
        "batch": 16,
        "imgsz": 640,
        "save_period": 5,
        "mixup": 0.3,
        "device": 'cpu',  # TODO: "mpl" is slower - do not understand why
        "mosaic": 1.0,
    },
    )

    @classmethod
    def load_from_yaml(cls, file_path: str | Path) -> "YOLOConfig":
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(hyperparameters=config["hyperparameters"])
