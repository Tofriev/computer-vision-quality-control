from pathlib import Path
import torch


class Config:

    BASE_DIR = Path(__file__).resolve().parent.parent

    TEST_SIZE = 0.2  # this var is used for both splitting the (training, val) and test as well as splitting training and val
    BATCH_SIZE = 4
    RESIZE_TO = 512
    NUM_EPOCHS = 50
    RANDOM_STATE = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = BASE_DIR / "data"
    CLASSES = ["background", "CamFront"]
    NUM_CLASSES = 2
    VISUALIZE_TRANSFORMED_IMAGES = True
    OUT_DIR = BASE_DIR / "data" / "output_test"
    SAVE_PLOTS_EPOCH = 100
    SAVE_MODEL_EPOCH = 100
