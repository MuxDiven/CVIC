import sys
import random
import os
import torch
from pathlib import Path
from model import model_construction
from classifier_ui import App, _label_event

BASE_DIR = Path(__file__).resolve().parent.parent 
MODEL_PATH = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data"
IMAGE_PATH = DATA_PATH / "test-images"
DATASET_PATH = DATA_PATH / "processed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    if "--headless" in sys.argv:  #makes training slightly faster as it eliminates thread context switches
        dataset = sys.argv[2]
        if "\\" not in dataset or "/" not in dataset:
            dataset_path = DATASET_PATH / f"{dataset}"
        else:
            dataset_path = Path(dataset)

        epochs = int(sys.argv[3]) if len(sys.argv) >= 4 else 20
        seed = int(sys.argv[4]) if len(sys.argv) == 5 else random.randint(0,2**32 - 1)

        _label_event(dataset_path,print)
        model = model_construction(dataset_path.name,epochs,seed,save=True).to(device)
        app = App(entry_point=__file__,mem_flag=True)
        app.current_model = model
        app.mainloop()
        os._exit(0)

    else:
        app = App(entry_point=__file__)
        app.mainloop()
