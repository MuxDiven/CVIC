import random
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent / "data"
WRITE_DIR = BASE_DIR / "processed"
IMAGE_DIM = (224, 224)


def img_squash(img):  # mutate img into a tensor form understood by the network
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMAGE_DIM)
    norm_img = img_resized.astype("float32") / 255.0
    return np.transpose(norm_img, (2, 0, 1))


def label_dataset(
    path, rand_n=42, log_callback=print
):  # use magic number for random state when testing for replicatable behaviour
    dataset_name = path.name
    classes = [f.name for f in sorted(path.iterdir()) if f.is_dir()]
    class_to_label = {label: i for i, label in enumerate(classes)}
    labels = []  # softmax indeces

    random.seed(rand_n)
    all_of_the_files = list(path.rglob("*"))
    random.shuffle(all_of_the_files)

    dat_path = (
        WRITE_DIR / f"{dataset_name}.raw"
    )  # processed tensors will be to big for memeory, write raw bytes to disk
    real_count = 0

    log_callback("Processing raw image data")
    Path(dat_path).parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(dat_path, "wb") as f:
        for file_path in all_of_the_files:
            folder = file_path.parent.name
            file_name = file_path.name

            img = cv2.imread(path / folder / file_name)
            if img is not None:
                norm_img = img_squash(img)
            else:
                continue

            real_count += 1
            f.write(norm_img.tobytes())
            labels.append(class_to_label[folder])

    log_callback(f"Processed {real_count} images")

    data_shape = (real_count, 3, *IMAGE_DIM)
    data = np.memmap(
        dat_path, dtype="float32", mode="r", shape=data_shape
    )  # construct refference to memmap
    labels = np.array(labels, dtype=np.int32)

    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=rand_n, stratify=labels
    )

    log_callback("Writing tensor dataset to file")
    np.save(WRITE_DIR / f"{dataset_name}_images.npy", data)
    np.savez(
        WRITE_DIR / f"{dataset_name}_metadata.npz",
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
        classes=classes,
    )

    log_callback(f"time taken: ~{int(time.time() - start)}s")

    del data
    if dat_path.exists():
        dat_path.unlink()


if __name__ == "__main__":
    label_dataset()

