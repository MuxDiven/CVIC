import gc
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms.v2 as transforms
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models"
PROCESSED_PATH = BASE_DIR / "data" / "processed"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hard_reset_environment(seed):
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Enforce absolute global determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = (
        False  # This can be turned to true incase anything breaks (NaNs,not learning)
    )
    torch.backends.cudnn.benchmark = False


class ImageDataset(Dataset):
    def __init__(self, dataset_name="PetImages", split="train"):
        images_path = PROCESSED_PATH / f"{dataset_name}_images.npy"
        metadata_path = PROCESSED_PATH / f"{dataset_name}_metadata.npz"

        images = np.load(
            images_path, mmap_mode="r"
        )  # image datasets are large, use memmap to poll into memory only when required
        metadata = np.load(metadata_path)

        self.transform = (
            transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                ]
            )
            if device.type != "cpu"
            else None
        )  # without a gpu it might take a while...

        self.images = images
        self.labels = metadata["labels"]
        self.classes = metadata["classes"].tolist()
        self.n_classes = len(self.classes)

        # indices are sorted for faster localization during memory/storage operations
        if split == "train":
            self.indices = np.sort(metadata["train_idx"])
        else:
            self.indices = np.sort(metadata["test_idx"])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]

        img = torch.from_numpy(self.images[idx])
        label = int(self.labels[idx])

        if self.transform:
            img = self.transform(img)

        return img, label


class ImageNet(nn.Module):  # using type annotations for torch JIT compile
    classes: List[str]

    def __init__(self, classes_list: List[str]):
        super().__init__()
        self.classes = classes_list  # distribution ordered alphabetically
        n_classes = len(classes_list)

        self.trunk = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # 224 -> 112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # 56  -> 28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # 28  -> 14
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # 14  ->  7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), *self._build_classifier(n_classes)
        )

        self.apply(self._init_weights)

    @torch.jit.export
    def n_classes(self) -> int:
        return len(self.classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        logits = self.classifier(features)
        return logits

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    @torch.jit.ignore
    def _build_classifier(self, n_classes: int) -> List[nn.Module]:
        if n_classes < 10:
            return [
                nn.Linear(256 * 7 * 7, 512),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.5),
                nn.Linear(512, n_classes),
            ]
        elif n_classes < 100:
            return [
                nn.Linear(256 * 7 * 7, 1024),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.5),
                nn.Linear(1024, n_classes),
            ]
        else:
            return [
                nn.Linear(256 * 7 * 7, 1024),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.5),
                nn.Linear(512, n_classes),
            ]

    @torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


def fit(model, device, train_dataset, epochs=2, log_callback=print):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.4)

    seq_sampler = SequentialSampler(train_dataset)
    batch_sampler = BatchSampler(seq_sampler, batch_size=32, drop_last=False)
    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler, num_workers=0, pin_memory=True
    )
    scaler = torch.amp.GradScaler("cuda")

    start = time.time()
    for i in range(epochs):
        total_loss = 0
        model.train()

        for image, label in train_loader:
            image, label = (
                image.to(device),
                label.to(device),
            )
            optim.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                logits = model(image)
                loss = criterion(logits, label)

            scaler.scale(loss).backward()
            total_loss += loss.item()
            scaler.step(optim)
            scaler.update()

        log_callback(
            f"epoch {i+1} / {epochs}: total_loss/n = {total_loss/len(train_loader)} : lr: {scheduler.get_last_lr()[0]:.6f}"
        )
        scheduler.step()

    log_callback(f"training took {(time.time() - start):.1f}s")


def evaluate(model, device, test_dataset):
    model.eval()
    test_loader = DataLoader(
        test_dataset, batch_size=32, num_workers=0, pin_memory=True
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for image, label in test_loader:
            image, label = (
                image.to(device),
                label.to(device),
            )
            with torch.amp.autocast(device_type="cuda"):
                output = model(image)

            prediction = output.argmax(dim=1)
            correct += (prediction == label).sum().item()
            total += label.size(0)

    return correct / total


def model_construction(dataset_name, epochs, seed, save=True, log_callback=print):
    hard_reset_environment(seed)
    start = time.time()
    log_callback("reading in data")
    train_ds = ImageDataset(dataset_name, "train")
    test_ds = ImageDataset(dataset_name, "test")

    log_callback("constructing model")
    model = ImageNet(train_ds.classes)
    model.to(device)

    log_callback("training model")
    fit(model, device, train_ds, epochs, log_callback)

    log_callback("testing model")
    results = evaluate(model, device, test_ds)
    log_callback(f"total construction took {(time.time() - start):.1f}s")
    log_callback(f"test accuracy was: {results * 100}%")

    if save:
        log_callback("saving model as .pt object")
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, MODEL_PATH / f"{dataset_name}.pt")

    # returning Eval(model)
    return model


# behaves as a headless train/test: ONLY WORKS WITH CUDA
if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU name:", torch.cuda.get_device_name(0))

    EPOCHS = 25
    SEED = 42
    DATASET_NAME = (
        "this should match the name of your processed dataset without file extensions"
    )
    model_construction(DATASET_NAME, EPOCHS, SEED)

