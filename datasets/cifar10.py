import os
import tempfile

import torchvision
from tqdm.auto import tqdm

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def download_cifar10():
    idx = 0
    for split in ["train", "test"]:
        out_dir = f"cifar_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.CIFAR10(
                root=tmp_dir, train=split == "train", download=True
            )

        print("dumping images into current dir...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            idx = idx + 1
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{idx:05d}.png")
            image.save(filename)


if __name__ == "__main__":
    download_cifar10()