import os
import tempfile
import numpy as  np
import torchvision
from tqdm.auto import tqdm
import cv2

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


def main():
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

        print("dumping images...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            idx = idx + 1
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{idx:05d}.png")
            image.save(filename)


def imgs_to_npz():
    npz = []

    for img in os.listdir("./cifar_train"):
        img_arr = cv2.imread("./cifar_train/" + img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)  # cv2默认为 bgr 顺序
        npz.append(img_arr)

    output_npz = np.array(npz)
    np.savez('cifar10_train.npz', output_npz)
    print(f"{output_npz.shape} size array saved into cifar10_train.npz")  # (50000, 32, 32, 3)


if __name__ == "__main__":
    main()
    # imgs_to_npz()