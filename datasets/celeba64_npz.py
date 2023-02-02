import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def imgs_to_npz():
    npz = []

    for img in os.listdir("./img_align_celeba"):
        img_arr = cv2.imread("./img_align_celeba/" + img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)  # bgr default in cv2
        resized_img = cv2.resize(img_arr, (64, 64))
        npz.append(resized_img)

    output_npz = np.array(npz)
    np.savez('celeba64_train.npz', output_npz)
    print(f"{output_npz.shape} size array saved into celeba64_train.npz")  # (202599, 64, 64, 3)


def show_images():
    x = np.load('./celeba64/celeba64_train.npz')['arr_0']
    plt.figure(figsize=(10, 10))
    for i in range(16):
        img = x[i, :, :, :]
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')
    # plt.savefig('./imgnet32_samples_4.jpg')
    plt.show()


if __name__ == '__main__':
    imgs_to_npz()
    show_images()