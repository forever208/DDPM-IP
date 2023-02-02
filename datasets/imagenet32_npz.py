import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def merge_all_train_dataset():
    output = None
    for i in range(1, 11):
        filename = './ImageNet64/train_data_batch_' + str(i) + '.npz'
        raw_data = np.load(filename)
        print(f"{filename} loaded")
        labels = list(raw_data['labels'])
        print(f"{raw_data['data'].shape[0]} images founded")

        # reshape data into 4D array (num_images, 64, 64, 3)
        x = raw_data['data']
        x = np.dstack((x[:, :1024], x[:, 1024:2 * 1024], x[:, 2 * 1024:]))
        x = x.reshape((x.shape[0], 32, 32, 3))

        if output is None:
            output = x
        elif output.any():
            output = np.concatenate((output, x), axis=0)

    output_array = np.array(output)
    np.savez('ImageNet32_train_all.npz', output_array)
    print(f"{output_array.shape} size array saved into ImageNet32_train_all.npz")


def show_partial_samples():
    x = np.load('./ImageNet32/ImageNet32_train_all.npz')['arr_0']
    img_arr = x[:16, :, :, :]
    np.savez('ImageNet32_samples.npz', img_arr)
    print(f"{img_arr.shape} size array saved into ImageNet32_samples.npz")


def visualize_samples():
    x = np.load('./ImageNet32/ImageNet32_samples.npz')['arr_0']
    plt.figure(figsize=(20, 10))
    for i in range(32):
        img = x[i, :, :, :]
        plt.subplot(8, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


def check_if_data_clean():
    x = np.load('./ImageNet32/ImageNet32_train_all.npz')['arr_0']
    for i in tqdm(range(x.shape[0])):
        img = x[i]
        img = img.astype(np.float32) / 127.5 -1

        if ((img>=-1) & (img<=1)).all():
            pass
        else:
            print(img)


if __name__ == '__main__':
    merge_all_train_dataset()
    # visualize_samples()
    # show_partial_samples()
    # check_if_data_clean()