import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats


def show_images():
    x = np.load('/m100_work/FF4_Axyon/DDPM/guided_diffusion/evaluations/celeba_noise_0.1_320k_900steps_50000_samples.npz')['arr_0']
    plt.figure(figsize=(20, 20))
    for i in range(36):  # 8*8 or 6*6
        img = x[i+64, :, :, :]
        plt.subplot(6, 6, i + 1)
        plt.imshow(img)
        plt.axis('off')
    # plt.savefig('./imgnet32_samples_4.jpg')
    fig = plt.gcf()
    fig.tight_layout()
    plt.show()


def gaussian_histogram(t):
    x = np.load('/m100_work/FF4_Axyon/DDPM/guided_diffusion/imagenet32_base_gaussian_error/gaussian_error_xstart_all_pixels.npz')['arr_0']
    print(f"array shape: {x.shape}" )  # (100, batch, 3, 32, 32)

    data = x[t, :, 2, 16, 16]
    mean = np.mean(data)
    std = np.std(data)

    plt.figure()
    plt.hist(data, bins=40, color=(31/255, 119/255, 180/255))
    plt.xlabel("$e_{t}^{i}$", size=14)
    plt.ylabel("frequency", size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)

    # plt.title(f'prediction error at timestep {t+1}')
    # plt.text(0, 200, f"mean={mean: .6f}", horizontalalignment="center", color=(255/255, 127/255, 14/255), size=14)
    # plt.text(0, 100, f"std={std: .4f}", horizontalalignment="center", color=(255/255, 127/255, 14/255), size=14)
    fig = plt.gcf()
    fig.tight_layout()
    plt.show()


def shapiro_wilk_test(t):
    x = np.load('/m100_work/FF4_Axyon/DDPM/guided_diffusion/imagenet32_base_gaussian_error/gaussian_error_xstart_all_pixels.npz')['arr_0']
    print(f"array shape: {x.shape}")  # (100, batch, 3, 32, 32)

    time, batch, channel, height, width = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]

    # use the expectation of mean and std to do normalization
    reject_rates = []
    for t in range(0, time):
        print(f"compute mean and std for timestep: {t}: ")
        means = []
        stds = []
        for c in range(0, channel):
            for h in range (0, height):
                for w in range(0, width):
                    data = x[t, :, c, h, w]
                    means.append(np.mean(data))
                    stds.append(np.std(data))
        mean = np.array(means).mean()
        std = np.array(stds).mean()
        print(f"mean: {mean}")
        print(f"std: {std}")

        cnt = 0
        reject_cnt = 0
        for c in range(0, channel):
            for h in range (0, height):
                for w in range(0, width):
                    data = x[t, np.random.choice(batch, size=100, replace=False), c, h, w]
                    norm_data = (data - mean) / std
                    # results = stats.kstest(rvs=norm_data, cdf='norm', alternative='two-sided')
                    results = stats.shapiro(norm_data)
                    cnt += 1
                    if results.pvalue < 0.05:
                        reject_cnt += 1
                        # print(results.pvalue)
        reject_rate = reject_cnt / cnt
        reject_rates.append(reject_rate)
        print(f"reject rate: {reject_cnt} / {cnt} = {reject_rate}")

    print(f"average raject rate: {np.array(reject_rates).mean()}")


def gaussian_error_std_for_each_pixel():
    x_base = np.load('/m100_work/FF4_Axyon/DDPM/guided_diffusion/imagenet32_base_gaussian_error/gaussian_error_xstart_all_pixels.npz')['arr_0']
    print(f"array shape: {x_base.shape}" )  # (100, batch, 3, 32, 32)

    data = x_base[90]  # (batch, 3, 32, 32)
    print(f"data shape: {data.shape}")
    channels, width, height = x_base.shape[2], x_base.shape[3], x_base.shape[4]
    std_by_pixel = []
    for i in range(0, channels):
        for j in range(0, width):
            for k in range(0, height):
                std = np.std(data[:, i, j, k], axis=0)  # compute std of the gaussian error along the batch
                std_by_pixel.append(std)
    print(f"number of pixels: {len(std_by_pixel)}")
    avg_std = np.array(std_by_pixel).mean()
    print(f"std mean of all pixels: {avg_std}")
    # print(f"mean: {np.array(std_pixel).mean()}")

    x = [i for i in range(1, len(std_by_pixel)+1)]
    plt.plot(x, std_by_pixel, 'r--', label='std at each pixel')
    plt.legend()
    plt.xlabel('pixel')
    plt.ylabel('std')
    plt.show()


def gaussian_error_std_along_timesteps():
    x_base = np.load('/m100_work/FF4_Axyon/DDPM/guided_diffusion/imagenet32_base_gaussian_error/gaussian_error_xstart_all_pixels.npz')['arr_0']
    print(f"array shape: {x_base.shape}" )  # (100, batch, 3, 32, 32)

    timesteps, channels, width, height = x_base.shape[0], x_base.shape[2], x_base.shape[3], x_base.shape[4]
    std_along_t_1 = []
    std_over_pixels = []
    for t in range(0, timesteps):
        data = x_base[t]
        print(f"data shape: {data.shape}")
        for i in range(0, channels):
            for j in range(0, width):
                for k in range(0, height):
                    std_pixel = np.std(data[:, i, j, k], axis=0)
                    std_over_pixels.append(std_pixel)
        std_of_each_t = np.array(std_over_pixels).mean()
        print(f"std mean of all pixels: {std_of_each_t} at timestep {t}" )
        std_along_t_1.append(std_of_each_t)
        std_over_pixels = []
    print(f"std mean of all t: {np.array(std_along_t_1).mean()}")


    x_base = np.load('/m100_work/FF4_Axyon/DDPM/guided_diffusion/cifar_base_gaussian_error/gaussian_error_xstart_all_pixels.npz')['arr_0']
    print(f"array shape: {x_base.shape}")  # (100, batch, 3, 32, 32)

    timesteps, channels, width, height = x_base.shape[0], x_base.shape[2], x_base.shape[3], x_base.shape[4]
    std_along_t_2 = []
    std_over_pixels_2 = []
    for t in range(0, timesteps):
        data = x_base[t]
        print(f"data shape: {data.shape}")
        for i in range(0, channels):
            for j in range(0, width):
                for k in range(0, height):
                    std_pixel = np.std(data[:, i, j, k], axis=0)
                    std_over_pixels_2.append(std_pixel)
        std_of_each_t = np.array(std_over_pixels_2).mean()
        print(f"std mean of all pixels: {std_of_each_t} at timestep {t}")
        std_along_t_2.append(std_of_each_t)
        std_over_pixels_2 = []
    print(f"std mean of all t: {np.array(std_along_t_2).mean()}")


    x = [i for i in range(10, 1001, 10)]
    print(x)
    print(f"imagenet: {std_along_t_1}")
    print(f"cifar10: {std_along_t_2}")
    plt.plot(x, std_along_t_2, color=(31 / 255, 119 / 255, 180 / 255), label='Cifar10 32x32')
    plt.plot(x, std_along_t_1, color=(255/255, 127/255, 14/255), label='ImageNet 32x32')
    plt.legend(prop = {'size':16})
    plt.xlabel('timesteps', size=16)
    plt.ylabel('$\\nu_{t}$', size=16)
    plt.xticks(size=16)
    plt.yticks(size=16)
    fig = plt.gcf()
    fig.tight_layout()
    plt.show()
    # plt.savefig('gaussian_error.eps', dpi=600, format='eps')


def FID_results():
    # # ImageNet
    # x = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    # ADM =    [5.59, 4.73, 4.16, 3.94, 3.78, 3.71, 3.6,  3.59, 3.53, 3.58]
    # ADM_IP = [4.69, 3.83, 3.31, 3.2,  3.05, 3,    2.87, 2.72, 2.76, None]

    # # LSUN tower
    # x = [60, 100, 140, 180, 220, 260, 300, 340]
    # ADM =    [8.45, 7.73, 6.87, 5.57, 5.06, 3.83, 3.39, 4.15]
    # ADM_IP = [4.45, 3.18, 2.79, 2.79, 2.68, 2.76, None, None]

    # # CelebA
    # x = [60, 120, 180, 240, 300, 360, 420, 480, 540]
    # ADM =    [3.19, 2.34, 2.06, 1.88, 1.78, 1.78, 1.63, 1.60, 1.66]
    # ADM_IP = [2.81, 1.51, 1.39, 1.38, 1.31, 1.33, 1.40, None, None]

    # FFHQ
    x = [60, 120, 180, 240, 300, 360, 420, 480]
    ADM =    [24.94, 18.69, 15.94, 15.25, 14.68, 14.68, 14.52, 14.84]
    ADM_IP = [8.81,  6.58,  5.94,  6.09,  6.14,  None,  None,  None]

    # Plot the data
    plt.plot(x, ADM_IP, linewidth=2, label='ADM-IP', color='indianred', alpha=0.9, marker='^', markersize=5, linestyle='-',
            markerfacecolor='mistyrose', markeredgecolor='indianred')
    plt.plot(x, ADM, linewidth=2, label='ADM', color='forestgreen', alpha=0.9, marker='o', markersize=5, linestyle='-',
            markerfacecolor='honeydew', markeredgecolor='forestgreen')

    # Customize the grid
    plt.grid(axis='x', linestyle='--', linewidth=0.5, color='darkgray')
    plt.grid(axis='y', linestyle='-', linewidth=0.5, color='darkgray')

    # remove right and top edges
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.ylabel("FID", fontsize=15)
    plt.xlabel("Training iterations (thousand)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper right', fontsize=15)

    # Show the plot
    plt.savefig('ffhq.pdf', bbox_inches='tight')
    plt.show()


def FID_cifar():
    # CIFAR10
    x = [140, 180, 220, 260, 300, 340, 380, 420, 460, 500, 540]
    ADM =        [4.42, 3.92, 3.86, 3.7,  3.54, 3.48, 3.41, 3.41, 3.43, 3.37, 3.44]
    ADM_IP_005 = [3.94, 3.57, 3.46, 3.3,  3.25, 3.25, 3.18, 3.05, 3.03, 3.05, None]
    ADM_IP_010 = [3.44, 3.16, 3.03, 2.98, 2.91, 2.85, 2.84, 2.79, 2.7,  2.74, None]
    ADM_IP_020 = [3.65, 3.43, 3.27, 3.16, 3.05, 2.98, 2.94, 2.89, 2.82, 2.88, None]


    # Plot the data
    plt.plot(x, ADM, linewidth=2, label='ADM', color='forestgreen', alpha=0.9, marker='o', markersize=5, linestyle='-',
            markerfacecolor='honeydew', markeredgecolor='forestgreen')
    plt.plot(x, ADM_IP_005, linewidth=2, label='ADM-IP ($\gamma=$0.05)', color='indianred', alpha=0.4, marker='x', markersize=5,
             linestyle='-',
             markerfacecolor='mistyrose', markeredgecolor='indianred')
    plt.plot(x, ADM_IP_010, linewidth=2, label='ADM-IP ($\gamma=$0.1)', color='indianred', alpha=1.0, marker='^', markersize=5,
             linestyle='-',
             markerfacecolor='mistyrose', markeredgecolor='indianred')
    plt.plot(x, ADM_IP_020, linewidth=2, label='ADM-IP ($\gamma=$0.2)', color='indianred', alpha=0.4, marker='s', markersize=5,
             linestyle='-',
             markerfacecolor='mistyrose', markeredgecolor='indianred')

    # Customize the grid
    plt.grid(axis='x', linestyle='--', linewidth=0.5, color='darkgray')
    plt.grid(axis='y', linestyle='-', linewidth=0.5, color='darkgray')

    # remove right and top edges
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.ylabel("FID", fontsize=15)
    plt.xlabel("Training iterations (thousand)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper right', fontsize=15)

    # Show the plot
    plt.savefig('cifar10.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # show_images()
    # gaussian_histogram(60)
    # shapiro_wilk_test(1)
    # gaussian_error_std_for_each_pixel()
    # gaussian_error_std_along_timesteps()
    FID_results()
    # FID_cifar()