import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


def load_npy(file_path):
    return np.load(file_path)


def save_npy(data, file_path):
    np.save(file_path, data)


def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # or other formats
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images


def augment_image(image):
    augmented_images = []
    # Original image and its mirrored version
    augmented_images.append(image)
    augmented_images.append(np.fliplr(image))
    # Rotations and their mirrored versions
    for k in range(1, 4):
        rotated = np.rot90(image, k)
        augmented_images.append(rotated)
        augmented_images.append(np.fliplr(rotated))
    return augmented_images


def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = image + noise
    return noisy_image


def random_shift(image, shift_h, shift_w):
    shifted_image = np.roll(image, shift_h, axis=0)
    shifted_image = np.roll(shifted_image, shift_w, axis=1)
    return shifted_image


def extend_dataset_with_shifts_and_augmentations(original_images, sigma):
    extended_data = []
    true_shifts_list = []
    original_noisy_non_shifted = []
    # for i in range(len(original_data)):
    #     clean_img = original_data[i, 0, :, :, 0]
    #     noisy_img = original_data[i, 1, :, :, 0]
    # #
    # #     # Add the original augmented images to the extended data
    #     augmented_clean_images = augment_image(clean_img)
    #     augmented_noisy_images = augment_image(noisy_img)
    #     for clean_img_aug, noisy_img_aug in zip(augmented_clean_images, augmented_noisy_images):
    #         extended_data.append((clean_img_aug[:, :, np.newaxis], noisy_img_aug[:, :, np.newaxis], noisy_img_aug[:, :, np.newaxis]))
    #         true_shifts_list.append((0, 0))

    for img in original_images:
        new_noisy_img = add_gaussian_noise(img, sigma)
        new_noisy_img2 = add_gaussian_noise(img, sigma)

        # Generate random shifts
        h, w = img.shape
        max_shift_h = int(h * 0.25)
        max_shift_w = int(w * 0.25)
        shift_h = np.random.randint(0, max_shift_h + 1)
        shift_w = np.random.randint(0, max_shift_w + 1)
        # shifted_clean_img = random_shift(new_noisy_img2, shift_h, shift_w)
        shifted_noisy_img = random_shift(new_noisy_img2, shift_h, shift_w)

        # Add the shifted augmented images to the extended data
        # augmented_clean_images = augment_image(img)
        # augmented_noisy_images = augment_image(new_noisy_img)
        # augmented_noisy_images2 = augment_image(new_noisy_img2)
        # augmented_noisy_images_shifted = augment_image(shifted_noisy_img)
        # for clean_img_aug, noisy_img_aug, noisy_img_aug2, noisy_img_shifted_aug in zip(
        #         augmented_clean_images,
        #         augmented_noisy_images,
        #         augmented_noisy_images2,
        #         augmented_noisy_images_shifted):
        # for clean_img_shifted_aug, noisy_img_aug, noisy_img_aug2 in zip(
        #         clean_images,
        #         augmented_noisy_images,
        #         augmented_noisy_images2
        # ):
        # extended_data.append(
        #     (
        #         clean_img_aug[:, :, np.newaxis],
        #         noisy_img_aug[:, :, np.newaxis],
        #         noisy_img_aug2[:, :, np.newaxis],
        #         noisy_img_shifted_aug[:, :, np.newaxis]
        #     )
        # )
        extended_data.append(
            (
                img[:, :, np.newaxis],
                new_noisy_img[:, :, np.newaxis],
                new_noisy_img2[:, :, np.newaxis],
                shifted_noisy_img[:, :, np.newaxis]
            )
        )
        true_shifts_list.append((shift_h, shift_w))

    return np.array(extended_data), true_shifts_list


def dft_registration(img1, img2):
    F1 = np.fft.fft2(img1)
    F2 = np.fft.fft2(img2)
    R = F1 * np.conj(F2)
    # R /= np.abs(R)
    r = np.fft.fftshift(np.fft.ifft2(R))
    max_idx = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    shifts = np.array(max_idx) - np.array(r.shape) // 2
    return shifts


def apply_shifts(image, shifts):
    shifted_image = np.roll(image, -shifts[0], axis=0)
    shifted_image = np.roll(shifted_image, -shifts[1], axis=1)
    return shifted_image


# original_npy_file = '/opt/KAIR/data/BSD68_reproducibility_data/train/DCNN400_train_gaussian25_pairs.npy'
# original_data = load_npy(original_npy_file)

clean_images_directory = '/opt/KAIR/data/BSD68_reproducibility_data/train/gt'
original_clean_images = load_images_from_directory(clean_images_directory)

extended_data, true_shifts_list = extend_dataset_with_shifts_and_augmentations(original_clean_images, 25)

new_npy = []
all_npy = []
sum_of_differences = 0
cnt_diff = 0
for idx, triplet in enumerate(extended_data):
    clean_img = triplet[0, :, :, 0]
    noisy_img = triplet[1, :, :, 0]
    noisy_img2 = triplet[2, :, :, 0]
    shifted_noisy_img = triplet[3, :, :, 0]
    # true_shifts = true_shifts_list[idx]

    computed_shifts = dft_registration(noisy_img, shifted_noisy_img)
    aligned_noisy_img = apply_shifts(shifted_noisy_img, -computed_shifts)
    new_npy.append((clean_img[:, :, np.newaxis], noisy_img[:, :, np.newaxis],
                    aligned_noisy_img[:, :, np.newaxis]))
    all_npy.append(
        (
            clean_img[:, :, np.newaxis],
            noisy_img[:, :, np.newaxis],
            noisy_img2[:, :, np.newaxis],
            shifted_noisy_img[:, :, np.newaxis],
            aligned_noisy_img[:, :, np.newaxis],
        )
    )

    # new_npy.append((clean_img[:, :, np.newaxis], noisy_img[:, :, np.newaxis],
    #                 noisy_img2[:, :, np.newaxis]))
    #
    difference = np.sum(np.abs(noisy_img2 - aligned_noisy_img))
    if difference > 0:
        cnt_diff += 1
        # print(
        #     f"For idx {idx} - True Shifts: {true_shifts}, Computed Shifts: {-computed_shifts}")  # , Difference: {difference}")

        # fig, ax = plt.subplots(1, 5, figsize=(20, 5))
        # ax[0].imshow(clean_img, cmap='gray')
        # ax[0].set_title('Clean Image')
        # ax[1].imshow(noisy_img, cmap='gray')
        # ax[1].set_title('First Noisy Image')
        # ax[2].imshow(noisy_img2, cmap='gray')
        # ax[2].set_title('Second Noisy Image')
        # ax[3].imshow(shifted_noisy_img, cmap='gray')
        # ax[3].set_title('Second Noisy Image Shifted')
        # ax[4].imshow(aligned_noisy_img, cmap='gray')
        # ax[4].set_title('Second Noisy Image Aligned')
        #
        # for a in ax:
        #     a.axis('off')
        #
        # plt.savefig(f"/opt/KAIR/data/BSD68_reproducibility_data/train/alignment_figs/{idx}.png")
        # plt.cla()
    sum_of_differences += difference
    #
    # if true_shifts[0] != -computed_shifts[0] or true_shifts[1] != -computed_shifts[1]:
print(f"Total different images: {cnt_diff}")
if cnt_diff < 2:
    print("Stopping due to low amount of different images")
    exit(1)

final_output_npy = []
final_all_npy = []
for idx, data in enumerate(np.array(all_npy)):
    clean_img = data[0, :, :, 0]
    noisy_img = data[1, :, :, 0]
    noisy_img2 = data[2, :, :, 0]
    shifted_noisy_img = data[3, :, :, 0]
    aligned_noisy_img = data[4, :, :, 0]

    augmented_clean_images = augment_image(clean_img)
    augmented_noisy_images = augment_image(noisy_img)
    augmented_noisy_images2 = augment_image(noisy_img2)
    augmented_noisy_images_shifted = augment_image(shifted_noisy_img)
    augmented_aligned_noisy_images = augment_image(aligned_noisy_img)

    for clean_img_aug, noisy_img_aug, noisy_img_aug2, noisy_img_shifted_aug, aligned_noisy_img_aug in zip(
            augmented_clean_images,
            augmented_noisy_images,
            augmented_noisy_images2,
            augmented_noisy_images_shifted,
            augmented_aligned_noisy_images
    ):
        final_output_npy.append(
            (
                clean_img_aug[:, :, np.newaxis],
                noisy_img_aug[:, :, np.newaxis],
                aligned_noisy_img_aug[:, :, np.newaxis],
            )
        )
        final_all_npy.append(
            (
                clean_img_aug[:, :, np.newaxis],
                noisy_img_aug[:, :, np.newaxis],
                noisy_img_aug2[:, :, np.newaxis],
                noisy_img_shifted_aug[:, :, np.newaxis],
                aligned_noisy_img_aug[:, :, np.newaxis],
            )
        )

final_output_npy = np.array(final_output_npy)
print(f"Output shape: {final_output_npy.shape}")
np.save("/opt/KAIR/data/BSD68_reproducibility_data/train/DCNN400_train_gaussian25_with_shifted_and_aligned.npy",
        final_output_npy)

final_all_npy = np.array(final_all_npy)
print(f"All output shape: {final_all_npy.shape}")
np.save("/opt/KAIR/data/BSD68_reproducibility_data/train/DCNN400_train_gaussian25_with_shifted_and_aligned_all.npy",
        final_all_npy)
# print(f"Sum of differences between aligned and original noisy images: {sum_of_differences}")
# attempt += 1
