import os
import numpy as np
import cv2


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


def extend_dataset_with_shifts_and_augmentations(original_data, original_images, sigma):
    extended_data = []

    for i in range(len(original_data)):
        clean_img = original_data[i, 0, :, :, 0]
        noisy_img = original_data[i, 1, :, :, 0]

        # Add the original augmented images to the extended data
        # augmented_clean_images = augment_image(clean_img)
        # augmented_noisy_images = augment_image(noisy_img)
        # for clean_img_aug, noisy_img_aug in zip(augmented_clean_images, augmented_noisy_images):
        extended_data.append((clean_img[:, :, np.newaxis], noisy_img[:, :, np.newaxis]))

    for img in original_images:
        new_noisy_img = add_gaussian_noise(img, sigma)

        # Generate random shifts
        h, w = img.shape
        max_shift_h = int(h * 0.25)
        max_shift_w = int(w * 0.25)
        shift_h = np.random.randint(0, max_shift_h + 1)
        shift_w = np.random.randint(0, max_shift_w + 1)

        shifted_clean_img = random_shift(img, shift_h, shift_w)
        shifted_noisy_img = random_shift(new_noisy_img, shift_h, shift_w)

        # Add the shifted augmented images to the extended data
        augmented_clean_images_shifted = augment_image(shifted_clean_img)
        augmented_noisy_images_shifted = augment_image(shifted_noisy_img)
        for clean_img_shifted_aug, noisy_img_shifted_aug in zip(augmented_clean_images_shifted,
                                                                augmented_noisy_images_shifted):
            extended_data.append((clean_img_shifted_aug[:, :, np.newaxis], noisy_img_shifted_aug[:, :, np.newaxis]))

    return np.array(extended_data)


# Load the original npy file
original_npy_file = '/opt/KAIR/data/BSD68_reproducibility_data/train/DCNN400_train_gaussian100_pairs.npy'
original_data = load_npy(original_npy_file)

# Load the original clean images from the directory
clean_images_directory = '/opt/KAIR/data/BSD68_reproducibility_data/train/gt'
original_clean_images = load_images_from_directory(clean_images_directory)

# Extend the dataset
extended_data = extend_dataset_with_shifts_and_augmentations(original_data, original_clean_images, 100)

# Save the extended dataset to a new npy file
new_npy_file = '/opt/KAIR/data/BSD68_reproducibility_data/train/DCNN400_train_gaussian100_with_shifted.npy'
save_npy(extended_data, new_npy_file)

print("Extended data saved successfully. Shape:", extended_data.shape)
