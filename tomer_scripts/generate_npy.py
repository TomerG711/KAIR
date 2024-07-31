import os
import numpy as np
import cv2


def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # or other formats
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return np.array(images)


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
    # print(image.min(),image.max())
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = image + noise
    return noisy_image


def create_clean_noisy_pairs(images, sigma):
    clean_noisy_pairs = []
    for img in images:
        noisy_img = add_gaussian_noise(img, sigma)
        augmented_clean_images = augment_image(img)
        augmented_noisy_images = augment_image(noisy_img)
        for clean_img, noisy_img in zip(augmented_clean_images, augmented_noisy_images):
            clean_noisy_pairs.append((clean_img[:, :, np.newaxis], noisy_img[:, :, np.newaxis]))
    return np.array(clean_noisy_pairs)


def create_clean_noisy_pairs_without_aug(images):
    clean_noisy_pairs = np.empty(68, dtype=object)
    i = 0
    for img in images:
        noisy_img = add_gaussian_noise(img)
        a = np.empty(2, dtype=object)
        a[0] = img[:, :, np.newaxis]
        a[1] = noisy_img[:, :, np.newaxis]
        clean_noisy_pairs[i] = a
        i += 1
    return clean_noisy_pairs


# Load images
directory = "/opt/KAIR/data/BSD68_reproducibility_data/train/gt"
images = load_images_from_directory(directory)

# Create clean and noisy pairs
# clean_noisy_pairs = create_clean_noisy_pairs_without_aug(images)
clean_noisy_pairs = create_clean_noisy_pairs(images, 100)
print(clean_noisy_pairs.shape)
# print(len(images))
# Reshape to desired format
# final_data = clean_noisy_pairs.reshape((68, 2, -1, -1, 1))
final_data = clean_noisy_pairs
# Save to .npy file
np.save("/opt/KAIR/data/BSD68_reproducibility_data/train/DCNN400_train_gaussian100_pairs.npy", final_data)

print("Data saved successfully. Shape:", final_data.shape)
