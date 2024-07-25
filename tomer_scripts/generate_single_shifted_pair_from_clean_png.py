import numpy as np
import cv2
import os


def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = image + noise
    return noisy_image


def random_shift(image, max_shift_fraction=0.25):
    h, w = image.shape
    max_shift_h = int(h * max_shift_fraction)
    max_shift_w = int(w * max_shift_fraction)

    shift_h = np.random.randint(0, max_shift_h + 1)
    shift_w = np.random.randint(0, max_shift_w + 1)

    shifted_image = np.roll(image, shift_h, axis=0)
    shifted_image = np.roll(shifted_image, shift_w, axis=1)

    return shifted_image, shift_h, shift_w


def save_image(image, path):
    cv2.imwrite(path, image)


def process_image(clean_image_path, output_dir, noise_level):
    clean_img = cv2.imread(clean_image_path, cv2.IMREAD_GRAYSCALE)

    if clean_img is None:
        raise FileNotFoundError(f"Clean image not found: {clean_image_path}")

    # Create a noisy image
    noisy_img = add_gaussian_noise(clean_img, sigma=noise_level)

    second_noisy_img = add_gaussian_noise(clean_img, sigma=noise_level)
    shifted_noisy_img = random_shift(second_noisy_img)[0]
    print(noisy_img.min(), noisy_img.max())
    os.makedirs(output_dir, exist_ok=True)

    noisy_path = os.path.join(output_dir, 'noisy.png')
    shifted_noisy_img_path = os.path.join(output_dir, 'shifted_noisy.png')

    save_image(noisy_img, noisy_path)
    save_image(shifted_noisy_img, shifted_noisy_img_path)


# Usage
clean_image_path = 'C:\\Users\\Tomer\\Downloads\\clean_1.png'
output_dir = 'C:\\Users\\Tomer\\Downloads\\output_images'
process_image(clean_image_path, output_dir, noise_level=175)

print("Images saved successfully.")
