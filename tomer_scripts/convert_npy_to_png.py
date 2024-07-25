import numpy as np
import cv2
import os


def save_images_from_npy(npy_file, output_dir, num_pairs=20):
    # Load the .npy file
    data = np.load(npy_file)

    # Create output directories if they don't exist
    clean_dir = os.path.join(output_dir, 'clean')
    noisy_dir = os.path.join(output_dir, 'noisy')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    # Save the first num_pairs pairs as PNG
    for i in range(num_pairs):
        clean_img = data[i, 0, :, :, 0]
        noisy_img = data[i, 1, :, :, 0]

        clean_img_path = os.path.join(clean_dir, f'clean_{i + 1}.png')
        noisy_img_path = os.path.join(noisy_dir, f'noisy_{i + 1}.png')

        cv2.imwrite(clean_img_path, clean_img)
        cv2.imwrite(noisy_img_path, noisy_img)


# Usage
npy_file = '/opt/KAIR/output_here/DCNN400_train_gaussian25_pairs.npy'
output_dir = '/opt/KAIR/output_here/test_png'
save_images_from_npy(npy_file, output_dir)
