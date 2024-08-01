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
        aligned_noisy_img = data[i, 2, :, :, 0]

        clean_img_path = os.path.join(clean_dir, f'clean_{i}.png')
        noisy_img_path = os.path.join(noisy_dir, f'noisy_{i}.png')
        aligned_noisy_img_path = os.path.join(noisy_dir, f'aligned_noisy_{i}.png')

        cv2.imwrite(clean_img_path, clean_img)
        cv2.imwrite(noisy_img_path, noisy_img)
        cv2.imwrite(aligned_noisy_img_path, aligned_noisy_img)


# Usage
npy_file = '/opt/KAIR/data/BSD68_reproducibility_data/train/DCNN400_train_gaussian100_with_shifted_and_aligned.npy'
output_dir = '/opt/KAIR/data/BSD68_reproducibility_data/train/test_n2n_alignment100'
save_images_from_npy(npy_file, output_dir)
