import numpy as np
import cv2
import os


def save_images_from_npy(npy_file, output_dir):
    # Load the .npy file
    data = np.load(npy_file, allow_pickle=True)
    print(f"Loaded input NPY file of shape: {data.shape}")
    # Create output directories if they don't exist
    clean_dir = os.path.join(output_dir, 'clean')
    noisy_dir = os.path.join(output_dir, 'noisy')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    # Save the first num_pairs pairs as PNG
    for i in range(len(data)):
        print(f"Processing image {i}")
        clean_img = data[i][0]
        noisy_img = data[i][1]


        clean_img_path = os.path.join(clean_dir, f'clean_{i}.png')
        noisy_img_path = os.path.join(noisy_dir, f'noisy_{i}.png')

        cv2.imwrite(clean_img_path, clean_img)
        cv2.imwrite(noisy_img_path, noisy_img)


# Usage
npy_file = '/opt/KAIR/data/BSD68_reproducibility_data/test/DCNN400_test_gaussian100_pairs.npy'
output_dir = '/opt/KAIR/data/BSD68_reproducibility_data/test/bsd68_noise100_examples'
save_images_from_npy(npy_file, output_dir)
print("Done converting to png")