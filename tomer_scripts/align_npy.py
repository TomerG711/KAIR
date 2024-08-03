import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import fourier_shift


def dft_registration(img1, img2):
    F1 = fft2(img1)
    F2 = fft2(img2)

    # Compute the cross-power spectrum
    R = F1 * np.conj(F2)
    R /= np.abs(R)

    r = fftshift(ifft2(R))

    # Find the peak in the cross-correlation
    max_idx = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    shifts = np.array(max_idx) - np.array(r.shape) // 2

    # img2_shifted = np.abs(ifft2(fourier_shift(fft2(img2), shifts)))
    #
    # return shifts, img2_shifted
    return shifts

def apply_shifts(image, shifts):
    shifted_image = np.roll(image, -shifts[0], axis=0)
    shifted_image = np.roll(shifted_image, -shifts[1], axis=1)
    return shifted_image


def process_images(npy_file_path, output_npy_file_path):
    # Load the npy file
    original_data = np.load(npy_file_path)

    # Number of images and dimensions
    num_images = len(original_data)
    assert num_images == 6336, "Expected 6336 images"

    # Prepare list to store results
    all_images = []
    print(f"Starting to process, got input shape: {original_data.shape}")

    # Process images
    for i in range(3168):
        print(f"Processing image {i}")
        clean_img = original_data[i, 0, :, :, 0]
        noisy_img = original_data[i, 1, :, :, 0]
        matching_noisy_img = original_data[i + 3168, 1, :, :, 0]

        # _, aligned_img = dft_registration(clean_img, matching_noisy_img)
        computed_shifts = dft_registration(noisy_img, matching_noisy_img)
        aligned_img = apply_shifts(matching_noisy_img, -computed_shifts)  # Apply negative of computed shifts to align

        # Append each set of images as a tuple
        all_images.append((clean_img[:, :, np.newaxis], noisy_img[:, :, np.newaxis], aligned_img[:, :, np.newaxis]))

    # Save to new npy file
    output_npy = np.array(all_images)
    print(f"Saving output of shape: {output_npy.shape}")
    np.save(output_npy_file_path, output_npy)


# Usage
process_images(
    '/opt/KAIR/data/BSD68_reproducibility_data/train/DCNN400_train_gaussian25_with_shifted.npy',
    '/opt/KAIR/data/BSD68_reproducibility_data/train/DCNN400_train_gaussian25_with_shifted_and_aligned.npy',
)
