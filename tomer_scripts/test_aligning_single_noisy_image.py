import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import fourier_shift
import matplotlib.pyplot as plt
from PIL import Image
import random

def dft_registration(img1, img2):
    F1 = fft2(img1)
    F2 = fft2(img2)
    R = F1 * np.conj(F2)
    R /= np.abs(R)
    r = fftshift(ifft2(R))
    max_idx = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    shifts = np.array(max_idx) - np.array(r.shape) // 2
    # img2_shifted = np.abs(ifft2(fourier_shift(fft2(img2), shifts)))
    # return shifts, img2_shifted
    return shifts

def cyclic_shift(image, max_shift_fraction=0.25):
    h, w = image.shape
    vertical_shift = random.randint(0, int(h * max_shift_fraction))
    horizontal_shift = random.randint(0, int(w * max_shift_fraction))
    shifted_image = np.roll(image, vertical_shift, axis=0)
    shifted_image = np.roll(shifted_image, horizontal_shift, axis=1)
    return shifted_image, (vertical_shift, horizontal_shift)

def apply_shifts(image, shifts):
    shifted_image = np.roll(image, -shifts[0], axis=0)
    shifted_image = np.roll(shifted_image, -shifts[1], axis=1)
    return shifted_image


# Read a clean image
image_path = 'C:\\Users\\Tomer\\Downloads\\clean_1.png'  # Replace with your image path
clean_image = Image.open(image_path).convert('L')  # Convert to grayscale
clean_image = np.array(clean_image)

# Add noise to the clean image
noise = np.random.normal(0, 25, clean_image.shape)
noisy_image = clean_image + noise
# noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Randomly roll the noisy image
shifted_image, true_shifts = cyclic_shift(noisy_image)
print(f"True Shifts (Vertical, Horizontal): {true_shifts}")
print(f"Shifted image min/max: {shifted_image.min()}, {shifted_image.max()}")


# Align the image back using DFT registration
# computed_shifts, aligned_image = dft_registration(noisy_image, shifted_image)
computed_shifts = dft_registration(noisy_image, shifted_image)
aligned_image = apply_shifts(shifted_image, -computed_shifts)  # Apply negative of computed shifts to align
print(f"Computed Shifts (Vertical, Horizontal): {computed_shifts}")
print(f"Aligned image min/max: {aligned_image.min()}, {aligned_image.max()}")


difference = noisy_image - aligned_image

# Display the original noisy image, shifted noisy image, and aligned noisy image
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(noisy_image, cmap='gray')
ax[0].set_title('Noisy Image')
ax[1].imshow(shifted_image, cmap='gray')
ax[1].set_title('Shifted Noisy Image')
ax[2].imshow(aligned_image, cmap='gray')
ax[2].set_title('Aligned Noisy Image')
ax[3].imshow(difference, cmap='gray')
ax[3].set_title('Difference')

for a in ax:
    a.axis('off')

plt.show()
