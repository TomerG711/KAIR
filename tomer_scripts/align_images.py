from PIL import Image
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import fourier_shift
import matplotlib.pyplot as plt

img1 = np.asarray(Image.open("C:\\Users\\Tomer\\Downloads\\output_images\\noisy.png"))
img2 = np.asarray(Image.open("C:\\Users\\Tomer\\Downloads\\output_images\\shifted_noisy.png"))


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

    img2_shifted = np.abs(ifft2(fourier_shift(fft2(img2), shifts)))

    return shifts, img2_shifted


shifts, img2_aligned = dft_registration(img1, img2)
# img2_aligned = img2_aligned.astype(np.uint8)
# img2_aligned = np.rint(img2_aligned).astype(np.uint8)
print(shifts)
print(img1.min(), img1.max())
print(img2.min(), img2.max())
print(img2_aligned.min(), img2_aligned.max())
# Compute the difference between the original and aligned images
difference = img1 - img2_aligned

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(img1, cmap='gray')
ax[0].set_title('Image')
ax[1].imshow(img2, cmap='gray')
ax[1].set_title('Shifted Image')
ax[2].imshow(img2_aligned, cmap='gray')
ax[2].set_title('Aligned Image')
ax[3].imshow(difference, cmap='gray')
ax[3].set_title('Difference')

for a in ax:
    a.axis('off')

noise = 0
plt.suptitle(f"Noise {noise}")
plt.show()
# plt.savefig(f"C:\\Users\\Tomer\\Downloads\\output_images\\noisy_{noise}_aligned.png")

"""
TODO:
run DNCNN, DNCNN+SURE,DNCNN as N2N with 25&100
test alignment for molecule datasets
"""
