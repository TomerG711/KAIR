# import numpy as np
# from PIL import Image
# import os
#
# # a = np.load()
# output_folder = 'tomers_shit'
#
# os.makedirs(output_folder, exist_ok=True)
#
# # Load the .npy file
# images = np.load('DCNN400_train_gaussian25.npy')
#
# # Check the shape of the array to determine the number of images
# num_images = images.shape[0]
#
# # Iterate through the images and save each one
# for i in range(num_images):
#     # Convert the numpy array to a PIL Image
#     img = Image.fromarray(images[i].astype('uint8'))
#
#     # Save the image
#     img.save(os.path.join(output_folder, f'image_{i}.png'))
#
# print(f'{num_images} images have been saved in the {output_folder} directory.')
#
#
# print(1)

import numpy as np
from PIL import Image
import os
import random

# Directory containing the original images
input_folder = 'trainsets/trainH'
output_folder = 'trainsets/trainH_shifted'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


# Function to add Gaussian noise
def add_noise(image, noise_level=25):
    noise = np.random.normal(0, noise_level, image.shape).astype('uint8')
    noised_image = np.clip(image + noise, 0, 255).astype('uint8')
    return noised_image


# Function to perform cyclic shift
def cyclic_shift(image, max_shift_fraction=0.5):
    h, w = image.shape
    vertical_shift = random.randint(0, int(h * max_shift_fraction))
    horizontal_shift = random.randint(0, int(w * max_shift_fraction))
    shifted_image = np.roll(image, vertical_shift, axis=0)
    shifted_image = np.roll(shifted_image, horizontal_shift, axis=1)
    return shifted_image


# Iterate through the images in the input directory and process each one
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        # Load the image
        original_image_path = os.path.join(input_folder, filename)
        original_image = np.array(Image.open(original_image_path))

        # Add noise to the original image
        noised_image = add_noise(original_image, noise_level=25)
        # noised_image_pil = Image.fromarray(noised_image)
        # noised_image_pil.save(os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_noised.png'))

        # Add noise again and perform cyclic shift
        # noised_shifted_image = add_noise(original_image, noise_level=25)
        noised_shifted_image = cyclic_shift(noised_image, max_shift_fraction=0.25)
        noised_shifted_image_pil = Image.fromarray(noised_shifted_image)
        noised_shifted_image_pil.save(
            os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_shifted.png'))

print(f'All images have been processed and saved in the {output_folder} directory.')