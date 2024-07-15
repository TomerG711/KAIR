from PIL import Image, ImageOps
import os

input_folder_path = '/opt/KAIR/datasets/train/gt'
output_folder_path = '/opt/KAIR/datasets/train/gt_aug'

os.makedirs(output_folder_path, exist_ok=True)

for filename in os.listdir(input_folder_path):
    # Skip non-image files
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue

    # Open the image file
    image_path = os.path.join(input_folder_path, filename)
    image = Image.open(image_path)

    # Save the original image
    original_filename = f'{os.path.splitext(filename)[0]}_original.png'
    image.save(os.path.join(output_folder_path, original_filename))

    # Mirror the original image horizontally
    mirrored_original_image = ImageOps.mirror(image)
    mirrored_original_filename = f'{os.path.splitext(filename)[0]}_original_mirrored.png'
    mirrored_original_image.save(os.path.join(output_folder_path, mirrored_original_filename))


    # Rotate by 90 degrees three times
    for angle in [90, 180, 270]:
        rotated_image = image.rotate(angle, expand=True)
        rotated_filename = f'{os.path.splitext(filename)[0]}_rotated_{angle}.png'
        rotated_image.save(os.path.join(output_folder_path, rotated_filename))

        # Mirror the rotated images horizontally
        mirrored_rotated_image = ImageOps.mirror(rotated_image)
        mirrored_rotated_filename = f'{os.path.splitext(filename)[0]}_rotated_{angle}_mirrored.png'
        mirrored_rotated_image.save(os.path.join(output_folder_path, mirrored_rotated_filename))

    # # Add mirrored versions (horizontal and vertical)
    # mirrored_horizontally = ImageOps.mirror(image)
    # mirrored_vertically = ImageOps.flip(image)
    #
    # mirrored_horizontally.save(os.path.join(output_folder_path, f'{os.path.splitext(filename)[0]}_mirrored_horizontal.jpg'))
    # mirrored_vertically.save(os.path.join(output_folder_path, f'{os.path.splitext(filename)[0]}_mirrored_vertical.jpg'))

    # Close the original image file
    image.close()
