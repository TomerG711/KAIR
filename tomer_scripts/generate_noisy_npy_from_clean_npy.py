import numpy as np

def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = image + noise
    return noisy_image

def generate_noisy_images_from_npy(npy_file, output_file, sigma):
    print("Loading original NPY")
    data = np.load(npy_file, allow_pickle=True)
    print(f"Loaded input NPY of shape: {data.shape}")
    clean_noisy_pairs = np.empty(len(data), dtype=object)


    for i in range(len(data)):
        a = np.empty(2, dtype=object)
        clean_img = data[i][0]
        print(f"Processing image {i} of shape: {clean_img.shape}")
        noisy_img = add_gaussian_noise(clean_img, sigma)
        print(f"Generated noisy version of {i} with shape: {noisy_img.shape}")
        a[0] = clean_img
        a[1] = noisy_img
        clean_noisy_pairs[i] = a
        # new_npy_data.append((clean_img, noisy_img))

    output_npy = np.array(clean_noisy_pairs)
    print(f"Saving output NPY of shape: {output_npy.shape}")
    np.save(output_file, output_npy, allow_pickle=True)

input_npy_file = '/opt/KAIR/data/BSD68_reproducibility_data/test/DCNN400_test_gaussian25_pairs.npy'
output_npy_file = '/opt/KAIR/data/BSD68_reproducibility_data/test/DCNN400_test_gaussian100_pairs.npy'

generate_noisy_images_from_npy(input_npy_file, output_npy_file, 100)