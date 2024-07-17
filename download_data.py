import numpy as np
import urllib.request
import os
import zipfile
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# create a folder for our data
if not os.path.isdir('./data'):
    os.mkdir('data')

# check if data has been downloaded already
zipPath="data/BSD68_reproducibility.zip"
if not os.path.exists(zipPath):
    #download and unzip data
    data = urllib.request.urlretrieve('https://download.fht.org/jug/n2v/BSD68_reproducibility.zip', zipPath)
    with zipfile.ZipFile(zipPath, 'r') as zip_ref:
        zip_ref.extractall("data")



X = np.load('data/BSD68_reproducibility_data/train/DCNN400_train_gaussian25.npy')

from PIL import Image
import numpy as np

# Assuming your array is named 'images'
# images = np.random.randint(0, 256, size=(3168, 180, 180), dtype=np.uint8)  # Example random array

# Select the image at index 1

for i in range(X.shape[0]):
    print(f"Working on image {i}")
    image_to_save = X[i]
    image_to_save = np.round(np.clip(image_to_save, 0, 255))
# print(image_to_save.max())
    image_to_save = image_to_save.astype(np.uint8)
# print(image_to_save.max())

# Convert numpy array to PIL Image
    image_to_save_pil = Image.fromarray(image_to_save)

# Save the image
    image_to_save_pil.save(f'/opt/KAIR/datasets/train/noisy/{i}.png')

print("Done downloading train data")