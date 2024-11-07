import os
#read h5 file
import h5py
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

data_dir = "/home/wasproject/Desktop/Can/DATASETS/sat-thermal-geo/thermal_h5_datasets"
print(os.listdir(data_dir))
data_file_optical = os.path.join(data_dir, "test_database.h5")
data_file_thermal = os.path.join(data_dir, "test_queries.h5")

with h5py.File(data_file_optical, 'r') as f_optical, h5py.File(data_file_thermal, 'r') as f_thermal:
    images_optical = f_optical['image_data']
    names_optical = f_optical['image_name']
    sizes_optical = f_optical['image_size']

    images_thermal = f_thermal['image_data']
    names_thermal = f_thermal['image_name']
    sizes_thermal = f_thermal['image_size']

    # Randomly select 10 samples
    sample_indices = random.sample(range(len(images_optical)), 10)

    for i in sample_indices:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(images_optical[i])
        axs[0].set_title(names_optical[i])
        axs[1].imshow(images_thermal[i])
        axs[1].set_title(names_thermal[i])
        plt.show()
        print(sizes_optical[i], sizes_thermal[i])
        


