import argparse
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import sys
from tqdm import tqdm
import os

import xpoint.datasets as data
import xpoint.utils as utils

parser = argparse.ArgumentParser(description='Show a sample of the dataset')
parser.add_argument('-d', '--dataset-path', required=True, help='Input dataset file')
parser.add_argument('-k', '--keypoint-file', required=True, help='Keypoint dataset file')
parser.add_argument('-n', dest='sample_number', type=int, default=0, help='Sample to show')
parser.add_argument('-r', '--radius', default=4, type=int, help='Radius of the keypoint circle')

args = parser.parse_args()


config_imagepair = {
    'filename': args.dataset_path if os.path.isfile(args.dataset_path) else None,
    'foldername' : args.dataset_path if os.path.isdir(args.dataset_path) else None,
    'height': -1,
    'width': -1,
    'raw_thermal': False,
    'single_image': False,
    'augmentation': {
        'photometric': {
            'enable': False,
        },
        'homographic': {
            'enable': False,
        },
    }
}


dataset = data.ImagePairDataset(config_imagepair)
try:
    keypoint_file = h5py.File(args.keypoint_file, 'r', swmr=True)
except IOError as e:
    print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, args.keypoint_file))
    sys.exit()

# display an individual sample
# get the data
print(keypoint_file.keys())
for i in range(0, len(dataset)):
    sample = dataset[i]
    name = dataset.get_name(i)
    # if ".jpg"  in name or ".png" in name:
    #     name = name[:-4]

    

    labels_optical = keypoint_file[name]['keypoints_optical'][...]
    labels_thermal = keypoint_file[name]['keypoints_thermal'][...]


    #print(dict(keypoint_file[name]).keys())
    out_thermal = cv2.cvtColor((np.clip(sample['thermal']['image'].squeeze().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
    out_optical = cv2.cvtColor((np.clip(sample['optical']['image'].squeeze().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
    #print("optical shape, thermal shape: ", out_optical.shape, out_thermal.shape)
    if 1: #labels_optical.shape[0] < 200 or labels_thermal.shape[0] < 200:
        print("Number of keypoints optical: {}".format(labels_optical.shape[0]))
        print("Number of keypoints thermal: {}".format(labels_thermal.shape[0]))
    predictions_optical = [cv2.KeyPoint(c[1], c[0], args.radius) for c in labels_optical.astype(np.float32)]
    predictions_thermal = [cv2.KeyPoint(c[1], c[0], args.radius) for c in labels_thermal.astype(np.float32)]

    # draw predictions and ground truth on image
    out_optical = cv2.drawKeypoints(out_optical,
                                    predictions_optical,
                                    outImage=np.array([]),
                                    color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_thermal = cv2.drawKeypoints(out_thermal,
                                    predictions_thermal,
                                    outImage=np.array([]),
                                    color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    mask_optical = np.repeat(np.expand_dims(sample['optical']['valid_mask'].squeeze().numpy(), axis=2), 3, axis=2)
    mask_thermal = np.repeat(np.expand_dims(sample['thermal']['valid_mask'].squeeze().numpy(), axis=2), 3, axis=2)
    fig, axs = plt.subplots(1, 2, figsize=(8,8))


    axs[0].imshow(out_thermal * mask_thermal)
    axs[0].set_title('Thermal Masked')

    axs[1].imshow(out_optical * mask_optical)
    axs[1].set_title('Optical Masked')

    # axs[1, 0].imshow(out_thermal)
    # axs[1, 0].set_title('Thermal')

    # axs[1, 1].imshow(out_optical)
    # axs[1, 1].set_title('Optical')



    plt.show()
