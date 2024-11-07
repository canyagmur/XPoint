import os
import h5py

# Define the paths
keypoint_file_path = "/home/wasproject/Desktop/Can/DATASETS/hdf5_DATASETS/keypoint_files/mscoco-window/labels_mscoco_window.hdf5"
new_keypoint_file_path = "/home/wasproject/Desktop/Can/DATASETS/hdf5_DATASETS/keypoint_files/mscoco-window/labels_mscoco_window_remapped.hdf5"
foldername = "/home/wasproject/Desktop/Can/DATASETS/COCO/train2014/images"

# Try to open the keypoint file
try:
    keypoint_file = h5py.File(keypoint_file_path, 'r', swmr=True)
except IOError as e:
    print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, keypoint_file_path))
    exit()

# Create a new HDF5 file for the remapped keys
try:
    new_keypoint_file = h5py.File(new_keypoint_file_path, 'w')
except IOError as e:
    print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, new_keypoint_file_path))
    exit()

# Get a sorted list of filenames (without extension) from the folder
image_filenames = sorted([filename[:-4] for filename in os.listdir(foldername) if filename.endswith(".jpg")])

# Check if the number of images matches the number of keys in the keypoint file
if len(image_filenames) != len(keypoint_file.keys()):
    print("Number of images and keypoints do not match!")
    keypoint_file.close()
    new_keypoint_file.close()
    exit()

# Create a mapping of old key names ("1", "2", ...) to new image filenames
mapping = {str(i+1): image_filenames[i] for i in range(len(image_filenames))}

# Now copy data to the new file with remapped keys
for old_key, new_key in mapping.items():
    # Copy the data directly from the old key to the new key, without nesting
    new_keypoint_file.create_group(new_key)
    new_keypoint_file[new_key].create_dataset("keypoints_optical", data=keypoint_file[old_key]["keypoints_optical"][...])
    new_keypoint_file[new_key].create_dataset("keypoints_thermal", data=keypoint_file[old_key]["keypoints_thermal"][...])

    

# Close both HDF5 files
keypoint_file.close()
new_keypoint_file.close()


print("Keys have been remapped and saved to the new file successfully!")
