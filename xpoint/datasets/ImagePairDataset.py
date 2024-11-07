from __future__ import print_function
import os
import numpy as np
import random
import sys
import torch
from torch.utils.data.dataset import Dataset
import h5py
import cv2
import math


import xpoint.utils as utils
from .augmentation import augmentation



class ImagePairDataset(Dataset):
    '''
    Class to load a sample from a given hdf5 file.
    '''
    default_config = {
        'filename': None, # for file dataset
        "foldername": None,  # for folder dataset
        'keypoints_filename': None,
        'height': -1,
        'width': -1,
        'raw_thermal': False,
        'single_image': True,
        'random_pairs': False,
        'return_name' : True,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'border_reflect': True,
                'valid_border_margin': 0,
                'mask_border': True,
            },
        }
    }

    def check_folder_structure(self, folder_path):
        optical_folder_path = os.path.join(folder_path, "optical")
        thermal_folder_path = os.path.join(folder_path, "thermal")
        images_folder_path = os.path.join(folder_path, "images")

        if (
            os.path.exists(optical_folder_path)
            and os.path.isdir(optical_folder_path)
            and os.path.exists(thermal_folder_path)
            and os.path.isdir(thermal_folder_path)
        ):
            return (optical_folder_path, thermal_folder_path)
        elif os.path.exists(images_folder_path) and os.path.isdir(images_folder_path):
            return (images_folder_path, images_folder_path)
        else:
            # Generate a tree-like representation of the folder structure
            folder_structure = f"""
            Expected folder structure:
            - {os.path.basename(folder_path)} (root)
            - optical
            - thermal
            OR
            - images
            """
            raise ValueError(f"Folder structure is not correct.\n{folder_structure}")

    def __init__(self, config):
        if config:
            import copy

            self.config = utils.dict_update(copy.copy(self.default_config), config)
        else:
            self.config = self.default_config
        if (self.config["filename"] is None) == (self.config["foldername"] is None):
            raise ValueError(
                "ImagePairDataset: The dataset filename XOR foldername needs to be present in the config file"
            )

        self.data_is_file = self.config["filename"] is not None
        self.data_path = None
        if self.data_is_file:
            self.data_path = self.config["filename"]
        else:
            self.data_path = self.config["foldername"]

        if not self.data_is_file:
            if not os.path.exists(self.data_path):
                raise ValueError(
                    "The folder {} does not exists.".format(self.data_path)
                )
            # check folder structure
            self.data_path = self.check_folder_structure(self.data_path)
            print("Folder structure is correct.")
            print("Paths:", self.data_path)

        self.num_files = None
        self.memberslist = None

        if self.data_is_file:
            try:
                h5_file = h5py.File(self.data_path, "r")
            except IOError as e:
                print(
                    "I/O error({0}): {1}: {2}".format(
                        e.errno, e.strerror, self.data_path
                    )
                )
                sys.exit()
            # extract info from the h5 file
            self.num_files = len(h5_file.keys())
            self.memberslist = list(h5_file.keys())
            h5_file.close()

        else:
            self.memberslist = [
                f
                for f in os.listdir(self.data_path[0])
                if f.endswith(".jpg") or f.endswith(".png")
            ]
            self.num_files = len(self.memberslist)

        if self.config["single_image"] and self.config["random_pairs"]:
            print("INFO: random_pairs has no influence if single_image is true")

        # process keypoints if filename is present
        if self.config["keypoints_filename"] is not None:
            # check if file exists
            try:
                keypoints_file = h5py.File(self.config["keypoints_filename"], "r")
            except IOError as e:
                print(
                    "I/O error({0}): {1}: {2}".format(
                        e.errno, e.strerror, self.config["keypoints_filename"]
                    )
                )
                sys.exit()

            # check that for every sample there are keypoints
            keypoint_members = list(keypoints_file.keys())
            #print(keypoint_members)
            missing_labels = []

            self.memberlist_checklist = self.memberslist.copy()
            #remove the extension from the memberlist_checklist for every element if exists :
            #if it contains extension like .jpg or .png dont delete it
            if not (".png" or ".jpg" in self.keypoint_members[0]):
                for i in range(len(self.memberlist_checklist)):
                    self.memberlist_checklist[i] = self.memberlist_checklist[i].rsplit(".", 1)[0]

            #print("First 10 kp : ",keypoint_members[:100])
            for item in self.memberlist_checklist:
                if item not in keypoint_members:
                    print("Key not found : ",item, keypoint_members[0])
                    missing_labels.append(item)

            if len(missing_labels) > 0:
                raise IndexError(
                    "Labels for the following samples (display only first 10) not available: {}, len : {}".format(
                        missing_labels[:10], len(missing_labels)
                    )
                )

        if self.data_is_file:
            print(
                "The dataset "
                + self.config["filename"]
                + " contains {} samples".format(self.num_files)
            )
        else:
            print(
                "The dataset "
                + self.config["foldername"]
                + " contains {} samples".format(self.num_files)
            )

    def __getitem__(self, index):
        sample = None
        if self.data_is_file:
            h5_file = h5py.File(self.data_path, "r", swmr=True)
            sample = h5_file[self.memberslist[index]]

            optical = sample["optical"][...]
            if "thermal" in sample.keys() or "thermal_raw" in sample.keys():
                if self.config["raw_thermal"]:
                    thermal = sample["thermal_raw"][...]
                else:
                    thermal = sample["thermal"][...]
            else:
                thermal = sample["optical"][...].copy()
        else:
            sample_optical = cv2.imread(
                os.path.join(self.data_path[0], self.memberslist[index])
            )
            sample_optical_gray = cv2.cvtColor(sample_optical, cv2.COLOR_BGR2GRAY)
            sample_thermal = cv2.imread(
                os.path.join(self.data_path[1], self.memberslist[index])
            )
            sample_thermal_gray = cv2.cvtColor(sample_thermal, cv2.COLOR_BGR2GRAY)
            optical = sample_optical_gray / 255.0
            thermal = sample_thermal_gray / 255.0




        if thermal.shape != optical.shape:
            raise ValueError('ImagePairDataset: The optical and thermal image must have the same shape')

        path_to_dataset = self.config["filename"] if self.data_is_file else self.config["foldername"]
        if "coco" in path_to_dataset.lower():
            optical = cv2.resize(optical, (256,256))
            thermal = cv2.resize(thermal, (256,256)) #since keypoints are in 256x256
            #print("Resized to 256x256 for COCO dataset, REMEMBER TO DELETE THAT!!!!!!")
        

        

        #HERE I CHANGED
        if self.config['keypoints_filename'] is not None:
            with h5py.File(self.config['keypoints_filename'], 'r', swmr=True) as keypoints_file:
                if "keypoints_optical" in keypoints_file[self.memberlist_checklist[index]].keys():
                    keypoints = [np.array(keypoints_file[self.memberlist_checklist[index]]["keypoints_optical"])  ,  np.array(keypoints_file[self.memberlist_checklist[index]]["keypoints_thermal"])]
                else:
                    keypoints = np.array(keypoints_file[self.memberlist_checklist[index]]["keypoints"])

        else:
            keypoints = None
        if "redfeat" in path_to_dataset.lower():
            #print("Resizing to min length side to 256 for redfeat dataset")
            minsize = self.config['height'] if self.config['height'] > 0 else 256
            if self.config['keypoints_filename'] is not None:
                if type(keypoints) is list:
                    keypoints_optical = keypoints[0]
                    keypoints_thermal = keypoints[1]
                    optical,keypoints_optical = self.resize_image_and_keypoints(optical,keypoints_optical,min_size=minsize)
                    thermal,keypoints_thermal = self.resize_image_and_keypoints(thermal,keypoints_thermal,min_size=minsize)
                    keypoints = [keypoints_optical,keypoints_thermal]
                else:
                    optical,keypoints = self.resize_image_and_keypoints(optical,keypoints,min_size=minsize)
                    thermal,_ = self.resize_image_and_keypoints(thermal,None,min_size=minsize)
            
            else:
                optical,_= self.resize_image_and_keypoints(optical,keypoints=None,min_size=minsize)
                thermal,_= self.resize_image_and_keypoints(thermal,keypoints=None,min_size=minsize)

        


        # subsample images if requested
        if self.config['height'] > 0 or self.config['width'] > 0:
            if self.config['height'] > 0:
                h = self.config['height'] // 32 *32 #make sure it is divisible by 32(encoder downsampling ratio)
            else:
                h = thermal.shape[0]

            if self.config['width'] > 0:
                w = self.config['width'] // 32 *32 #make sure it is divisible by 32(encoder downsampling ratio)
            else:
                w = thermal.shape[1]
            
            if w > thermal.shape[1] or h > thermal.shape[0]:
                print("optical shape : ",optical.shape, "thermal shape : ",thermal.shape)
                raise ValueError('ImagePairDataset: Requested height/width exceeds original image size')

            # subsample the image
            i_h = random.randint(0, thermal.shape[0]-h)
            i_w = random.randint(0, thermal.shape[1]-w)

            optical = optical[i_h:i_h+h, i_w:i_w+w]
            thermal = thermal[i_h:i_h+h, i_w:i_w+w]

            if keypoints is not None:
                if type(keypoints) is list:
                    keypoints[0] = keypoints[0] - np.array([[i_h,i_w]])
                    keypoints[1] = keypoints[1] - np.array([[i_h,i_w]])

                    # filter out bad ones
                    keypoints[0] = keypoints[0][np.logical_and(
                                            np.logical_and(keypoints[0][:,0] >=0,keypoints[0][:,0] < h),
                                            np.logical_and(keypoints[0][:,1] >=0,keypoints[0][:,1] < w))]
                    keypoints[1] = keypoints[1][np.logical_and(
                                            np.logical_and(keypoints[1][:,0] >=0,keypoints[1][:,0] < h),
                                            np.logical_and(keypoints[1][:,1] >=0,keypoints[1][:,1] < w))]
                else:
                    keypoints = keypoints - np.array([[i_h,i_w]])

                    # filter out bad ones
                    keypoints = keypoints[np.logical_and(
                                        np.logical_and(keypoints[:,0] >=0,keypoints[:,0] < h),
                                        np.logical_and(keypoints[:,1] >=0,keypoints[:,1] < w))]

        else:
            h = thermal.shape[0]
            w = thermal.shape[1]
        out = {}
        if self.config['single_image']:
            is_optical = bool(random.randint(0,1))

            if is_optical:
                image = optical
            else:
                image = thermal

            # augmentation
            if self.config['augmentation']['photometric']['enable']:
                image = augmentation.photometric_augmentation(image, **self.config['augmentation']['photometric'])

            if self.config['augmentation']['homographic']['enable']:
                image, keypoints, valid_mask = augmentation.homographic_augmentation(image, keypoints, **self.config['augmentation']['homographic'])
            else:
                valid_mask = augmentation.dummy_valid_mask(image.shape)

            # add channel information to image and mask
            image = np.expand_dims(image, 0)
            valid_mask = np.expand_dims(valid_mask, 0)

            # add to output dict
            out['image'] = torch.from_numpy(image.astype(np.float32))
            out['valid_mask'] = torch.from_numpy(valid_mask.astype(bool))
            out['is_optical'] = torch.BoolTensor([is_optical])
            if keypoints is not None:
                keypoints[0] = utils.generate_keypoint_map(keypoints[0], (h,w))
                keypoints[1] = utils.generate_keypoint_map(keypoints[1], (h,w))
                out['keypoints']["optical"] = torch.from_numpy(keypoints[0].astype(bool))
                out['keypoints']["thermal"] = torch.from_numpy(keypoints[1].astype(bool))

        else:
            # initialize the images
            out['optical'] = {}
            out['thermal'] = {}

            optical_is_optical = True
            thermal_is_optical = False
            if self.config['random_pairs']:
                tmp_optical = optical
                tmp_thermal = thermal
                if bool(random.randint(0,1)):
                    optical = tmp_thermal
                    optical_is_optical = False
                if bool(random.randint(0,1)):
                    thermal = tmp_optical
                    thermal_is_optical = True

            # augmentation
            if self.config['augmentation']['photometric']['enable']:
                optical = augmentation.photometric_augmentation(optical, **self.config['augmentation']['photometric'])
                thermal = augmentation.photometric_augmentation(thermal, **self.config['augmentation']['photometric'])

            if self.config['augmentation']['homographic']['enable'] :
                # randomly pick one image to warp
                my_keypoints = [None,None]
                if (keypoints is not None):
                    my_keypoints = keypoints
                    if type(keypoints) is not list:
                        my_keypoints = [keypoints,keypoints] #added for working in non window cases
                if bool(random.randint(0,1)):
                    valid_mask_thermal = augmentation.dummy_valid_mask(thermal.shape)
                    
                    keypoints_thermal = my_keypoints[1]
                    optical, keypoints_optical, valid_mask_optical, H = augmentation.homographic_augmentation(optical,
                                                                                                              my_keypoints[0],
                                                                                                              return_homography = True,
                                                                                                              **self.config['augmentation']['homographic'])
                    out['optical']['homography'] = torch.from_numpy(H.astype(np.float32))
                    out['thermal']['homography'] = torch.eye(3, dtype=torch.float32)
                else:
                    valid_mask_optical = augmentation.dummy_valid_mask(optical.shape)
                    keypoints_optical = my_keypoints[0]
                    thermal, keypoints_thermal, valid_mask_thermal, H = augmentation.homographic_augmentation(thermal,
                                                                                                              my_keypoints[1],
                                                                                                              return_homography = True,
                                                                                                              **self.config['augmentation']['homographic'])
                    out['thermal']['homography'] = torch.from_numpy(H.astype(np.float32))
                    out['optical']['homography'] = torch.eye(3, dtype=torch.float32)
                

                myopt = torch.from_numpy(np.expand_dims(optical,0).astype(np.float32))
                myth = torch.from_numpy(np.expand_dims(thermal,0).astype(np.float32))
                #here added by me for homography regression
                patch_size = self.config["augmentation"]["homographic"]["params"]["corner_homography"]["params"]["patch_size"]
                out['hm_input'],out["hfour_points"] = self.prep_hm_regression_input(myopt,myth,optical_homography=out['optical']['homography'], thermal_homography=out['thermal']['homography'],
                                                                    top_left_point=[h//2-64,w//2-64], patch_size_h_w=[128,128])
            else:
                if type(keypoints) is "list":
                    keypoints_optical = keypoints[0]
                    keypoints_thermal = keypoints[1]
                else:
                    keypoints_optical = keypoints
                    keypoints_thermal = keypoints
                valid_mask_optical = valid_mask_thermal = augmentation.dummy_valid_mask(optical.shape)
            
            # add channel information to image and mask
            optical = np.expand_dims(optical, 0)
            thermal = np.expand_dims(thermal, 0)
            valid_mask_optical = np.expand_dims(valid_mask_optical, 0)
            valid_mask_thermal = np.expand_dims(valid_mask_thermal, 0)
    

            out['optical']['image'] = torch.from_numpy(optical.astype(np.float32))
            out['optical']['valid_mask'] = torch.from_numpy(valid_mask_optical.astype(bool))
            out['optical']['is_optical'] = torch.BoolTensor([optical_is_optical])

            keypoints_optical =None
            keypoints_thermal = None
            if keypoints is not None:
                if type(keypoints) is list:
                    keypoints_optical  = keypoints[0]
                    keypoints_thermal = keypoints[1]
                else:
                    keypoints_optical = keypoints
                    keypoints_thermal = keypoints

            if keypoints_optical is not None:
                keypoints_optical_map = utils.generate_keypoint_map(keypoints_optical, (h,w))
                out['optical']['keypoints'] = torch.from_numpy(keypoints_optical_map.astype(bool))
                #out['optical']['keypoints_coordinates'] = keypoints_optical #this causes problem in the dataloader

            out['thermal']['image'] = torch.from_numpy(thermal.astype(np.float32))
            out['thermal']['valid_mask'] = torch.from_numpy(valid_mask_thermal.astype(bool))
            out['thermal']['is_optical'] = torch.BoolTensor([thermal_is_optical])

            if keypoints_thermal is not None:
                keypoints_thermal_map = utils.generate_keypoint_map(keypoints_thermal, (h,w))
                out['thermal']['keypoints'] = torch.from_numpy(keypoints_thermal_map.astype(bool))
                #out['thermal']['keypoints_coordinates'] = keypoints_thermal

            if "hm_input" in out.keys():
                out['hm_input'] = torch.from_numpy( out['hm_input'].astype(np.float32))

        if self.config['return_name']:
            out['name'] = self.memberslist[index]

        #import cv2
        # h,w = out["optical"]["image"].shape[1:]
        # opt_img = np.uint8(out["optical"]["image"][0].cpu().numpy()*255)
        # th_img = np.uint8(out["thermal"]["image"][0].cpu().numpy()*255)

        # #print(out["optical"]["homography"],out["thermal"]["homography"])
        
        # patch_size = 64
        # top_left_point = [0,0] #w//2-patch_size,h//2-patch_size
        # top_right_point = [w,0] #w//2+patch_size,h//2-patch_size
        # bottom_left_point = [0,h]#w//2-patch_size,h//2+patch_size
        # bottom_right_point = [w,h]#w//2+patch_size,h//2+patch_size
        # four_points = [top_left_point, top_right_point, bottom_right_point, bottom_left_point]

        # perturbed_four_points = []
        # for point in four_points:
        #     point_hom = np.array([[point[0]], [point[1]], [1]])
        #     point_hom_transformed = out["thermal"]["homography"].cpu().numpy() @ out["optical"]["homography"].cpu().numpy() @ point_hom
        #     perturbed_four_points.append([int(point_hom_transformed[0]),int(point_hom_transformed[1])])
        # #print(perturbed_four_points)
        # #print(four_points)
        # H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
        #print("Diff : ", np.abs(H_four_points))
        #point_hom = np.array([[top_left], [y1], [1]])
        #point_hom_transformed = out["thermal"]["homography"].cpu().numpy() @ out["optical"]["homography"].cpu().numpy() @ point_hom
        # print(point_hom_transformed)
        # print(opt_img)
        #x2, y2 = int(point_hom_transformed[0]), int(point_hom_transformed[1])

        # Draw a circle at the corresponding point in the second image
        # for point in four_points:
        #     cv2.circle(opt_img, (point[0], point[1]), 5, (0, 0, 255), -1)
        # for point in perturbed_four_points:
        #     cv2.circle(th_img, (point[0], point[1]), 5, (0, 0, 255), -1)
        # # cv2.circle(opt_img, (x2, y2), 5, (0, 0, 255), -1)
        # # cv2.circle(th_img, (x1, y1), 5, (0, 0, 255), -1)

        # # Display the images
        # cv2.imshow("Ground truth", opt_img)
        # cv2.imshow("Test image", th_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        #print(out['optical']['homography'])
        #print(out['thermal']['homography'])





        # from matplotlib import pyplot as plt
        # h,w = out["optical"]["image"].shape[1:]
        # print("H,W : ",h,w)
        # out_thermal = cv2.cvtColor((np.clip(out['thermal']['image'].squeeze().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
        # out_optical = cv2.cvtColor((np.clip(out['optical']['image'].squeeze().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
        # #print("optical shape, thermal shape: ", out_optical.shape, out_thermal.shape)
        # if 1: #labels_optical.shape[0] < 200 or labels_thermal.shape[0] < 200:
        #     print("Number of keypoints optical: {}".format(keypoints_optical.shape[0]))
        #     print("Number of keypoints thermal: {}".format(keypoints_thermal.shape[0]))

        # predictions_optical = [cv2.KeyPoint(c[1], c[0], 3) for c in keypoints_optical.astype(np.float32)]
        # predictions_thermal = [cv2.KeyPoint(c[1], c[0], 3) for c in keypoints_thermal.astype(np.float32)]

        # # draw predictions and ground truth on image
        # out_optical = cv2.drawKeypoints(out_optical,
        #                                 predictions_optical,
        #                                 outImage=np.array([]),
        #                                 color=(0, 255, 0),
        #                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # out_thermal = cv2.drawKeypoints(out_thermal,
        #                                 predictions_thermal,
        #                                 outImage=np.array([]),
        #                                 color=(0, 255, 0),
        #                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # mask_optical = np.repeat(np.expand_dims(out['optical']['valid_mask'].squeeze().numpy(), axis=2), 3, axis=2)
        # mask_thermal = np.repeat(np.expand_dims(out['thermal']['valid_mask'].squeeze().numpy(), axis=2), 3, axis=2)
        # fig, axs = plt.subplots(1, 2, figsize=(8,8))


        # axs[0].imshow(out_thermal * mask_thermal)
        # axs[0].set_title('Thermal Masked')

        # axs[1].imshow(out_optical * mask_optical)
        # axs[1].set_title('Optical Masked')

        # # axs[1, 0].imshow(out_thermal)
        # # axs[1, 0].set_title('Thermal')

        # # axs[1, 1].imshow(out_optical)
        # # axs[1, 1].set_title('Optical')

        # plt.show()

        

        return out

    
    def prep_hm_regression_input(self, optical_data, thermal_data,optical_homography, thermal_homography, top_left_point=[0,0], patch_size_h_w=[128,128]):
        top_left_point = np.array(top_left_point)
        top_right_point = top_left_point+[patch_size_h_w[1],0] 
        bottom_left_point = top_left_point+[0,patch_size_h_w[0]]
        bottom_right_point = top_left_point+[patch_size_h_w[1],patch_size_h_w[0]]
        four_points = [top_left_point, top_right_point, bottom_right_point, bottom_left_point]
        
        # Create arrays to store the results
        H_four_points = []
        # Loop over the batch dimension
        perturbed_four_points = []
        for point in four_points:
            point_hom = np.array([[point[0]], [point[1]], [1]])
            point_hom_transformed = optical_homography @ thermal_homography @ point_hom
            perturbed_four_points.append([int(point_hom_transformed[0]),int(point_hom_transformed[1])])
        #print("four points : ",four_points)
        #print("pertub points : ",perturbed_four_points)
        H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points)) #/ self.config["augmentation"]["homographic"]["params"]["corner_homography"]["params"]["rho"]
        # print(four_points)
        # print(perturbed_four_points)
        #print(H_four_points)
        # print("-----")
        
        # Extract the x and y coordinates
        x = [top_left_point[0], top_right_point[0], bottom_right_point[0], bottom_left_point[0]]
        y = [top_left_point[1], top_right_point[1], bottom_right_point[1], bottom_left_point[1]]

        # Find the bounding box coordinates
        min_x = int(torch.min(torch.tensor(x)))
        max_x = int(torch.max(torch.tensor(x)))
        min_y = int(torch.min(torch.tensor(y)))
        max_y = int(torch.max(torch.tensor(y)))

        # Crop the image tensor        
        cropped_opt = optical_data[:, min_y:max_y, min_x:max_x]
        cropped_th = thermal_data[:, min_y:max_y, min_x:max_x]
        #print(H_four_points)
        # import cv2
        # cv2.imshow("opt",cropped_opt[0].numpy())
        # cv2.imshow("th",cropped_th[0].numpy())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        concatenated = np.concatenate((cropped_opt, cropped_th), axis=0)

        return concatenated,H_four_points



    def resize_image_and_keypoints(self, image, keypoints=None, min_size=256):
        """
        Resizes the image so that both sides are at least min_size pixels,
        and adjusts the keypoints accordingly if provided.

        Parameters:
        - image: Input grayscale image as a NumPy array.
        - keypoints: (Optional) NumPy array of keypoints with shape (N, 2), where N is the number of keypoints.
        - min_size: The minimum size for both sides of the image.

        Returns:
        - resized_image: The resized image.
        - adjusted_keypoints: The keypoints adjusted to the new image size, or None if keypoints was None.
        """
        h, w = image.shape[:2]

        # Check if resizing is necessary
        if h >= min_size and w >= min_size:
            # Ensure keypoints are within bounds
            if keypoints is not None:
                keypoints[:, 0] = np.clip(keypoints[:, 0], 0, w - 1)
                keypoints[:, 1] = np.clip(keypoints[:, 1], 0, h - 1)
            return image, keypoints

        # Compute the scaling factor to ensure both dimensions are at least min_size
        scale_h = min_size / h if h < min_size else 1
        scale_w = min_size / w if w < min_size else 1
        scale = max(scale_h, scale_w)

        # Calculate new dimensions and ensure they are at least min_size
        new_w = max(int(math.ceil(w * scale)), min_size)
        new_h = max(int(math.ceil(h * scale)), min_size)

        # Resize the image
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Adjust the keypoints if they are provided
        if keypoints is not None:
            # Adjust keypoints using individual scaling factors
            scale_x = new_w / w
            scale_y = new_h / h
            adjusted_keypoints = keypoints.copy()
            adjusted_keypoints = adjusted_keypoints.astype(np.float64)
            adjusted_keypoints[:, 0] *= scale_x  # x coordinates
            adjusted_keypoints[:, 1] *= scale_y  # y coordinates

            # Clip keypoints to be within the image bounds
            adjusted_keypoints[:, 0] = np.clip(adjusted_keypoints[:, 0], 0, new_w - 1)
            adjusted_keypoints[:, 1] = np.clip(adjusted_keypoints[:, 1], 0, new_h - 1)
            adjusted_keypoints = adjusted_keypoints.astype(np.int64)
        else:
            adjusted_keypoints = None
        

        return resized_image, adjusted_keypoints




    
    def get_name(self, index):
        return self.memberslist[index]

    def returns_pair(self):
        return not self.config['single_image']

    def __len__(self):
        return self.num_files
