import argparse
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from tqdm import tqdm
import os

import xpoint.datasets as data
import xpoint.models as models
import xpoint.utils as utils
import yaml

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='Show a sample of the dataset')
parser.add_argument('-d', '--dataset-path', required=True, help='Input dataset file')
parser.add_argument('-n', dest='sample_number', type=int, default=0, help='Sample to show')
parser.add_argument('-r', '--radius', default=4, type=int, help='Radius of the keypoint circle')
parser.add_argument('-m', '--model-dir', default='model_weights/surf', help='Directory of the model')
parser.add_argument('-v', '--version', default='none', help='Model version (name of the .model file)')

args = parser.parse_args()

config = {
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
    },
    'prediction': {
        'disable_hmhead': True,
        'allow_gpu': True,
        'num_worker': 4,
        'batchsize': 1,
        'detection_threshold': 0.015,
        'nms': 4,
        'cpu_nms': True,
        'topk': 0,
        'reprojection_threshold': 3,
        'matching': {
            'method': 'bfmatcher',
            'method_kwargs': {
                'crossCheck': True
            },
            'knn_matches': False
        }
    }
}

# # with open(args.yaml_config, 'r') as f:
# #     config = yaml.load(f, Loader=yaml.FullLoader)

with open(os.path.join(args.model_dir, 'params.yaml'), 'r') as f:
    # overwrite the model params
    config['model'] = yaml.load(f, Loader=yaml.FullLoader)['model']



# check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Predicting on device: {}'.format(device))

# network
net = getattr(models, config['model']['type'])(config['model'])
print(net)
assert net is not None, "Model not found"

if args.version != 'none':
    weights = torch.load(os.path.join(args.model_dir, args.version + '.model'), map_location=torch.device('cpu'))
    weights = utils.fix_model_weigth_keys(weights) 
    if  args.version != 'none' and "use_attention" in config["model"].keys() and config['model']['use_attention']["check"] == 1 \
                                and config['model']['use_attention']["type"] =="Swinv2":
        # Divide the weights into two dictionaries
        encoder_weights = {k.replace("encoder.",""): v for k, v in weights.items() if k.startswith("encoder")}
        other_weights = {k: v for k, v in weights.items() if not k.startswith("encoder")}
        net.load_state_dict(other_weights,strict=False)
        #net.encoder.load_state_dict(encoder_weights,strict=False)
        if net.encoder.register_buff: #this if is not necessary actually setting strict=False solves it but i want to do it explicitly
            net.encoder.load_state_dict(encoder_weights,strict=False) #True
        else:
            substrings_to_remove = ["attn_mask", "relative_coords_table", "relative_position_index"]
            for key in list(weights.keys()):  # Using list to iterate over a copy of the keys
                if any(sub in key for sub in substrings_to_remove):
                    weights.pop(key)

    missing_keys, unexpected_keys = net.load_state_dict(weights,strict=False)
    # Count the successfully loaded weights
    loaded_keys = set(weights.keys()) - set(missing_keys)
    print(f"Successfully loaded {len(loaded_keys)} keys.")
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")

    if len(loaded_keys) < 1:
        raise ValueError("No weights were loaded correctly! Please check the model and weights file.")



# move net to the right device
net.to(device)

# put the network into the evaluation mode
net.eval()
dataset = data.ImagePairDataset(config)


# display an individual sample
# get the data
sample = dataset[args.sample_number]
name = dataset.get_name(args.sample_number)
if ".jpg"  in name or ".png" in name:
    name = name[:-4]

sample = utils.data_to_device(sample, device)
sample = utils.data_unsqueeze(sample, 0)

if not net.takes_pair():
    out_optical = net(sample['optical'])
    out_thermal = net(sample['thermal'])
else :
    out_optical,out_thermal,out_hm = net(sample) # give both

nms = 8
# compute the nms probablity
if nms > 0:
    out_optical['prob'] = utils.box_nms(out_optical['prob'] * sample['optical']['valid_mask'],
                                        nms,
                                        config['prediction']['detection_threshold'],
                                        keep_top_k=config['prediction']['topk'],
                                        on_cpu=config['prediction']['cpu_nms'])
    out_thermal['prob'] = utils.box_nms(out_thermal['prob'] * sample['thermal']['valid_mask'],
                                        nms,
                                        config['prediction']['detection_threshold'],
                                        keep_top_k=config['prediction']['topk'],
                                        on_cpu=config['prediction']['cpu_nms'])

# add homography to data if not available
if 'homography' not in sample['optical'].keys():
    sample['optical']['homography'] =  torch.eye(3, dtype=torch.float32).to(device).view(sample['optical']['image'].shape[0],3,3)

if 'homography' not in sample['thermal'].keys():
    sample['thermal']['homography'] =  torch.eye(3, dtype=torch.float32).to(device).view(sample['optical']['image'].shape[0],3,3)

for i, (optical, thermal,
        prob_optical, prob_thermal,
        mask_optical, mask_thermal,
        H_optical, H_thermal,
        desc_optical, desc_thermal) in enumerate(zip(sample['optical']['image'],
                                                        sample['thermal']['image'],
                                                        out_optical['prob'],
                                                        out_thermal['prob'],
                                                        sample['optical']['valid_mask'],
                                                        sample['thermal']['valid_mask'],
                                                        sample['optical']['homography'],
                                                        sample['thermal']['homography'],
                                                        out_optical['desc'],
                                                        out_thermal['desc'],)):

    # get the keypoints
    pred_optical = torch.nonzero((prob_optical.squeeze() > config['prediction']['detection_threshold']).float())
    pred_thermal = torch.nonzero((prob_thermal.squeeze() > config['prediction']['detection_threshold']).float())
    kp_optical = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred_optical.cpu().numpy().astype(np.float32)]
    kp_thermal = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred_thermal.cpu().numpy().astype(np.float32)]


    import cv2
    import numpy as np
    myopt = np.uint8(sample["optical"]["image"][0][0].cpu().numpy()*255)
    myth = np.uint8(sample["thermal"]["image"][0][0].cpu().numpy()*255)
    myopt = cv2.drawKeypoints(myopt, kp_optical,
                                    outImage=np.array([]),
                                    color=(0, 0, 255),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    myth = cv2.drawKeypoints(myth, kp_thermal,
                                    outImage=np.array([]),
                                    color=(0, 0, 255),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("opt",myopt)
    cv2.imshow("th",myth)


    
    # get the descriptors
    if desc_optical.shape[1:] == prob_optical.shape[1:]:
        # classic descriptors, directly take values
        desc_optical_sampled = desc_optical[:, pred_optical[:,0], pred_optical[:,1]].transpose(0,1)
        desc_thermal_sampled = desc_thermal[:, pred_thermal[:,0], pred_thermal[:,1]].transpose(0,1)
    else:
        H, W = sample['optical']['image'].shape[2:]
        desc_optical_sampled = utils.interpolate_descriptors(pred_optical, desc_optical, H, W)
        desc_thermal_sampled = utils.interpolate_descriptors(pred_thermal, desc_thermal, H, W)

    # match the keypoints
    matches = utils.get_matches(desc_optical_sampled.cpu().numpy(),
                                desc_thermal_sampled.cpu().numpy(),
                                config['prediction']['matching']['method'],
                                config['prediction']['matching']['knn_matches'],
                                **config['prediction']['matching']['method_kwargs'])

    # mask the image if requested
    optical *= mask_optical
    thermal *= mask_thermal 

    # convert images to numpy arrays
    im_optical = cv2.cvtColor((np.clip(optical.squeeze().cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
    im_thermal = cv2.cvtColor((np.clip(thermal.squeeze().cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)



    # draw the matches
    out_image = cv2.drawMatches(im_optical, kp_optical, im_thermal, kp_thermal, matches, None, flags=2)
    #cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('matches', out_image.shape[1]*2, out_image.shape[0]*2 + 50)
    cv2.imshow('matches', out_image)

    # align images to estimate homography and get good matches
    optical_pts = np.float32([kp_optical[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    thermal_pts = np.float32([kp_thermal[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    #print("Above or equal to 4.5" if tuple(map(int, cv2.__version__.split('.')[:2])) >= (4, 5) else "Below 4.5")
    
    if optical_pts.shape[0] < 4 or thermal_pts.shape[0] < 4:
        H_est = np.eye(3,3)
        matchesMask = []
    else:
        if tuple(map(int, cv2.__version__.split('.')[:2])) >= (4, 5):
            #LONG LIVE MAGSAC!
            print("Using MAGSAC")
            H_est, mask = cv2.findHomography(
                                optical_pts,
                                thermal_pts,
                                method=cv2.USAC_MAGSAC,
                                ransacReprojThreshold=config['prediction']['reprojection_threshold'],
                                confidence=0.9999,
                                maxIters=10000,
                            )
        else:
            print("Using RANSAC")
            H_est, mask = cv2.findHomography(optical_pts, thermal_pts, cv2.RANSAC, ransacReprojThreshold=config['prediction']['reprojection_threshold'])
        matchesMask = mask.ravel().tolist()

    warped_image = cv2.warpPerspective(im_optical, H_est, im_optical.shape[:2][::-1], borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow('warped optical with estimated homography', warped_image)


    # correct matches mask
    H_gt = np.matmul(H_thermal.cpu().numpy(), np.linalg.inv(H_optical.cpu().numpy()))
    warped_optical = utils.warp_keypoints(optical_pts.squeeze()[:,::-1], H_gt)[:,::-1]
    diff = thermal_pts.squeeze() - warped_optical
    diff = np.linalg.norm(diff, axis=1)
    matchesMask = (diff < config['prediction']['reprojection_threshold']).tolist() # 4 is reprojection threshold i guess?? #matchesMask = (diff < 4.0).tolist()


    inlier_matches  = [matches[k] for k in range(len(matchesMask)) if matchesMask[k] == 1]
    # draw refined matches
    out_image_refined = cv2.drawMatches(im_optical,
                                        kp_optical,
                                        im_thermal,
                                        kp_thermal,
                                        inlier_matches,
                                        outImg=None,
                                        matchColor=(0, 255, 0),
                                        singlePointColor=(0, 0, 255),
                                        flags=0,)
                                        #matchesMask = matchesMask)

    #cv2.namedWindow('refined_matches', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('refined_matches', out_image_refined.shape[1]*2, out_image_refined.shape[0]*2 + 50)
    cv2.imshow('refined_matches', out_image_refined)

    # out_img_name=os.path.join(args.model_dir, 'descriptor_evaluation',"index_{}_matches.png".format(args.sample_number))
    # cv2.imwrite(out_img_name,out_image_refined)

        

    # compare estimated and computed homography
    print('--------------------------------------------------------')
    print('Estimated Homography:')
    print(H_est)
    print('Ground Truth Homography:')
    print(np.matmul(H_thermal.cpu().numpy(), np.linalg.inv(H_optical.cpu().numpy())))
    print('--------------------------------------------------------')

    cv2.waitKey(0)
