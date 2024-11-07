import argparse
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from tqdm import tqdm
import os

import xpoint.datasets as datasets
import xpoint.models as models
import xpoint.utils as utils
import yaml

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"





def compute_box_nms(prob, config, single_nms=False):
    if not single_nms:
        prob = utils.box_nms(prob,
                             config['prediction']['nms'],
                             config['prediction']['detection_threshold'],
                             keep_top_k=config['prediction']['topk'],
                             on_cpu=config['prediction']['cpu_nms'])
    else:
        for i, sample in enumerate(prob.split(1)):
            prob[i, 0] = utils.box_nms(sample.squeeze(),
                                       config['prediction']['nms'],
                                       config['prediction']['detection_threshold'],
                                       keep_top_k=config['prediction']['topk'],
                                       on_cpu=config['prediction']['cpu_nms'])
    return prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show a sample of the dataset')
    parser.add_argument('-d', '--dataset-path', required=True, help='Input dataset file')
    parser.add_argument('-k', '--keypoint-file', required=True, help='Keypoint dataset file')
    parser.add_argument('-n', dest='sample_number', type=int, default=0, help='Sample to show')
    parser.add_argument('-r', '--radius', default=4, type=int, help='Radius of the keypoint circle')
    parser.add_argument('-m', '--model-dir', default='model_weights/surf', help='Directory of the model')
    parser.add_argument('-v', '--version', default='none', help='Model version (name of the .model file)')

    args = parser.parse_args()


    config = {
        'dataset_type': 'ImagePairDataset',
        'filename': args.dataset_path if os.path.isfile(args.dataset_path) else None,
        'foldername' : args.dataset_path if os.path.isdir(args.dataset_path) else None,
        'keypoints_filename': args.keypoint_file,
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
            },
            "homographic_adaptation" : {
                'num': 10,
                'aggregation': 'window',
                'weighted_window': False,
                'window_size': 5,
                'erosion_radius': 3,
                'mask_border': True,
                'min_count': 5,
                'filter_size': 0,
                'homographies': {
                    'translation': True,
                    'rotation': True,
                    'scaling': True,
                    'perspective': True,
                    'scaling_amplitude': 0.2,
                    'perspective_amplitude_x': 0.2,
                    'perspective_amplitude_y': 0.2,
                    'patch_ratio': 0.85,
                    'max_angle': 1.57,
                    'allow_artifacts': True
                }
            }
        }
    }
    if config['dataset_type'] == 'SatThermalGeoDataset':
        config.remove('filename')
        config.remove('foldername')
        config['filename'] = "/home/wasproject/Desktop/Can/DATASETS/sat-thermal-geo/thermal_h5_datasets/test_database.h5 "
        config['filename_thermal'] = "/home/wasproject/Desktop/Can/DATASETS/sat-thermal-geo/thermal_h5_datasets/test_queries.h5"
        #config['dataset_type'] = 'SatThermalGeoDataset'


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

    # dataset
    dataset = getattr(datasets, config['dataset_type'])(config)
    print("Dataset length : ",len(dataset))

    # try:
    #     keypoint_file = h5py.File(args.keypoint_file, 'r', swmr=True)
    # except IOError as e:
    #     print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, args.keypoint_file))
    #     exit()


    indices_to_finetune = []
    for i in range(0, len(dataset)):
        sample = dataset[i]
        name = dataset.get_name(i)

        labels_optical_kpfile = sample["optical"]["keypoints_coordinates"]
        labels_thermal_kpfile = sample["thermal"]["keypoints_coordinates"]
        

        if len(labels_optical_kpfile) < 100 or len(labels_thermal_kpfile) < 100:
            indices_to_finetune.append(i)
    
    print("Found {} samples to finetune".format(len(indices_to_finetune)))



    for i in indices_to_finetune:
        sample = dataset[i]
        name = dataset.get_name(i)

        sample = utils.data_to_device(sample, device)
        sample = utils.data_unsqueeze(sample, 0)

        labels_optical_kpfile = sample["optical"]["keypoints_coordinates"]
        labels_thermal_kpfile = sample["thermal"]["keypoints_coordinates"]
        

        if len(labels_optical_kpfile) < 100 or len(labels_thermal_kpfile) < 100:
            gt_kp_opt =labels_optical_kpfile.shape[0]
            gt_kp_th =labels_thermal_kpfile.shape[0]
            print("Ground Truth Number of keypoints optical : {}".format(gt_kp_opt))
            print("Ground Truth Number of keypoints thermal : {}".format(gt_kp_th))
            

            # compute the homographic adaptation probabilities
            if dataset.returns_pair():
                out_dict = utils.homographic_adaptation_multispectral(
                    sample, net, config['prediction']['homographic_adaptation'])
            else:
                prob_ha = utils.homographic_adaptation(
                    sample, net, config['prediction']['homographic_adaptation'])
            
            prob_ha, prob_o, prob_t = out_dict["out"]["prob"] ,out_dict["out_optical"]["prob"],out_dict["out_thermal"]["prob"]
            desc_opt,desc_th = out_dict["desc_optical"],out_dict["desc_thermal"] #added by me 
            # compute the nms probablity
            if config['prediction']['nms'] > 0:
                if config["prediction"]["homographic_adaptation"]['aggregation'] != 'window':
                    prob_ha = compute_box_nms(prob_ha, config)
                else:
                    prob_o = compute_box_nms(prob_o, config)
                    prob_t = compute_box_nms(prob_t, config)

            if config["prediction"]["homographic_adaptation"]['aggregation'] != 'window':
                for name, prob in zip(sample['name'], prob_ha.split(1)):                    
                        pred_optical = pred_thermal = torch.nonzero(
                            (prob.squeeze() > config['prediction']['detection_threshold']).float())       
            else:
                for name, prob_o, prob_t in zip(sample['name'], prob_o.split(1), prob_t.split(1)):
                        pred_optical = torch.nonzero(
                            (prob_o.squeeze() > config['prediction']['detection_threshold']).float())
                        pred_thermal = torch.nonzero(
                            (prob_t.squeeze() > config['prediction']['detection_threshold']).float())






            pred_kp_opt = len(pred_optical)
            pred_kp_th = len(pred_thermal)            
            print("Inferred Number of keypoints optical : {} by model : {}".format(pred_kp_opt,args.model_dir))
            print("Inferred Number of keypoints thermal: {} by model : {}".format(pred_kp_th,args.model_dir))

            if pred_kp_opt > gt_kp_opt or pred_kp_th > gt_kp_th:
                print("Overwriting keypoints for sample named {}".format(name))
                extended_name = name + ".png"
                with open(args.keypoint_file, 'r+') as f:
                    keypoint_file = h5py.File(args.keypoint_file, 'r+', swmr=True)
                    if config["prediction"]["homographic_adaptation"]['aggregation'] == 'window':
                        keypoint_file[extended_name]['keypoints_optical'] = pred_optical
                        keypoint_file[extended_name]['keypoints_thermal'] = pred_thermal
                    else:
                        keypoint_file[extended_name]['keypoints'] = pred_optical #it is same for both optical and thermal

                    keypoint_file.close()                