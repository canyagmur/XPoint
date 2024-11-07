import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import yaml

import xpoint.datasets as datasets
import xpoint.models as models
import xpoint.utils as utils

# from pick_GPU import pickGPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #str(pickGPU())

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def main():
    parser = argparse.ArgumentParser(description='Predict the keypoints of an image')
    parser.add_argument('-y', '--yaml-config', default='configs/cipdp.yaml', help='YAML config file')
    parser.add_argument('-m', '--model-dir', default='model_weights/xpoint', help='Directory of the model')
    parser.add_argument('-v', '--version', default='latest', help='Model version (name of the param file), none for no weights')
    parser.add_argument('-i', '--index', default=0, type=int, help='Index of the sample to predict and show')
    parser.add_argument('-r', '--radius', default=4, type=int, help='Radius of the keypoint circle')
    parser.add_argument('-p', dest='plot', action='store_true', help='If set the prediction the results_descriptor are displayed')
    parser.add_argument('-e', dest='evaluation', action='store_true', help='If set the evaluation metrics are computed')
    parser.add_argument('-tk', dest='threshold_keypoints', default=4, type=int, help='Distance below which two keypoints are considered a match')
    parser.add_argument('-th', dest='threshold_homography', default=2, type=int, help='Homography correctness threshold')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Seed of the random generators')
    parser.add_argument('-yv', '--yaml-variable', help='YAML variable config file for my bash SCRIPTS') #ADDED BY ME
    parser.add_argument('-o', '--output_dir', default='outputs', help='output file')


    #below used for keypoint prediction
    parser.add_argument('-t', dest='threshold', default=3, type=int, help='Distance threshold for two keypoints to be considered a match')
    parser.add_argument('-mask', dest='mask', action='store_true', help='If set invalid image pixels will be set to 0')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(args.model_dir, 'params.yaml'), 'r') as f:
        # overwrite the model params
        config['model'] = yaml.load(f, Loader=yaml.FullLoader)['model']

    try:
        config['model']['homography_regression_head']['check'] = not bool(config['prediction']['disable_hmhead'])
    except:
        pass #for models without hmhead to work 
    
        
    #ADDED BY ME
    if args.yaml_variable:
        with open(args.yaml_variable, 'r') as f:
            # overwrite the model params
            myconfig = yaml.load(f, Loader=yaml.FullLoader)
            import copy
            config = utils.dict_update(config, myconfig)

    if "use_attention" in config["model"].keys() and config["model"]["use_attention"]["check"]:
        pretrained_height,pretrained_width = config["model"]["use_attention"]["height"],config["model"]["use_attention"]["width"]
        if "model_parameters" in config["model"]["use_attention"].keys():
            config["model"]["use_attention"]["model_parameters"]["DATA"]["IMG_SIZE"] =(pretrained_height,pretrained_width)

        config["model"]["use_attention"]["height"] = config["dataset"]["height"]
        config["model"]["use_attention"]["width"] = config["dataset"]["width"]

    # check training device
    device = torch.device("cpu")
    if config['prediction']['allow_gpu']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Predicting on device: {}'.format(device))

    # dataset
    dataset = getattr(datasets, config['dataset']['type'])(config['dataset'])
    subset_dataset = torch.utils.data.Subset(dataset, [0,1,2])
    loader_dataset = torch.utils.data.DataLoader(dataset, batch_size=config['prediction']['batchsize'],
                                                 shuffle=False, num_workers=config['prediction']['num_worker'])
    


    # network
    net = getattr(models, config['model']['type'])(config['model'])
    weights = None
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

    if weights:
        missing_keys, unexpected_keys = net.load_state_dict(weights,strict=False)
        # Count the successfully loaded weights
        loaded_keys = set(weights.keys()) - set(missing_keys)
        print(f"Successfully loaded {len(loaded_keys)} keys.")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")

        if len(loaded_keys) < 1:
            raise ValueError("No weights were loaded correctly! Please check the model and weights file.")

    net.to(device)

    # put the network into the evaluation mode
    net.eval()
    

    #important for swin
    with torch.no_grad():
        print(args.output_dir)
        print("MODEL : ",args.model_dir)
        threshold_homography_list = [1,2,3,4,5,6,7,8,9,10] #args.threshold_homography
        thresh_repeatability = [1,2,3,4,5,6,7,8,9,10]  #args.threshold
        threshold_keypoints = [1,2,3,4,5,6,7,8,9,10] #args.threshold_keypoints
        ransac_reproj_thresholds = [2] #args.threshold_homography
        keypoint_detection_threshold =[config["prediction"]["detection_threshold"]] #[0.001]
        print("Keypoint detection threshold : ",keypoint_detection_threshold)
        
        #[133, 167, 89, 63, 22] vedai 512 worst
        #[160, 182, 184, 162, 65, 24, 198, 149, 186, 185] vedai 512 best

        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        args.index = list(np.random.randint(0,len(loader_dataset),size=5)) #[133, 167, 89, 63, 22] #[2450 ,1915,  878 , 406 , 875 ,1623 ,2446, 3052  ,  8  , 12]#1,11,111,1111,2222,3333,1234,2345,3210]
        print("Experimenting on :", len(args.index)," random indices.")
        print("First 10 index : ",args.index[:10])

        #ONE PREDICTION START
        #predict for descriptor metrics for one sample!!!
        time_dict_sec =  utils.desc_process_and_display_sample(net, dataset, device, config, args)
        mean_time_dict_sec_HZ = {}
        print("-----------------------------")
        print("---------Time experiment on : ",args.model_dir.split("/")[-1])
        total_runtime = 0
        for experiment_name,time_list in time_dict_sec.items():
            time_list = np.array(time_list)
            #maximum_three_indices = np.argpartition(time_list, -3)[-3:]
            time_list_mean = time_list.mean() #time_list[maximum_three_indices].mean()
            print("Experiment {} took ms : {}, HZ: {} ".format(experiment_name,round(time_list_mean*1000,3),round(1/time_list_mean,3)))
            total_runtime += time_list_mean
            new_key_name = experiment_name+"_mean"
            mean_time_dict_sec_HZ[new_key_name] = str(round(time_list_mean*1000,3))+ " ms , "+str(round(1/time_list_mean,3))+" HZ"
        print("Total runtime : {} ms , {} HZ".format(round(total_runtime*1000,3),round(1/total_runtime,3)))

        print("\n\n")

        #predict for repeatability metrics for one sample!!!
        utils.repeatability_process_and_display_sample(net, dataset, device, config, args)
        #ONE PREDICTION END




        if args.evaluation:
            model_name = os.path.split(args.model_dir.strip("/"))[-1]
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            folder_path = os.path.join(args.output_dir,model_name) + "_" + timestamp
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)
            for detection_threshold in keypoint_detection_threshold:
                out = utils.compute_metrics(net, loader_dataset, device, config,detection_threshold ,thresh_repeatability=thresh_repeatability,thresh_keypoints = threshold_keypoints,  thresh_warp =threshold_homography_list, ransac_reproj_thresholds=ransac_reproj_thresholds)
                # also add the params to store them
                out['config'] = config
                #repeatability START
                print("-----------repeatability Results:--------------")
                results_repeatability = out["repeatability"]
                #results_repeatability['distance_threshold'] = thresh_repeatability
                print('Repeatability: {}'.format(results_repeatability['repeatability_mean']))
                print('Number of optical keypoints: {}'.format(results_repeatability['n_kp_optical']))
                print('Number of thermal keypoints: {}'.format(results_repeatability['n_kp_thermal']))
                #repeatability END

                #descriptor START
                print("-------Descriptor Results:-------------")
                results_descriptor = out["descriptor"]
                for key in results_descriptor.keys():
                    print('th kp : {}, NN-mAP: {}'.format(key,results_descriptor[key]['nn_map']))
                    print('th kp : {}, M-Score: {}'.format(key,results_descriptor[key]['m_score']))

                #homography correctness START
                print("-------Homography correctness Results:-------------")
                results_homography = out["homography"]
                for ransac_reproj_th in results_homography.keys():
                    print('th ransac reprojection : {}, Homography correctness: {}'.format(key,results_homography[ransac_reproj_th]['h_correctness']))

                


                #save results
                import json
                import copy

                # mytarget_dir = os.path.join(args.model_dir, 'descriptor_evaluation',"results")
                # if not os.path.isdir(mytarget_dir):

                #     os.makedirs(mytarget_dir)
                keys_to_copy = ['nn_map', 'm_score']
                # for key,value in results_descriptor.items():
                #     for key2,value2 in value.items():
                #         print(key,key2)
                
                myresults = {}
                descriptor_written_results = {"th_kp_{}".format(th_kp) :{k: copy.deepcopy(results_descriptor[th_kp][k]) for k in keys_to_copy if k in results_descriptor[th_kp]} for th_kp in results_descriptor.keys()}
                
                myresults["repeatability"] = results_repeatability
                myresults["descriptor"] = descriptor_written_results
                myresults["homography"] = results_homography

                #print(myresults)
                
                

                myresults["model_dir"] = args.model_dir
                myresults["model_version"] = args.version
                myresults["height-width"] = "{},{}".format(config["dataset"]["height"],config["dataset"]["width"])
                myresults["dataset"] = config["dataset"]["filename"] if "filename" in config["dataset"] and config["dataset"]["filename"]  else config["dataset"]["foldername"]
                #myresults["reprojection_threshold"] = config["prediction"]["reprojection_threshold"]
                myresults["nms"] = config["prediction"]["nms"]
                myresults["detection_th"] = detection_threshold
                myresults['threshold_keypoints_for_descriptor'] = threshold_keypoints 
                myresults['threshold_homography_epsilon'] = threshold_homography_list
                myresults['threshold_repeatability_distance_th'] = thresh_repeatability
                myresults['ransac_reproj_thresholds'] = ransac_reproj_thresholds


                # Save dictionary to txt file

                #print(results_descriptor['matching_kp_numbers'])
                #folder_path = mytarget_dir  
                #file_name = os.path.split(args.model_dir.strip("/"))[-1] + '_' + args.version+".txt"
                filename = "detection_threshold_{}".format(detection_threshold)+ ".txt"
                #unique_file_name =utils.get_new_filename(args.output_dir, file_name)
                #myresults_dir = os.path.join(mytarget_dir,unique_file_name)
                myoutput_dir = os.path.join(folder_path,filename)




                # if args.plot:
                #     #this is for descriptor metrics
                #     plt.figure()
                #     plt.title('PR curve')
                #     plt.xlabel('precision')
                #     plt.ylabel('recall')
                #     plt.plot(results_descriptor['recall_optical'], results_descriptor['precision_optical'], 'r')
                #     plt.plot(results_descriptor['recall_thermal'], results_descriptor['precision_thermal'], 'g')
                #     plt.legend(['optical', 'thermal'])

                #     # plt.figure()
                #     # plt.title('Optical M-score')
                #     # plt.hist(results_descriptor['m_score_optical'], 50)

                #     # plt.figure()
                #     # plt.title('Thermal M-score')
                #     # plt.hist(results_descriptor['m_score_thermal'], 50)

                #     # plt.figure()
                #     # plt.title('Warp point distance error')
                #     # plt.hist(results_descriptor['pts_dist'], 50)

                #     plt.show()

                #descriptor END

                for key,value in  mean_time_dict_sec_HZ.items():
                    myresults[key] = value
                # myresults['matching_kp_numbers'] = results_descriptor['matching_kp_numbers']
                # myresults['pts_dist'] = results_descriptor['pts_dist']
                with open(myoutput_dir, 'w') as file:
                    file.write(json.dumps(myresults, indent=4))
                print("-----done : ",myoutput_dir,"\n")

if __name__ == "__main__":
    main()
