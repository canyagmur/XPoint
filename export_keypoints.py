import argparse
import h5py
import os
from tqdm import tqdm
import torch
import yaml

import xpoint.datasets as datasets
import xpoint.models as models
import xpoint.utils as utils


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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


def main():

    #-------------------GET ARGS FROM THE USER------------------
    parser = argparse.ArgumentParser(description='Script to export the keypoints for images in a dataset using a base detector')
    parser.add_argument('-y', '--yaml-config', default='configs/config_export_keypoints.yaml', help='YAML config file')
    parser.add_argument('-o', '--output_file', required=True, help='Output file name')
    parser.add_argument('-m', '--model-dir', default='model_weights/surf', help='Directory of the model')
    parser.add_argument('-v', '--version', default='none', help='Model version (name of the .model file)')
    parser.add_argument('-snms', '--single-nms', action='store_true', help='Do the nms calculation for each sample separately')
    parser.add_argument('-skip', dest='skip_processed', action='store_true', help='Skip already processed samples')
    parser.add_argument('-f', '--frequency', default=1,type=int, help='save every f epoch')

    args = parser.parse_args()

    #set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random

    random.seed(0)
    import numpy as np
    np.random.seed(0)
    #-----------------------------------------------------------

    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(args.model_dir, 'params.yaml'), 'r') as f:
        # overwrite the model params
        config['model'] = yaml.load(f, Loader=yaml.FullLoader)['model']
    # ---------------------------------------------

    #if folder does not exist create it
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    if os.path.isfile(args.output_file):
        print("Output file already exists : ", args.output_file)
        print("It will be overwritten...")
        if input("are you sure? (y/n)") != "y":
            os.remove(args.output_file)
    

    #if config has self.config['homography_regression_head']['check'] , then make it false
    if "homography_regression_head" in config["model"].keys():
        config["model"]["homography_regression_head"]["check"] = False

    # dataset
    dataset = getattr(datasets, config['dataset']['type'])(config['dataset'])

    # create output file
    backup_output_files = []
    output_file = h5py.File(args.output_file,"w")        
    if not args.skip_processed:
        if(args.frequency > 1):
            for i in range(1,len(dataset)//args.frequency+1):
                name = "{}_e{}.hdf5".format(args.output_file[:-5],i*args.frequency)
                backup_output_files.append(h5py.File(name,"w"))

    # check device
    device = torch.device("cpu")
    if config['prediction']['allow_gpu']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Predicting on device: {}'.format(device))


    # dataset
    n=5
    random_indices_n = [5632] 
    print(random_indices_n)
    loader_dataset = torch.utils.data.DataLoader(dataset, batch_size=config['prediction']['batchsize'],
                                                 shuffle=False, num_workers=config['prediction']['num_worker'])

    config["model"]["takes_pair"] = False
    try:
        config['model']['homography_regression_head']['check'] = not bool(config['prediction']['disable_hmhead'])
    except:
        pass #for models without hmhead to work 

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

    # multi gpu prediction
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    print("Using", torch.cuda.device_count(), "GPUs for prediction")

    # move net to the right device
    net.to(device)

    # put the network into the evaluation mode
    net.eval()
    with torch.no_grad():
        for epoch_counter,batch in enumerate(tqdm(loader_dataset),start=1):
            if args.skip_processed:
                all_processed = True
                for name in batch['name']:
                    all_processed = all_processed and (
                        name in output_file.keys())

                if all_processed:
                    continue

            # move data to device
            batch = utils.data_to_device(batch, device)

            # compute the homographic adaptation probabilities
            if dataset.returns_pair():
                out_dict = utils.homographic_adaptation_multispectral(
                    batch, net, config['prediction']['homographic_adaptation'])
            else:
                prob_ha = utils.homographic_adaptation(
                    batch, net, config['prediction']['homographic_adaptation'])
            
            prob_ha, prob_o, prob_t = out_dict["out"]["prob"] ,out_dict["out_optical"]["prob"],out_dict["out_thermal"]["prob"]
            desc_opt,desc_th = out_dict["desc_optical"],out_dict["desc_thermal"] 
            # compute the nms probablity
            if config['prediction']['nms'] > 0:
                if args.single_nms:
                    if config["prediction"]["homographic_adaptation"]['aggregation'] != 'window':
                        prob_ha = compute_box_nms(
                            prob_ha, config, single_nms=args.single_nms)
                    else:
                        prob_o = compute_box_nms(
                            prob_o, config, single_nms=args.single_nms)
                        prob_t = compute_box_nms(
                            prob_t, config, single_nms=args.single_nms)

                else:
                    if config["prediction"]["homographic_adaptation"]['aggregation'] != 'window':
                        prob_ha = compute_box_nms(prob_ha, config)
                    else:
                        prob_o = compute_box_nms(prob_o, config)
                        prob_t = compute_box_nms(prob_t, config)

            if config["prediction"]["homographic_adaptation"]['aggregation'] != 'window':
                for name, prob in zip(batch['name'], prob_ha.split(1)):
                    if not (args.skip_processed and (name in output_file.keys())):
                        
                        pred = torch.nonzero(
                            (prob.squeeze() > config['prediction']['detection_threshold']).float())
                        

                        output_file.create_group(name)
                        output_file[name].create_dataset(
                            'keypoints', data=pred.cpu().numpy())
                        


                        if (args.frequency > 1 and not args.skip_processed):
                            for file in backup_output_files:
                                file.create_group(name)
                                file[name].create_dataset(
                                    'keypoints', data=pred.cpu().numpy())
                                


                            if ((epoch_counter) % args.frequency == 0):
                                backup_output_files[0].close()
                                backup_output_files.pop(0)
            else:
                for name, prob_o, prob_t in zip(batch['name'], prob_o.split(1), prob_t.split(1)):
                    if not (args.skip_processed and (name in output_file.keys())):
                        pred_o = torch.nonzero(
                            (prob_o.squeeze() > config['prediction']['detection_threshold']).float())
                        pred_t = torch.nonzero(
                            (prob_t.squeeze() > config['prediction']['detection_threshold']).float())
                        
                        
                        output_file.create_group(name)
                        output_file[name].create_dataset(
                            'keypoints_optical', data=pred_o.cpu().numpy())
                        output_file[name].create_dataset(
                            'keypoints_thermal', data=pred_t.cpu().numpy())
                        
                        
                        if (args.frequency > 1 and not args.skip_processed):
                            for file in backup_output_files:
                                file.create_group(name)
                                file[name].create_dataset(
                                    'keypoints_optical', data=pred_o.cpu().numpy())
                                file[name].create_dataset(
                                    'keypoints_thermal', data=pred_t.cpu().numpy())
                                

                            if ((epoch_counter) % args.frequency == 0):
                                backup_output_files[0].close()
                                backup_output_files.pop(0)


    output_file.close()


if __name__ == "__main__":
    main()
