import argparse
import copy
import cv2
import os
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
import yaml

import xpoint.datasets as datasets
import xpoint.models as models
import xpoint.utils as utils
import xpoint.utils.losses as losses

from pick_GPU import pickGPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def main():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('-y', '--yaml-config', default='configs/cmt.yaml', help='YAML config file')
    parser.add_argument('-w', '--weight-file', help='File containing the weights to initialize the weights')
    args = parser.parse_args()

    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # create output directory if it does not exist yet
    if not os.path.isdir(str(config['training']['output_directory'])):
        os.makedirs(str(config['training']['output_directory']))

    if any(keyword in config["model"]["use_attention"]["type"].lower() for keyword in ["swin", "vmamba"]) \
    and config["model"]["use_attention"].get("check") and config["model"]["use_attention"]["pretrained"].get("check"):
        pretrained_folder = config['model']['use_attention']['pretrained']['type_dir']
        yaml_files = [f for f in os.listdir(pretrained_folder) if f.endswith('.yaml')]
        config["model"]["use_attention"]["pretrained"]["yaml_file"] = os.path.join(pretrained_folder, yaml_files[0])
        with open(config["model"]["use_attention"]["pretrained"]["yaml_file"], 'r') as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        config["model"]["use_attention"]["model_parameters"] = model_config

    # dump the params
    with open(os.path.join(config['training']['output_directory'], 'params.yaml'), 'wt') as fh:
        yaml.safe_dump({'model': config['model'], 'loss': config['loss'], 'training': config['training'], 'dataset': config['dataset']}, fh)

    # check training device
    device = torch.device("cpu")
    if config['training']['allow_gpu']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training on device: {}'.format(device))

    # dataset
    dataset_class = getattr(datasets, config['dataset']['type'])
    dataset = dataset_class(config['dataset'])

    loader_trainset = torch.utils.data.DataLoader(dataset, batch_size=config['training']['batchsize'],
                                              shuffle=True, num_workers=config['training']['num_worker'])

    if config['training']['validation']['compute_validation_loss']:
        val_config = copy.copy(config['dataset'])
        val_config['filename'] = config['training']['validation']['filename']
        val_config['keypoints_filename'] = config['training']['validation']['keypoints']
        validation_dataset = dataset_class(val_config)
        loader_validationset = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['batchsize'],
                                                           shuffle=False, num_workers=config['training']['num_worker'])

    # network
    net = getattr(models, config['model']['type'])(config['model'])
    print("CUDA VERSION ", torch.version.__version__)
    start_epoch = 0

    if args.weight_file is not None:
        try:
            start_epoch = int(os.path.split(args.weight_file)[-1].split('.')[0][1:])
        except ValueError:
            pass
        weights = torch.load(args.weight_file, map_location=torch.device('cpu'))
        #weights = utils.fix_model_weigth_keys(weights)

        if config['model']['use_attention']['check'] and config['model']['use_attention']['pretrained']['check'] and config['model']['use_attention']["type"] == "Swinv2":
            if not net.encoder.register_buff:  
                substrings_to_remove = ["attn_mask", "relative_coords_table", "relative_position_index"]
                for key in list(weights.keys()):  # Using list to iterate over a copy of the keys
                    if any(sub in key for sub in substrings_to_remove):
                        weights.pop(key)
        missing_keys,unexpected_keys = net.load_state_dict(weights, strict=False)
    elif args.weight_file is None and config['model']['use_attention']['check'] and config['model']['use_attention']['pretrained']['check'] and config['model']['use_attention']["type"] == "Swinv2":
        # initialize weights of
        folder_path = config['model']['use_attention']['pretrained']['type_dir']
        files = os.listdir(folder_path)
        pth_files = [f for f in files if f.endswith('.pth')] 
        weight_path = os.path.join(config['model']['use_attention']['pretrained']['type_dir'], pth_files[0])  # assumed only 1 .pth file
        print(weight_path) 
        weights = torch.load(weight_path, map_location=torch.device('cpu'))
        if "model" in weights.keys():
            weights = weights["model"]
        if net.encoder.register_buff:
            weights.pop("head.weight")
            weights.pop("head.bias")
            print("head.weight and head.bias are removed from the weights")
        else:
            substrings_to_remove = ["attn_mask", "relative_coords_table", "relative_position_index"]
            for key in list(weights.keys()):  # Using list to iterate over a copy of the keys
                if any(sub in key for sub in substrings_to_remove):
                    weights.pop(key)
        
        missing_keys,unexpected_keys = net.encoder.load_state_dict(weights, strict=False)
        print("[INFO]\n{} parameters are loaded to Swinv2 architecture!".format(weight_path.split("/")[-1]))
        
    elif args.weight_file is None and config['model']['use_attention']['check'] and config['model']['use_attention']['pretrained']['check'] and config['model']['use_attention']["type"] == "VMamba":
        folder_path = config['model']['use_attention']['pretrained']['type_dir']
        files = os.listdir(folder_path)
        pth_files = [f for f in files if f.endswith('.pth')] 
        weight_path = os.path.join(config['model']['use_attention']['pretrained']['type_dir'], pth_files[0])  # assumed only 1 .pth file
        print(weight_path) 
        weights = torch.load(weight_path, map_location=torch.device('cpu'))  # ["model"]
        if "state_dict" in weights:
            weights = weights["state_dict"]
        if "model" in weights:
            weights = weights["model"]
        
        
        
        #if weight keys starts with "backbone." remove it
        for key in list(weights.keys()):
            if key.startswith("backbone."):
                weights[key.replace("backbone.", "")] = weights.pop(key)

        # Load the weights into the model
        missing_keys, unexpected_keys = net.encoder.load_state_dict(weights, strict=False)
        print("[INFO]\n{} parameters are loaded to VMamba architecture!".format(weight_path.split("/")[-1]))
    

    if args.weight_file is not None:
        # Count the successfully loaded weights
        loaded_keys = set(weights.keys()) - set(missing_keys)
        print(f"Successfully loaded {len(loaded_keys)} keys.")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")

        if len(loaded_keys) < 1:
            raise ValueError("No weights were loaded correctly! Please check the model and weights file.")
    

    # if not utils.check_loaded_weights(net,weights):
    #     raise ValueError("No weights were loaded correctly! Please check the model and weights file.")

    # exit()
    
    takes_pair = net.takes_pair()
    if config['training']['allow_gpu'] and (torch.cuda.device_count() > 1):
        print("Using ", torch.cuda.device_count(), " GPUs to train the model")
        net = torch.nn.DataParallel(net)
    
    net.to(device)

    # Use AMP if enabled in config
    use_amp = config['training']['mixed_precision']
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # loss
    try:
        config["loss"]["space_to_depth_ratio"] = net.module.get_encoder_downsample_ratio()
    except AttributeError:
        config["loss"]["space_to_depth_ratio"] = net.get_encoder_downsample_ratio()

    loss_fn = getattr(losses, config['loss']['type'])(config['loss'])
    loss_input_dict = {
        'data' : None,
        'pred' : None,
        'pred2' : None,
        'pred_hm' : None,
    }

    # optimizer
    # Freeze the encoder layers
    # for param in net.encoder.parameters():
    #     param.requires_grad = False
    #print("Parameters that are trainable: ", [name for name, param in net.named_parameters() if param.requires_grad])
    print("Parameters that are not trainable: ", [name for name, param in net.named_parameters() if not param.requires_grad])

    optimizer = torch.optim.Adam(net.parameters(), lr=float(config['training']['learningrate']), weight_decay=float(config['training']['weight_decay']))

    if config['training']['scheduler']['use_scheduler']:
        from torch.optim import lr_scheduler


        # Initialize scheduler based on config
        scheduler_config = config['training']['scheduler']
        if scheduler_config['type'] == 'StepLR':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_config['step_size'], gamma=scheduler_config['gamma'])
        elif scheduler_config['type'] == 'ExponentialLR':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_config['gamma'])
        # Add other schedulers as needed


    # initialize writer if requested
    if config['training']['use_writer']:
        writer = SummaryWriter(os.path.join(config['training']['output_directory'], 'learningcurve'))

    num_iter = len(loader_trainset) * (config['training']['n_epochs'] - start_epoch)
    


    with tqdm(total=num_iter) as pbar:
        for epoch in range(start_epoch, config['training']['n_epochs']):
            epoch_loss = 0.0
            #epoch_hm_loss = 0.0
            epoch_loss_components = {}
            epoch_val_loss = 0.0
            epoch_val_loss_components = {}

            for i, data in enumerate(loader_trainset, 0):
                # move data to correct device
                data = utils.data_to_device(data, device)

                # zero the parameter gradients
                optimizer.zero_grad()

                loss_input_dict['data'] = data
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if dataset.returns_pair():
                        if not takes_pair:
                            pred_optical = net(data['optical'])
                            pred_thermal = net(data['thermal'])
                        else:
                            pred_optical, pred_thermal, pred_hm = net(data)
                            loss_input_dict['pred_hm'] = pred_hm 
                        
                        loss_input_dict['pred'] = pred_optical
                        loss_input_dict['pred2'] = pred_thermal

                    else:
                        pred = net(data)
                        loss_input_dict['pred'] = pred
                    loss, loss_components = loss_fn(loss_input_dict)
                        

                # update params
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss_components = update_loss_components(epoch_loss_components, loss_components)
                epoch_loss += loss.item()
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_description('epoch {}/{}, batch {}/{} - epoch_loss: {},hm_reg_loss: {}, batch_loss: {}, lr: {}'.format(
                    epoch+1, config['training']['n_epochs'],
                    i+1, len(loader_trainset),
                    epoch_loss / (i + 1),
                    epoch_loss_components['homography_regression_loss'] / (i + 1) if 'homography_regression_loss' in epoch_loss_components else 0,
                    loss.item(),
                    current_lr))
                pbar.update(1)

                #store the loss every batch
                if config['training']['use_writer']:
                    writer.add_scalar('batch_loss', loss.item(), epoch * len(loader_trainset) + i + 1)
                    for key in loss_components.keys():
                        writer.add_scalar('batch_loss/' + key, loss_components[key], epoch * len(loader_trainset) + i + 1)
                        
           


            if config['training']['validation']['compute_validation_loss']: #HERE I ASSUME NO VALID LOSS FOR ENCODER SIM & HM REGRESSION!!!!!
                if epoch % config['training']['validation']['every_nth_epoch'] == 0:
                    with torch.no_grad():
                        for k, data in enumerate(loader_validationset, 0):
                            data = utils.data_to_device(data, device)
                            loss_input_dict['data'] = data
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                if dataset.returns_pair():
                                    if not takes_pair:
                                        pred_optical = net(data['optical'])
                                        pred_thermal = net(data['thermal'])
                                    else:
                                        pred_optical, pred_thermal, pred_hm = net(data)
                                        loss_input_dict['pred_hm'] = pred_hm 
                                    
                                    loss_input_dict['pred'] = pred_optical
                                    loss_input_dict['pred2'] = pred_thermal

                                else:
                                    pred = net(data)
                                    loss_input_dict['pred'] = pred
                                loss, loss_components = loss_fn(loss_input_dict)

                            epoch_val_loss_components = update_loss_components(epoch_val_loss_components, loss_components)
                            epoch_val_loss += loss.item()

                        if config['training']['use_writer']:
                            writer.add_scalar('validation_loss', epoch_val_loss / len(loader_validationset), epoch + 1)
                            for key in epoch_val_loss_components.keys():
                                writer.add_scalar('validation_loss/' + key, epoch_val_loss_components[key] / len(loader_validationset), epoch + 1)

            # store the loss every epoch
            if config['training']['use_writer']:
                writer.add_scalar('loss', epoch_loss / len(loader_trainset), epoch + 1)
                for key in epoch_loss_components.keys():
                    writer.add_scalar('loss/' + key, epoch_loss_components[key] / len(loader_trainset), epoch + 1)
                current_lr = optimizer.param_groups[0]['lr']
                #writer.add_scalar('loss/Cosine Loss', loss_encoder_sim / len(loader_trainset), epoch + 1)
                #writer.add_scalar('loss/HM Regressor Loss', epoch_hm_loss / len(loader_trainset), epoch + 1)
                writer.add_scalar('learning_rate', current_lr, epoch + 1)

            # save model every save_model_every_n_epoch epochs
            if ((epoch + 1) % config['training']['save_every_n_epoch'] == 0) and config['training']['save_every_n_epoch'] > 0:
                try:
                    state_dict = net.module.state_dict()  # for when the model is trained on multi-gpu
                except AttributeError:
                    state_dict = net.state_dict()
                torch.save(state_dict, os.path.join(config['training']['output_directory'], 'e{}.model'.format(epoch + 1)))
                
            if config['training']['scheduler']['use_scheduler']:
                scheduler.step()
            # print('epoch {}/{}, epoch_loss: {}, epoch_HM_loss: {}'.format(
            #     epoch + 1, config['training']['n_epochs'],
            #     epoch_loss / len(loader_trainset),
            #     epoch_hm_loss / len(loader_trainset)))

    try:
        state_dict = net.module.state_dict()  # for when the model is trained on multi-gpu
    except AttributeError:
        state_dict = net.state_dict()
    torch.save(state_dict, os.path.join(config['training']['output_directory'], 'latest.model'.format(epoch + 1)))

def update_loss_components(epoch_loss_components, loss_components):
    for key in loss_components.keys():
        if key in epoch_loss_components.keys():
            epoch_loss_components[key] += loss_components[key]
        else:
            epoch_loss_components[key] = loss_components[key]
    return epoch_loss_components

if __name__ == "__main__":
    main()
