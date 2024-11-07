import collections
import numpy as np
import torch
from torchvision.ops import nms
from torchvision.ops.boxes import batched_nms

import torch.nn as nn
import math

import os

def check_loaded_weights(net, weights):
    """
    Function to check if weights are loaded correctly into the model.
    
    Args:
    - net (torch.nn.Module): The model into which the weights are being loaded.
    - weights (dict): The state_dict containing the weights to be loaded.
    
    Raises:
    - ValueError: If no weights are loaded correctly.
    """


    # Initialize counters
    total_weights = len(weights)
    correctly_loaded_count = 0
    differing_weights_count = 0
    not_found_count = 0
    #print(weights.keys())
    # Iterate through the weights and compare them to the model's state_dict
    for name, param in net.state_dict().items():
        if name in weights:
            # Compare the values
            if torch.equal(param, weights[name]):
                correctly_loaded_count += 1
                #print(f"Layer '{name}' weights loaded correctly.")
            else:
                differing_weights_count += 1
                #print(f"Layer '{name}' weights differ from the loaded weights!")
        else:
            not_found_count += 1
            #print(f"Layer '{name}' not found in the loaded weights!")

    # Print a summary of the results
    print("\nSummary:")
    print(f"Total weights in loaded state_dict: {total_weights}")
    print(f"Correctly loaded weights: {correctly_loaded_count}")
    print(f"Differing weights: {differing_weights_count}")
    print(f"Weights not found in model: {not_found_count}")

    # Assert an error if no weights were loaded correctly
    if correctly_loaded_count < 1:
        return False #raise ValueError("No weights were loaded correctly! Please check the model and weights file.")
    return True

def get_new_filename(base_path, original_name):
    """
    Return a filename such that it doesn't collide with existing files.
    Append numbers as needed.
    """
    extension = os.path.splitext(original_name)[1]
    base_name = os.path.splitext(original_name)[0]
    counter = 1
    
    new_name = original_name
    while os.path.exists(os.path.join(base_path, new_name)):
        new_name = f"{base_name}_{counter}{extension}"
        counter += 1

    return new_name

def dict_update(d, u):
    """
    Update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def data_to_device(data, device):
    for key in data.keys():
        if type(data[key]) is torch.Tensor:
            data[key] = data[key].to(device)
        elif type(data[key]) is dict:
            data[key] = data_to_device(data[key], device)
    return data

def tensors_to_dtype(data, dtype):
    for key in data.keys():
        if type(data[key]) is torch.Tensor:
            data[key] = data[key].to(dtype)
        elif type(data[key]) is dict:
            data[key] = data_to_device(data[key], dtype)
    return data

def data_unsqueeze(data, dim):
    for key in data.keys():
        if type(data[key]) is torch.Tensor:
            data[key] = data[key].unsqueeze(dim)
        elif type(data[key]) is dict:
            data[key] = data_unsqueeze(data[key], dim)
    return data

def parse_primitives(names, all_primitives):
    p = all_primitives if (names == 'all') \
            else (names if isinstance(names, list) else [names])
    assert set(p) <= set(all_primitives)
    return p

def generate_keypoint_map(keypoints, image_shape):    
    tmp = keypoints.astype(np.int64)
    keypoint_map = np.zeros(image_shape, dtype=bool)
    keypoint_map[tmp[:,0],tmp[:,1]] = True
    return keypoint_map

def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        idx = ~np.isfinite(c)
        c[idx] = np.where(a[idx] == 0, 1, 0)  # -inf inf NaN
    return c

def depth_to_space(x, block_size):
    N, C, H, W = x.size()
    x = x.view(N, block_size, block_size, C // (block_size ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
    x = x.view(N, C // (block_size ** 2), H * block_size, W * block_size)  # (N, C//bs^2, H * bs, W * bs)
    return x

def space_to_depth(x, block_size):
    N, C, H, W = x.size()
    x = x.view(N, C, H // block_size, block_size, W // block_size, block_size)  # (N, C, H//bs, bs, W//bs, bs)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
    x = x.view(N, C * (block_size ** 2), H // block_size, W // block_size)  # (N, C*bs^2, H//bs, W//bs)
    return x

def box_nms(prob, size, min_prob, iou=0.1, keep_top_k=0, on_cpu=False):
    """Performs non maximum suppression on the heatmap by considering hypothetical
    bounding boxes centered at each pixel's location (e.g. corresponding to the receptive
    field). Optionally only keeps the top k detections.

    Arguments:
        prob: the probability heatmap, with shape `[(B, 1) ,H, W]`
        size: a scalar, the size of the bounding boxes.
        iou: a scalar, the IoU overlap threshold.
        min_prob: a threshold under which all probabilities are discarded before NMS.
        keep_top_k: an integer, the number of top scores to keep.
    """
    if not (len(prob.shape) == 2 or len(prob.shape) == 4):
        raise ValueError('The probability must be either 2D (H,W), or 4D (B, 1, H, W)')

    device = prob.device
    if on_cpu:
        prob = prob.cpu()

    points = (prob > min_prob).nonzero()
    scores = prob[points.split(1, dim=1)]
    batched = len(prob.shape) > 2
    if batched:
        boxes = torch.cat([ points[:,2:]-size * 0.5,  points[:,2:]+size * 0.5], dim=1)
        idxs = points[:,0]
        indices = batched_nms(boxes, scores[:,0], idxs, iou)
    else:
        boxes = torch.cat([points-size * 0.5, points+size * 0.5], dim=1)
        indices = nms(boxes, scores[:,0], iou)

    # nms already returns the sorted indices so just keep the first k elements
    if keep_top_k > 0:
        if batched:
            tmp = torch.ones((0), dtype=indices.dtype, device=indices.device)
            for i in range(prob.shape[0]):
                tmp = torch.cat((tmp, indices[idxs[indices] == i][:keep_top_k]), 0)
            indices = tmp
        else:
            indices = indices[:keep_top_k]

    # start with a zero tensor and fill in the probabilities where they are kept
    prob_nms = torch.zeros_like(prob)
    prob_nms[points[indices].split(1, dim=1)] = scores[indices]

    return prob_nms.to(device)

def get_gaussian_filter(kernel_size, sigma=None, channels = 1):
    if sigma is None:
        sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

def interpolate_descriptors(keypoints, descriptors_lowres, H, W):
    keypoints_ = keypoints.float().clone()

    keypoints_[:,0] = (keypoints_[:,0] / (float(H) * 0.5)) - 1.0
    keypoints_[:,1] = (keypoints_[:,1] / (float(W) * 0.5)) - 1.0
    keypoints_ = torch.flip(keypoints_.view(1, 1, -1, 2), [3])


    desc = torch.nn.functional.grid_sample(descriptors_lowres.unsqueeze(0), keypoints_, align_corners=True)[0,:,0,:].transpose(0,1)
    return torch.nn.functional.normalize(desc, p=2, dim=1)

def fix_model_weigth_keys(weights):
    new_weights = collections.OrderedDict()
    for key, value in weights.items():
        renamed = key.split('__')[-1]
        new_weights[renamed] = value

    return new_weights

def dict_to_obj(d):
    import types
    if isinstance(d, dict):
        return types.SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_obj(v) for v in d]
    else:
        return d

def get_yaml_object(path_to_type_dir):
    import os,yaml
    folder_path = path_to_type_dir
    files = os.listdir(folder_path)
    yaml_files = [f for f in files if f.endswith('.yaml')]
    config_file_path = os.path.join(folder_path,yaml_files[0])
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f) #yaml.load(f, Loader=yaml.FullLoader)

    return  dict_to_obj(config)

