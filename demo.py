import argparse
import cv2
import numpy as np
import os
import torch
import yaml
import matplotlib.pyplot as plt
import time
import json

import xpoint.models as models
import xpoint.utils as utils
from xpoint.utils.utils import box_nms, interpolate_descriptors
from xpoint.utils.matching import get_matches

def load_image(image_path, target_size=None):
    """Load and preprocess image."""

    #ensure image path is correct and image is found
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    if target_size:
        img = cv2.resize(img, target_size)
    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0
    return img

def process_image_pair(net, visible_img, other_img, config, device):
    """Process a pair of images through the network."""
    # Prepare input data
    data = {
        'optical': {
            'image': torch.from_numpy(visible_img).unsqueeze(0).unsqueeze(0).to(device),
            'valid_mask': torch.ones_like(torch.from_numpy(visible_img)).unsqueeze(0).unsqueeze(0).to(device)
        },
        'thermal': {
            'image': torch.from_numpy(other_img).unsqueeze(0).unsqueeze(0).to(device),
            'valid_mask': torch.ones_like(torch.from_numpy(other_img)).unsqueeze(0).unsqueeze(0).to(device)
        }
    }

    # Forward pass
    with torch.no_grad():
        if not net.takes_pair():
            out_visible = net(data['optical'])
            out_other = net(data['thermal'])
        else:
            out_visible, out_other, _ = net(data)

    # Apply NMS
    if config['prediction']['nms'] > 0:
        out_visible['prob'] = box_nms(
            out_visible['prob'] * data['optical']['valid_mask'],
            config['prediction']['nms'],
            config['prediction']['detection_threshold'],
            keep_top_k=config['prediction']['topk'],
            on_cpu=config['prediction']['cpu_nms']
        )
        out_other['prob'] = box_nms(
            out_other['prob'] * data['thermal']['valid_mask'],
            config['prediction']['nms'],
            config['prediction']['detection_threshold'],
            keep_top_k=config['prediction']['topk'],
            on_cpu=config['prediction']['cpu_nms']
        )

    return out_visible, out_other, data

def visualize_results(img_visible, img_other, kp_visible, kp_other, matches, output_path):
    """Create a comprehensive visualization with 3x2 subplot layout."""
    # Calculate the figure size based on image aspect ratio
    img_height, img_width = img_visible.shape
    aspect_ratio = img_width / img_height
    
    # Adjust figure size to maintain aspect ratio while keeping reasonable display size
    base_height = 8  # Reduced base height for better default window size
    fig_width = base_height * 2 * aspect_ratio  # 2 columns
    fig_height = base_height * 3  # 3 rows
    
    # Create figure with controlled size
    plt.figure(figsize=(min(fig_width, 15), min(fig_height, 20)))
    
    # Add small spacing between subplots
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    
    # 1. Original Images
    plt.subplot(3, 2, 1)
    plt.imshow(img_visible, cmap='gray')
    plt.title('Visible Spectrum Image', pad=10)
    plt.axis('off')
    
    plt.subplot(3, 2, 2)
    plt.imshow(img_other, cmap='gray')
    plt.title('Other Spectrum Image', pad=10)
    plt.axis('off')
    
    # 2. Images with Keypoints
    img_visible_kp = cv2.drawKeypoints(
        (img_visible * 255).astype(np.uint8),
        kp_visible,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    img_other_kp = cv2.drawKeypoints(
        (img_other * 255).astype(np.uint8),
        kp_other,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    plt.subplot(3, 2, 3)
    plt.imshow(cv2.cvtColor(img_visible_kp, cv2.COLOR_BGR2RGB))
    plt.title(f'Visible Image Keypoints ({len(kp_visible)} points)', pad=10)
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    plt.imshow(cv2.cvtColor(img_other_kp, cv2.COLOR_BGR2RGB))
    plt.title(f'Other Image Keypoints ({len(kp_other)} points)', pad=10)
    plt.axis('off')
    
    # 3. Matches
    img_visible_rgb = cv2.cvtColor((img_visible * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    img_other_rgb = cv2.cvtColor((img_other * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    matches_img = cv2.drawMatches(
        img_visible_rgb, kp_visible,
        img_other_rgb, kp_other,
        matches, None,
        matchColor=(0, 255, 0),  # Green color for matches
        singlePointColor=(255, 0, 0),  # Red color for single points
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    plt.subplot(3, 2, (5, 6))
    plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Matches ({len(matches)} correspondences)', pad=10)
    plt.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(output_path, bbox_inches='tight', dpi=300, pad_inches=0.5)
    
    # Display the plot
    plt.show()

def compute_homography_and_correctness(matches, kp_visible, kp_other, img_shape, ransac_threshold=3.0):
    """Compute homography and its correctness metrics."""
    if len(matches) < 4:
        return None, None, 0.0
    
    # Extract matched points
    visible_pts = np.float32([kp_visible[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    other_pts = np.float32([kp_other[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    # Estimate homography using RANSAC
    if tuple(map(int, cv2.__version__.split('.')[:2])) >= (4, 5):
        H_est, mask = cv2.findHomography(
            visible_pts, other_pts,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=0.9999,
            maxIters=10000,
        )
    else:
        H_est, mask = cv2.findHomography(visible_pts, other_pts, cv2.RANSAC, 
                                       ransacReprojThreshold=ransac_threshold)
    
    if H_est is None:
        return None, None, 0.0
    
    inliers = mask.ravel().tolist()
    inlier_ratio = sum(inliers) / len(inliers) if inliers else 0
    
    return H_est, inliers, inlier_ratio

def compute_repeatability(pred_visible, pred_other, H_est, img_shape, distance_thresh=3):
    """Compute repeatability score."""
    if H_est is None or len(pred_visible) == 0 or len(pred_other) == 0:
        return 0.0
    
    # Convert keypoints to numpy arrays
    kp_visible = pred_visible.cpu().numpy()
    kp_other = pred_other.cpu().numpy()
    
    # Warp visible keypoints to other frame
    ones = np.ones((kp_visible.shape[0], 1))
    kp_visible_homogeneous = np.hstack([kp_visible[:, [1,0]], ones])  # Note the swap of x,y
    warped_visible = H_est @ kp_visible_homogeneous.T
    warped_visible = warped_visible[:2, :] / warped_visible[2, :]
    warped_visible = warped_visible.T[:, [1,0]]  # Swap back to y,x
    
    # Filter points that are out of bounds after warping
    H, W = img_shape
    valid_mask = (warped_visible[:, 0] >= 0) & (warped_visible[:, 0] < H) & \
                (warped_visible[:, 1] >= 0) & (warped_visible[:, 1] < W)
    warped_visible = warped_visible[valid_mask]
    
    if len(warped_visible) == 0:
        return 0.0
    
    # Compute distances between warped visible and other keypoints
    warped_visible = np.expand_dims(warped_visible, 1)
    kp_other = np.expand_dims(kp_other, 0)
    distances = np.linalg.norm(warped_visible - kp_other, axis=2)
    
    # Count correct matches
    min_distances = np.min(distances, axis=1)
    correct_matches = (min_distances <= distance_thresh).sum()
    
    # Compute repeatability
    repeatability = correct_matches / min(len(pred_visible), len(pred_other))
    
    return float(repeatability)

def create_checkerboard_visualization(img_visible, img_other, H):
    """Create a checkerboard visualization of the alignment."""
    H_img, W_img = img_other.shape
    warped_visible = cv2.warpPerspective(img_visible, H, (W_img, H_img))
    
    # Create checkerboard pattern
    cell_size = 50  # Size of each checkerboard cell
    x, y = np.meshgrid(np.arange(W_img), np.arange(H_img))
    checker = ((x // cell_size) + (y // cell_size)) % 2
    
    # Create composite image
    composite = np.where(checker, warped_visible, img_other)
    return composite

def visualize_alignment(img_visible, img_other, H_est, output_path):
    """Visualize image alignment using estimated homography in both directions."""
    if H_est is None:
        return
    
    # Get image dimensions
    H, W = img_other.shape
    
    # Warp visible to other frame (forward warping)
    warped_visible = cv2.warpPerspective(img_visible, H_est, (W, H))
    
    # Warp other to visible frame (inverse warping)
    H_inv = np.linalg.inv(H_est)
    warped_other = cv2.warpPerspective(img_other, H_inv, (W, H))
    
    # Create visualization with both warpings
    plt.figure(figsize=(20, 10))
    
    # First row: Original images
    plt.subplot(231)
    plt.imshow(img_visible, cmap='gray')
    plt.title('Visible Spectrum Image')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(img_other, cmap='gray')
    plt.title('Other Spectrum Image')
    plt.axis('off')
    
    # Checkerboard visualization
    plt.subplot(233)
    checker = create_checkerboard_visualization(img_visible, img_other, H_est)
    plt.imshow(checker, cmap='gray')
    plt.title('Checkerboard Visualization')
    plt.axis('off')
    
    # Second row: Warped images and difference
    plt.subplot(234)
    plt.imshow(warped_visible, cmap='gray')
    plt.title('Visible Warped to Other')
    plt.axis('off')
    
    plt.subplot(235)
    plt.imshow(warped_other, cmap='gray')
    plt.title('Other Warped to Visible')
    plt.axis('off')
    
    # Add difference image
    plt.subplot(236)
    diff_img = np.abs(warped_visible - img_other)
    plt.imshow(diff_img, cmap='hot')
    plt.title('Difference Image')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

def compute_metrics(matches, pred_visible, pred_other, kp_visible, kp_other, time_dict, img_shape):
    """Compute various metrics for the matching results."""
    metrics = {}
    
    # Runtime metrics
    metrics['runtime'] = {
        'total_time': sum(time_dict.values()),
        'forward_pass_time': time_dict.get('forward_pass', 0),
        'keypoint_detection_time': time_dict.get('keypoint_detection', 0),
        'descriptor_computation_time': time_dict.get('descriptor_computation', 0),
        'matching_time': time_dict.get('matching', 0)
    }
    
    # Keypoint metrics
    metrics['keypoints'] = {
        'n_keypoints_visible': len(kp_visible),
        'n_keypoints_other': len(kp_other),
        'n_matches': len(matches)
    }
    
    # Matching score (ratio of matches to keypoints)
    if len(kp_visible) > 0 and len(kp_other) > 0:
        metrics['matching_score'] = len(matches) / min(len(kp_visible), len(kp_other))
    else:
        metrics['matching_score'] = 0.0
    
    # Distribution of matches
    if matches:
        distances = [m.distance for m in matches]
        metrics['match_statistics'] = {
            'min_distance': float(min(distances)),  # Convert to float for JSON serialization
            'max_distance': float(max(distances)),
            'mean_distance': float(sum(distances) / len(distances)),
            'median_distance': float(sorted(distances)[len(distances)//2])
        }
    
    # Compute homography and its metrics
    H_est, inliers, inlier_ratio = compute_homography_and_correctness(matches, kp_visible, kp_other, img_shape)
    
    # Add homography metrics
    metrics['homography'] = {
        'estimated': H_est is not None,
        'inlier_ratio': inlier_ratio,
        'num_inliers': sum(inliers) if inliers else 0 if H_est is not None else 0
    }
    
    # Compute repeatability
    repeatability = compute_repeatability(pred_visible, pred_other, H_est, img_shape)
    metrics['repeatability'] = repeatability
    
    return metrics, H_est

def print_metrics(metrics):
    """Print metrics in a formatted way."""
    print("\n=== Performance Metrics ===")
    
    print("\nRuntime Metrics:")
    print(f"Total processing time: {metrics['runtime']['total_time']*1000:.2f} ms")
    print(f"Forward pass time: {metrics['runtime']['forward_pass_time']*1000:.2f} ms")
    print(f"Keypoint detection time: {metrics['runtime']['keypoint_detection_time']*1000:.2f} ms")
    print(f"Descriptor computation time: {metrics['runtime']['descriptor_computation_time']*1000:.2f} ms")
    print(f"Matching time: {metrics['runtime']['matching_time']*1000:.2f} ms")
    
    print("\nKeypoint Metrics:")
    print(f"Number of visible keypoints: {metrics['keypoints']['n_keypoints_visible']}")
    print(f"Number of other keypoints: {metrics['keypoints']['n_keypoints_other']}")
    print(f"Number of matches: {metrics['keypoints']['n_matches']}")
    print(f"Matching score: {metrics['matching_score']:.3f}")
    
    if 'match_statistics' in metrics:
        print("\nMatch Statistics:")
        print(f"Min distance: {metrics['match_statistics']['min_distance']:.3f}")
        print(f"Max distance: {metrics['match_statistics']['max_distance']:.3f}")
        print(f"Mean distance: {metrics['match_statistics']['mean_distance']:.3f}")
        print(f"Median distance: {metrics['match_statistics']['median_distance']:.3f}")
    
    print("\nHomography Metrics:")
    print(f"Homography estimated: {metrics['homography']['estimated']}")
    print(f"Inlier ratio: {metrics['homography']['inlier_ratio']:.3f}")
    print(f"Number of inliers: {metrics['homography']['num_inliers']}")
    
    print("\nRepeatability Metrics:")
    print(f"Repeatability score: {metrics['repeatability']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Demo script for XPoint')
    parser.add_argument('--visible', required=True, help='Path to visible spectrum image')
    parser.add_argument('--other', required=True, help='Path to other spectrum image (e.g., thermal, NIR, etc.)')
    parser.add_argument('--config', default='configs/cipdp.yaml', help='Path to config file')
    parser.add_argument('--model-dir', default='model_weights/xpoint', help='Directory containing model weights')
    parser.add_argument('--version', default='latest', help='Model version')
    parser.add_argument('--output', default='demo_results', help='Output directory path')
    parser.add_argument('--plot', action='store_true', help='Create detailed visualization plot')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    with open(os.path.join(args.model_dir, 'params.yaml'), 'r') as f:
        config['model'] = yaml.load(f, Loader=yaml.FullLoader)['model']

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() and config['prediction']['allow_gpu'] else "cpu")
    print(f'Using device: {device}')

    # Load model
    net = getattr(models, config['model']['type'])(config['model'])
    weights = torch.load(os.path.join(args.model_dir, args.version + '.model'), map_location=device)
    weights = utils.fix_model_weigth_keys(weights)
    net.load_state_dict(weights, strict=False)
    net.to(device)
    net.eval()

    # Initialize timing dictionary
    time_dict = {}

    # Load and preprocess images
    t_start = time.time()
    target_size = (config['dataset']['width'], config['dataset']['height'])
    visible_img = load_image(args.visible, target_size)
    other_img = load_image(args.other, target_size)
    time_dict['preprocessing'] = time.time() - t_start

    # Process images
    t_start = time.time()
    out_visible, out_other, data = process_image_pair(net, visible_img, other_img, config, device)
    time_dict['forward_pass'] = time.time() - t_start

    # Extract keypoints
    t_start = time.time()
    prob_visible = out_visible['prob'].squeeze()
    prob_other = out_other['prob'].squeeze()
    
    pred_visible = torch.nonzero((prob_visible > config['prediction']['detection_threshold']).float())
    pred_other = torch.nonzero((prob_other > config['prediction']['detection_threshold']).float())
    
    kp_visible = [cv2.KeyPoint(float(x[1]), float(x[0]), 4) for x in pred_visible]
    kp_other = [cv2.KeyPoint(float(x[1]), float(x[0]), 4) for x in pred_other]
    time_dict['keypoint_detection'] = time.time() - t_start

    # Get descriptors
    t_start = time.time()
    desc_visible = out_visible['desc'].squeeze()
    desc_other = out_other['desc'].squeeze()

    if desc_visible.shape[1:] == prob_visible.shape[1:]:
        desc_visible = desc_visible[:, pred_visible[:,0], pred_visible[:,1]].transpose(0,1)
        desc_other = desc_other[:, pred_other[:,0], pred_other[:,1]].transpose(0,1)
    else:
        H, W = data['optical']['image'].shape[2:]
        desc_visible = interpolate_descriptors(pred_visible, desc_visible, H, W)
        desc_other = interpolate_descriptors(pred_other, desc_other, H, W)
    time_dict['descriptor_computation'] = time.time() - t_start

    # Match descriptors
    t_start = time.time()
    matches = get_matches(
        desc_visible.cpu().numpy(),
        desc_other.cpu().numpy(),
        config['prediction']['matching']['method'],
        config['prediction']['matching']['knn_matches'],
        **config['prediction']['matching']['method_kwargs']
    )
    time_dict['matching'] = time.time() - t_start

    # Compute metrics with homography and repeatability
    metrics, H_est = compute_metrics(matches, pred_visible, pred_other, kp_visible, kp_other, 
                                   time_dict, visible_img.shape)
    print_metrics(metrics)
    
    # Generate timestamp for unique output
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create base filename from input images
    visible_name = os.path.splitext(os.path.basename(args.visible))[0]
    other_name = os.path.splitext(os.path.basename(args.other))[0]
    base_name = f"{visible_name}_{other_name}_{timestamp}"
    
    # Save results
    if args.plot:
        # Save original visualization
        matches_path = os.path.join(args.output, f"{base_name}_matches.png")
        visualize_results(visible_img, other_img, kp_visible, kp_other, 
                         matches, matches_path)
        
        # Save alignment visualization
        alignment_path = os.path.join(args.output, f"{base_name}_alignment.png")
        visualize_alignment(visible_img, other_img, H_est, 
                          alignment_path)
        
        # Save metrics
        metrics_path = os.path.join(args.output, f"{base_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics_json = metrics.copy()
            if metrics_json['homography']['estimated']:
                metrics_json['homography']['matrix'] = H_est.tolist()
            json.dump(metrics_json, f, indent=4)
        
        print(f'\nResults saved to directory: {args.output}')
        print(f'- Matches visualization: {os.path.basename(matches_path)}')
        print(f'- Alignment visualization: {os.path.basename(alignment_path)}')
        print(f'- Metrics: {os.path.basename(metrics_path)}')
    else:
        # Save simple matches visualization
        img_visible = cv2.cvtColor((visible_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img_other = cv2.cvtColor((other_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    out_img = cv2.drawMatches(
            img_visible, kp_visible,
            img_other, kp_other,
        matches, None,
            matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
        basic_path = os.path.join(args.output, f"{base_name}_basic.png")
        cv2.imwrite(basic_path, out_img)
        print(f'\nBasic visualization saved to: {basic_path}')

if __name__ == '__main__':
    main()