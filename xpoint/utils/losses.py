import numpy as np
import torch
from torch.nn import Module

from .utils import space_to_depth, dict_update, tensors_to_dtype
from .homographies import warp_points_pytorch

from typing import Tuple
from typing import Optional
from typing import Dict





class FocalLoss(torch.nn.Module):
    """
    Implements Focal Loss as described in https://arxiv.org/pdf/1708.02002.pdf
    This is useful for addressing class imbalance by down-weighting well-classified examples.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'none',debug=False):
        """
        Initializes the FocalLoss module.

        Args:
            alpha (float): Weighting factor for the rare class.
            gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.debug = debug

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the focal loss between `inputs` and `targets`.

        Args:
            inputs (torch.Tensor): Predicted logits with shape (batch_size, C, H, W).
            targets (torch.Tensor): Ground truth class indices with shape (batch_size, H, W).

        Returns:
            torch.Tensor: Computed focal loss.
        """
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', weight=None)
        pt = torch.exp(-ce_loss)  # pt is the probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss


        #start debugging
        if self.debug:
            #find the indices of the values larger than 0 in inputs
            # Find the indices in the C dimension where values are larger than 0
            # This will return a boolean mask for where the values are greater than 0
            index_y, index_x = 10,10
            batch_idx = 0
            topk = 3

            myinput = inputs[batch_idx,:,index_y,index_x]
            myinput = torch.nn.functional.softmax(myinput)
            
            top_k_values, top_k_indices = torch.topk(myinput, topk)
            zipped_predictions = dict(zip(top_k_indices.detach().cpu().numpy(), top_k_values.detach().cpu().numpy()))
            print("Model predictions: ", zipped_predictions)
            
            #if targets in B,C,H,W format instead of B,H,W format
            if targets.dim() == 4:
                mygt = targets[batch_idx,:,index_y,index_x].detach().cpu()
                # Get a mask where values are greater than 0
                mask = mygt > 0
                # Apply the mask to get the values and their corresponding indices
                filtered_values = mygt[mask]
                filtered_indices = torch.nonzero(mask, as_tuple=True)[0]  # Getting the indices where mask is True
                # Zip the indices and values into a dictionary
                zipped_predictions = dict(zip(filtered_indices.detach().cpu().numpy(), filtered_values.detach().cpu().numpy()))
                print("GT Classes : ", zipped_predictions)
                
            else:
                print("GT Classes : ", targets[batch_idx][index_y][index_x].item())

            print("CE loss ", ce_loss[batch_idx][index_y][index_x].item())
            print("Probability ", pt[batch_idx][index_y][index_x].item())
            print("Focal Loss ", focal_loss[batch_idx][index_y][index_x].item())
            print("-------------------------------------------------")

            import matplotlib.pyplot as plt
            # Function to visualize heatmaps side by side
            def visualize_heatmaps_side_by_side(tensors, titles, cmaps, figsize=(18, 5)):
                """
                Visualizes multiple 2D tensors as heatmaps side by side in one figure.

                Args:
                    tensors (List[torch.Tensor]): List of 2D tensors to visualize.
                    titles (List[str]): List of titles for each heatmap.
                    cmaps (List[str]): List of colormaps for each heatmap.
                    figsize (tuple, optional): Size of the entire figure. Defaults to (18, 5).
                """
                num_heatmaps = len(tensors)
                fig, axes = plt.subplots(1, num_heatmaps, figsize=figsize)

                # If there's only one heatmap, axes is not a list
                if num_heatmaps == 1:
                    axes = [axes]

                for ax, tensor, title, cmap in zip(axes, tensors, titles, cmaps):
                    # Ensure tensor is detached and moved to CPU
                    tensor_np = tensor.detach().cpu().numpy()
                    
                    # Plot heatmap
                    im = ax.imshow(tensor_np, cmap=cmap, interpolation='nearest')
                    
                    # Set title and labels
                    ax.set_title(title, fontsize=14)
                    ax.set_xlabel('Width', fontsize=12)
                    ax.set_ylabel('Height', fontsize=12)
                    
                    # Add colorbar
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=10)

                plt.tight_layout()
                plt.show()

            # Prepare the tensors, titles, and colormaps
            tensors = [ce_loss[0], pt[0], focal_loss[0]]
            titles = ['Cross-Entropy Loss Heatmap', 'Probability of True Class (pt) Heatmap', 'Focal Loss Heatmap']
            cmaps = ['hot', 'plasma', 'hot']

            # # Visualize the heatmaps side by side
            #visualize_heatmaps_side_by_side(tensors, titles, cmaps)
            #end debugging


        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # 'none'


class XPointLoss(Module):
    '''
    Loss to train the XPoint model with configurable detector and descriptor losses.
    '''
    default_config = {
        'detector_loss': True,
        'detector_loss_function': 'focal_loss',  # Options: 'cross_entropy', 'focal_loss'
        'detector_handle_multiple_keypoints': 'random_selection',  # Options: 'random_selection'
        'detector_dustbin_loss_weight': 1.0,
        'detector_label_normalization': False,  # Whether to normalize labels in BCE path
        'detector_focal_loss': {
            'use': True,  # Whether to use Focal Loss
            'alpha': 0.25,  # Focal Loss alpha parameter
            'gamma': 2.0,    # Focal Loss gamma parameter
        },
        'descriptor_loss': True,
        'descriptor_loss_threshold': 8.0,
        'sparse_descriptor_loss': False,
        'sparse_descriptor_loss_num_cell_divisor': 64,
        'descriptor_loss_use_mask': True,
        'positive_margin': 1.0,
        'negative_margin': 0.2,
        'lambda_d': 250,
        'lambda': 0.0001,
        'space_to_depth_ratio': 8,
        'use_encoder_similarity': False,
        'homography_regression_loss': {
            'check' : False,
            'gamma': 1.0,
        },
        
    }

    def __init__(self, config: Optional[dict] = None):
        super(XPointLoss, self).__init__()

        # Merge user config with default config
        if config:
            self.config = dict_update(self.default_config, config)
        else:
            self.config = self.default_config


        # import numpy as np
        # from scipy.stats import norm

        # # Set array size
        # size = 64
        # # Create a 1D array of positions centered at the middle
        # x = np.linspace(-1, 1, size)
        # # Generate a Gaussian distribution, centered at 0, with standard deviation 0.2 (adjust as needed)
        # sigma = 0.2
        # gaussian = norm.pdf(x, 0, sigma)
        # # Normalize the Gaussian so that the sum is 1 (optional, if you need normalized values)
        # gaussian /= np.sum(gaussian)
        # gaussian *=100
        # average = np.average(gaussian)
        # gaussian = list(gaussian)
        # gaussian.append(average)
        # self.config['detector_class_weights'] = gaussian

        self.cross_entropy_weights = [1] *64 + [self.config['detector_dustbin_loss_weight']]



        # Initialize loss criteria based on configuration
        self.criterion_encoder_similarity = torch.nn.CosineSimilarity(dim=1).cuda() if self.config['use_encoder_similarity'] else None

        # Homography regression loss
        if self.config['homography_regression_loss']['check']:
            self.criterion_hm_regressor = torch.nn.MSELoss().cuda()
        else:
            self.criterion_hm_regressor = None

        # Initialize detector loss functions
        if self.config['detector_loss']:
            loss_function = self.config['detector_loss_function']
            if loss_function == 'cross_entropy':
                self.detector_loss_fn1 = torch.nn.CrossEntropyLoss(reduction='none')
                self.detector_loss_fn2 = torch.nn.CrossEntropyLoss(weight=self._get_class_weights(), reduction='none')
            elif loss_function == 'focal_loss':
                focal_config = self.config['detector_focal_loss']
                if focal_config['use']:
                    self.detector_loss_fn1 = FocalLoss(alpha=focal_config['alpha'],
                                                     gamma=focal_config['gamma'],
                                                     reduction="none",debug=True)
                    self.detector_loss_fn2 = self.detector_loss_fn1
                else:
                    raise ValueError("Focal Loss is not enabled in 'detector_focal_loss' config.")
            elif loss_function == 'cross_entropy_focal_blended':
                focal_config = self.config['detector_focal_loss']
                cross_entropy_loss_fn = torch.nn.CrossEntropyLoss(weight=self._get_class_weights(),reduction='none')
                focal_loss_fn = FocalLoss(alpha=focal_config['alpha'],
                                                     gamma=focal_config['gamma'],
                                                     reduction="none",debug=True)
                self.detector_loss_fn = cross_entropy_loss_fn, focal_loss_fn

            else:
                raise ValueError(f"Unsupported detector_loss_function: {loss_function}")

        # Print general information about the loss
        self.print_loss_details()

    def _get_class_weights(self) -> Optional[torch.Tensor]:
        """
        Converts class weights from config to a torch.Tensor if provided.

        Returns:
            Optional[torch.Tensor]: Tensor of class weights or None.
        """
        class_weights = torch.tensor(self.cross_entropy_weights).float().cuda()
        return class_weights

    def forward(self, loss_input_dict: dict) -> Tuple[torch.Tensor, dict]:
        """
        Computes the total loss by aggregating detector loss, descriptor loss,
        homography regression loss, and encoder similarity loss based on the provided inputs.

        Args:
            loss_input_dict (dict): A dictionary containing the following keys:
                - 'data' (dict): Ground truth data. Can contain 'optical', 'thermal', 'hfour_points'.
                - 'pred' (dict): Primary predictions. Must contain 'logits' and 'desc'.
                - 'pred2' (dict, optional): Secondary predictions. Must contain 'logits' and 'desc'.
                - 'pred_hm' (torch.Tensor, optional): Predicted homography if homography regression loss is enabled.

        Returns:
            Tuple[torch.Tensor, dict]: A tuple containing:
                - loss (torch.Tensor): The aggregated total loss.
                - loss_components (dict): Dictionary of individual loss components for monitoring.
        """
        original_data_dict = loss_input_dict['data']
        data = original_data_dict['optical'] if "optical" in original_data_dict.keys() else original_data_dict
        data2 = original_data_dict['thermal'] if "optical" in original_data_dict.keys() else None
        pred = loss_input_dict['pred']
        pred2 = loss_input_dict['pred2'] if 'pred2' in loss_input_dict.keys() else None
        gt_hm = original_data_dict["hfour_points"] if "hfour_points" in original_data_dict.keys() else None
        pred_hm = loss_input_dict['pred_hm'] if 'pred_hm' in loss_input_dict.keys() else None

        # Input validation
        if ((pred2 is None and data2 is not None) or
            (pred2 is not None and data2 is None)):
            raise ValueError('Both pred2 and data2 must be provided together to compute the loss.')

        if self.config["homography_regression_loss"]["check"] and (
            (gt_hm is not None and pred_hm is None) or 
            (gt_hm is None and pred_hm is not None)
        ):
            raise ValueError('Both ground truth homography and predicted homography must be provided for homography regression loss.')

        if self.config['use_encoder_similarity'] and pred2 is None:
            raise ValueError('Encoder similarity loss requires predictions from two images (pred and pred2).')

        # Ensure predictions are in float
        pred = tensors_to_dtype(pred, torch.float)
        if pred2 is not None:
            pred2 = tensors_to_dtype(pred2, torch.float)

        # Initialize loss
        device = data['keypoints'].device
        loss_components = {}
        loss = torch.tensor(0.0, device=device)

        # Detector loss
        if self.config['detector_loss']:
            detector_loss1, det1_components = self.detector_loss(self.detector_loss_fn2,
                pred['logits'], data['keypoints'], data['valid_mask']
            )
            loss += detector_loss1
            #update keys in det1_components, add '1' to the key
            det1_components = {key + '1': value for key, value in det1_components.items()}
            loss_components.update(det1_components)

            if pred2 is not None:
                detector_loss2, det2_components = self.detector_loss(self.detector_loss_fn2,
                    pred2['logits'], data2['keypoints'], data2['valid_mask']
                )
                loss += detector_loss2
                det2_components = {key + '2': value for key, value in det2_components.items()}
                loss_components.update(det2_components)







            

        # Descriptor loss
        if self.config['descriptor_loss']:
            if pred2 is None:
                raise ValueError('The descriptor loss requires predictions from two images.')

            homography = data.get('homography', None)
            homography2 = data2.get('homography', None)

            descriptor_loss, positive_dist, negative_dist = self.descriptor_loss(
                pred['desc'],
                pred2['desc'],
                homography,
                homography2,
                data.get('valid_mask', None),
                data2.get('valid_mask', None)
            )

            loss_components['descriptor_loss'] = descriptor_loss.item()
            loss_components['positive_dist'] = positive_dist.item()
            loss_components['negative_dist'] = negative_dist.item()

            loss += self.config['lambda'] * descriptor_loss

        # Homography regression loss
        if self.config['homography_regression_loss']['check']:
            assert gt_hm is not None and pred_hm is not None, "Homography regression loss requires both gt_hm and pred_hm."
            gt_hm_processed = torch.nn.functional.normalize(gt_hm.view(-1, 8).float())
            homography_loss = self.criterion_hm_regressor(pred_hm, gt_hm_processed)
            loss += self.config["homography_regression_loss"]["gamma"] * homography_loss
            loss_components['homography_regression_loss'] = homography_loss.item()

        # Encoder similarity loss
        if self.config['use_encoder_similarity']:
            assert pred2 is not None, "Encoder similarity loss requires predictions from two images."
            opt_flatten = pred['encoder_output'].flatten(start_dim=1)
            th_flatten = pred2['encoder_output'].flatten(start_dim=1)
            loss_encoder_sim = 1 - self.criterion_encoder_similarity(opt_flatten, th_flatten).mean()
            loss += loss_encoder_sim
            loss_components['encoder_similarity_loss'] = loss_encoder_sim.item()

        return loss, loss_components

    def detector_loss(self,detector_loss_function, logits: torch.Tensor, keypoint_map: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Computes the detector loss by comparing predicted logits against ground truth keypoint maps.

        Args:
            logits (torch.Tensor): Predicted logits from the model with shape (batch_size, C, H', W'),
                                where C is the number of classes (including dustbin if applicable).
            keypoint_map (torch.Tensor): Ground truth keypoint maps with shape (batch_size, H, W).
            valid_mask (torch.Tensor, optional): Binary mask indicating valid regions with shape (batch_size, H, W).
                                                Defaults to all ones if not provided.

        Returns:
            Tuple[torch.Tensor, dict]: A tuple containing:
                - loss (torch.Tensor): Scalar loss value representing the average loss over valid bins.
                - loss_components (dict): Dictionary containing individual loss components.
        """
        # Shape and dtype assertions for robustness
        assert logits.dim() == 4, f"Logits must be a 4D tensor, got {logits.dim()}D."
        assert keypoint_map.dim() == 3, f"Keypoint map must be a 3D tensor, got {keypoint_map.dim()}D."
        # if valid_mask is not None:
        #     assert valid_mask.shape == keypoint_map.shape, "Valid mask must have the same shape as keypoint_map."
        #     assert valid_mask.dtype == torch.bool, "Valid mask must be of type torch.bool."

        # Step 1: Label Encoding
        labels = space_to_depth(keypoint_map.unsqueeze(1), self.config['space_to_depth_ratio'])  # Shape: (batch_size, 1, H', W')
        shape = list(labels.shape)
        shape[1] = 1  # For dustbin/noip channel if needed





        

        # Step 2: Valid Mask Processing
        if valid_mask is None:
            valid_mask = torch.ones_like(keypoint_map, dtype=torch.bool).unsqueeze(1).to(labels.device)
        valid_mask = space_to_depth(valid_mask, self.config['space_to_depth_ratio'])  # Shape: (batch_size, 1, H', W')
        valid_mask = torch.prod(valid_mask, dim=1)  # Shape: (batch_size, H', W')

        # Initialize loss components dictionary
        loss_components = {}

        # Step 3: Handle Multiple Keypoints per Bin
        handle_method = self.config['detector_handle_multiple_keypoints']

        # Randomly selects one keypoint per bin by adding random noise
        labels_hard_assigned = 3.0 * labels + torch.rand(labels.shape, device=labels.device)
        labels_hard_assigned = torch.cat([labels_hard_assigned, 2.0 * torch.ones(shape, device=labels.device)], dim=1)
        labels_hard_assigned = torch.argmax(labels_hard_assigned, dim=1)  # Shape: (batch_size, H', W')
        if handle_method == 'hard_assignment':
            labels_processed = labels_hard_assigned
        elif handle_method == 'soft_assignment':
            # Soft Assignment based on the number of keypoints
            keypoint_count = labels.sum(dim=1, keepdim=True)  # Shape: (batch_size, 1, H', W')
            labels_processed = labels.float() / (keypoint_count + 1e-6)  # Avoid division by zero
            labels_processed = torch.cat([labels_processed, 1 - labels_processed.sum(dim=1, keepdim=True)], dim=1)  # Add dustbin
            #start debugging
            # if keypoint_count[0,:,0,0].item() >=3:
            #     print("keypoint_count",keypoint_count[0,:,0,0])
            #     print("labels_processed",labels_processed[0,:,0,0])
            #     print("labels_processed",labels_processed[0,:,0,0])
            #     print("---------------")
            #end debugging
        else:
            raise ValueError(f"Unsupported detector_handle_multiple_keypoints method: {handle_method}")

        # Step 4: Select Loss Function
        loss_function = self.config['detector_loss_function']
        if loss_function == 'cross_entropy':
            loss_values = detector_loss_function(logits, labels_processed)  # Shape: (batch_size, H', W')
        elif loss_function == 'focal_loss':
            loss_values = detector_loss_function(logits, labels_processed)  # Shape: (batch_size, H', W')
        elif loss_function == 'cross_entropy_focal_blended':
            cross_entropy_loss_fn, focal_loss_fn = detector_loss_function
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=1)  # Shape: (B, C, H, W)
            C = probabilities.shape[1] #65 in this case
            
            # Sum probabilities of keypoint classes (0-63)
            keypoint_probs_sum = probabilities[:, : C-1, :, :].sum(dim=1)  # Shape: (B, H, W)
            # Dustbin probability (class 64)
            dustbin_probs = probabilities[:,  C-1, :, :]  # Shape: (B, H, W)
            # Create intermediate binary heatmap
            binary_heatmap = (keypoint_probs_sum > dustbin_probs).float()  # Shape: (B, H, W)
            # Calculate ratio of non-keypoints
            keypoint_ratio = (binary_heatmap == 1).float().mean()
            #print("Non-keypoint ratio:", non_keypoint_ratio.item())

            threshold = 0.0015
            #if keypoint_ratio.item() < threshold:
                # print("keypoint ratio is less than threshold:",threshold)
                # print("----")

            blend_factor = torch.clamp((threshold - keypoint_ratio) / threshold, min=0.0, max=1.0)
                

            # Compute both losses

            loss_values_ce = cross_entropy_loss_fn(logits, labels_processed)
            
            loss_values_focal = focal_loss_fn(logits, labels_processed)
            
            loss_values = (1 - blend_factor) * loss_values_ce + blend_factor * loss_values_focal

            # print("blend factor", blend_factor)
            # print("keypoint_ratio", keypoint_ratio)
            #print("loss_values_ce", loss_values_ce)
            #print("loss_values_focal", loss_values_focal)
            #print("loss_values", loss_values)
            #print("----")
            # Blend the losses

        else:
            raise ValueError(f"Unsupported detector_loss_function: {loss_function}")
        

        
        labels_hard_assigned_masked = labels_hard_assigned * valid_mask
        logits_prob = torch.nn.functional.softmax(logits, dim=1)
        logits_prob_max = torch.argmax(logits_prob, dim=1)

        # Compute correct predictions for the entire batch
        correct = torch.sum(logits_prob_max == labels_hard_assigned_masked)

        # Compute incorrect predictions for the entire batch
        incorrect = torch.sum(logits_prob_max != labels_hard_assigned_masked)
        
        # Total number of elements in the entire batch
        total_elements = labels_hard_assigned_masked.numel()

        # Calculate ratios
        correct_ratio = correct.float() / total_elements
        incorrect_ratio = incorrect.float() / total_elements

    
        # True Positive (TP): Predicted 0-63 and Ground Truth is also 0-63
        TP = torch.sum((logits_prob_max <= 63) & (labels_hard_assigned_masked <= 63))

        # False Positive (FP): Predicted 0-63, but Ground Truth is 64
        FP = torch.sum((logits_prob_max <= 63) & (labels_hard_assigned_masked == 64))

        # False Negative (FN): Predicted 64, but Ground Truth is 0-63
        FN = torch.sum((logits_prob_max == 64) & (labels_hard_assigned_masked <= 63))

        # True Negative (TN): Predicted 64, and Ground Truth is also 64
        TN = torch.sum((logits_prob_max == 64) & (labels_hard_assigned_masked == 64))

        # Calculate Precision
        precision = TP.float() / (TP + FP).float()

        # Calculate Recall
        recall = TP.float() / (TP + FN).float()

        # Calculate F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Calculate Accuracy
        accuracy = (TP + TN).float() / (TP + TN + FP + FN).float()

        # print("correct ratio", correct_ratio)
        # print("incorrect ratio", incorrect_ratio)
        # print("TP ratio", TP.float() / total_elements)
        # print("FP ratio", FP.float() / total_elements)
        # print("FN ratio", FN.float() / total_elements)
        # print("TN ratio", TN.float() / total_elements)
        # print("precision", precision)
        # print("recall", recall)
        # print("f1_score", f1_score)
        # print("accuracy", accuracy)
        #print("-------------------------------------------------")

        loss_components['correct_ratio'] = correct_ratio.item()
        loss_components['incorrect_ratio'] = incorrect_ratio.item()
        loss_components['TP_ratio'] = TP.float() / total_elements
        loss_components['FP_ratio'] = FP.float() / total_elements
        loss_components['FN_ratio'] = FN.float() / total_elements
        loss_components['TN_ratio'] = TN.float() / total_elements
        # loss_components['precision'] = precision.item()
        # loss_components['recall'] = recall.item()
        # loss_components['f1_score'] = f1_score.item()
        # loss_components['accuracy'] = accuracy.item()
        




        

        # Step 5: Apply Valid Mask
        loss_values = loss_values * valid_mask  # Shape: (batch_size, H', W')
        loss_components['detector_loss'] = loss_values.mean().item()

        # Step 6: Normalize and Aggregate Loss
        # To prevent division by zero, add a small epsilon where necessary
        denominator = valid_mask.sum(dim=[1, 2])  # Shape: (batch_size,)
        denominator = denominator.clamp(min=1.0)  # Prevent division by zero
        normalized_loss = (loss_values.sum(dim=[1, 2]) / denominator).mean()
        loss_components['detector_normalized_loss'] = normalized_loss.item()

        

        return normalized_loss, loss_components


    def descriptor_loss(self, descriptor1: torch.Tensor, descriptor2: torch.Tensor, 
                        homography1: Optional[torch.Tensor], homography2: Optional[torch.Tensor],
                        valid_mask1: Optional[torch.Tensor] = None, valid_mask2: Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the descriptor loss based on the descriptors from two images and their homographies.

        Args:
            descriptor1 (torch.Tensor): Descriptors from the first image with shape (batch_size, desc_size, Hc, Wc).
            descriptor2 (torch.Tensor): Descriptors from the second image with shape (batch_size, desc_size, Hc, Wc).
            homography1 (torch.Tensor, optional): Homography matrix for the first image.
            homography2 (torch.Tensor, optional): Homography matrix for the second image.
            valid_mask1 (torch.Tensor, optional): Validity mask for the first image.
            valid_mask2 (torch.Tensor, optional): Validity mask for the second image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - descriptor_loss (torch.Tensor): The computed descriptor loss.
                - positive_dist (torch.Tensor): Positive distance component of the loss.
                - negative_dist (torch.Tensor): Negative distance component of the loss.
        """
        # Input check
        assert descriptor1.shape == descriptor2.shape, "Descriptor shapes must match."

        if homography1 is not None and homography2 is not None:
            assert homography1.shape == homography2.shape, "Homography shapes must match."
            assert descriptor1.shape[0] == homography1.shape[0], "Batch size of descriptors and homographies must match."

        # Compute the position of the center pixel of every cell in the image
        batch_size = descriptor1.shape[0]
        desc_size = descriptor1.shape[1]
        Hc = descriptor1.shape[2]
        Wc = descriptor1.shape[3]

        if self.config['sparse_descriptor_loss']:
            # Sparse Descriptor Loss Implementation
            # Select random indices
            num_cells = int(np.floor(Hc * Wc / self.config['sparse_descriptor_loss_num_cell_divisor']))
            coord_cells = torch.stack((torch.randint(Hc, (num_cells,)), torch.randint(Wc, (num_cells,))), dim=-1)  # Shape: (num_cells, 2)

            # Extend the cell coordinates in the batch dimension
            coord_cells = coord_cells.unsqueeze(0).expand([batch_size, -1, -1]).clone().to(descriptor1.device)  # Shape: (batch_size, num_cells, 2)

            # Warp the coordinates into the common frame
            if homography1 is not None:
                warped_cells_1 = warp_points_pytorch(coord_cells.float(), homography1)  # Shape: (batch_size, num_cells, 2)
            else:
                warped_cells_1 = coord_cells.float()

            if homography2 is not None:
                warped_cells_2 = warp_points_pytorch(coord_cells.float(), homography2)  # Shape: (batch_size, num_cells, 2)
            else:
                warped_cells_2 = coord_cells.float()

            # Compute the correspondence
            # Distance threshold to determine correspondence
            dist = (coord_cells.unsqueeze(1).float() - coord_cells.unsqueeze(-2).float()).norm(dim=-1)  # Shape: (batch_size, num_cells, num_cells)
            correspondence = (dist <= np.sqrt(0.5)).float()  # Shape: (batch_size, num_cells, num_cells)

            # Create a valid mask based on which cells are visible in both images
            valid = (((warped_cells_1[:, :, 0] > -0.5).float() *
                     (warped_cells_1[:, :, 0] < Hc - 0.5).float()).unsqueeze(1) *
                    ((warped_cells_2[:, :, 1] > -0.5).float() *
                     (warped_cells_2[:, :, 1] < Wc - 0.5).float()).unsqueeze(-1))  # Shape: (batch_size, num_cells, num_cells)

            # Make sure the indexes are within the image frame
            idx_1 = warped_cells_1.round().int()  # Shape: (batch_size, num_cells, 2)
            idx_1[:, :, 0].clamp_(0, Hc - 1)
            idx_1[:, :, 1].clamp_(0, Wc - 1)
            idx_2 = warped_cells_2.round().int()  # Shape: (batch_size, num_cells, 2)
            idx_2[:, :, 0].clamp_(0, Hc - 1)
            idx_2[:, :, 1].clamp_(0, Wc - 1)

            # Extract the descriptors
            desc_1 = torch.zeros((batch_size, desc_size, num_cells)).to(descriptor1.device)
            desc_2 = torch.zeros((batch_size, desc_size, num_cells)).to(descriptor1.device)

            for i, idx in enumerate(idx_1):
                desc_1[i] = descriptor1[i, :, idx[:, 0].long(), idx[:, 1].long()]

            for i, idx in enumerate(idx_2):
                desc_2[i] = descriptor2[i, :, idx[:, 0].long(), idx[:, 1].long()]

            # Compute the dot product
            dot_product_desc = torch.matmul(desc_2.permute([0, 2, 1]), desc_1)  # Shape: (batch_size, num_cells, num_cells)

            # Compute the loss
            positive_dist = self.config['lambda_d'] * correspondence * torch.max(
                torch.zeros(1, device=descriptor1.device), 
                self.config['positive_margin'] - dot_product_desc
            )
            negative_dist = (1 - correspondence) * torch.max(
                torch.zeros(1, device=descriptor1.device), 
                dot_product_desc - self.config['negative_margin']
            )

            # Apply the valid mask
            positive_dist *= valid
            negative_dist *= valid

            loss = negative_dist + positive_dist

            normalization = valid.sum(-1).sum(-1)  # Shape: (batch_size,)

            loss = (loss.sum(-1).sum(-1) / normalization.unsqueeze(1)).mean()
            positive_dist = (positive_dist.sum(-1).sum(-1) / normalization.unsqueeze(1)).mean()
            negative_dist = (negative_dist.sum(-1).sum(-1) / normalization.unsqueeze(1)).mean()

        else:
            # Dense Descriptor Loss Implementation
            coord_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=-1)  # Shape: (Hc, Wc, 2)
            coord_cells = coord_cells * 8.0 + 4.0  # Assuming space_to_depth_ratio=8, adjust if different
            coord_cells = coord_cells.unsqueeze(0).expand([batch_size, -1, -1, -1]).clone().to(descriptor1.device)  # Shape: (batch_size, Hc, Wc, 2)

            # Warp the pixel centers
            shape = coord_cells.shape
            if homography1 is not None:
                warped_cells_1 = warp_points_pytorch(coord_cells.reshape(batch_size, -1, 2), homography1.inverse()).reshape(shape)  # Shape: (batch_size, Hc, Wc, 2)
            else:
                warped_cells_1 = coord_cells

            if homography2 is not None:
                warped_cells_2 = warp_points_pytorch(coord_cells.reshape(batch_size, -1, 2), homography2.inverse()).reshape(shape)  # Shape: (batch_size, Hc, Wc, 2)
            else:
                warped_cells_2 = coord_cells

            # Compute the pairwise distance
            dist = (warped_cells_1.unsqueeze(1).unsqueeze(1) - warped_cells_2.unsqueeze(-2).unsqueeze(-2)).norm(dim=-1)  # Shape: (batch_size, Hc, Wc, Hc, Wc)
            correspondence = (dist <= self.config['descriptor_loss_threshold']).float()  # Shape: (batch_size, Hc, Wc, Hc, Wc)

            # Compute the dot product
            dot_product_desc = torch.matmul(
                descriptor2.view(batch_size, desc_size, -1).permute([0, 2, 1]),
                descriptor1.view(batch_size, desc_size, -1)
            ).view(batch_size, Hc, Wc, Hc, Wc)  # Shape: (batch_size, Hc, Wc, Hc, Wc)

            # Compute the loss
            positive_dist = self.config['lambda_d'] * correspondence * torch.max(
                torch.zeros(1, device=descriptor1.device), 
                self.config['positive_margin'] - dot_product_desc
            )
            negative_dist = (1 - correspondence) * torch.max(
                torch.zeros(1, device=descriptor1.device), 
                dot_product_desc - self.config['negative_margin']
            )
            del dot_product_desc
            loss = positive_dist + negative_dist

            if self.config['descriptor_loss_use_mask']:
                # Get the valid mask
                if valid_mask1 is None:
                    valid_mask1 = torch.ones(batch_size, 1, Hc, Wc).to(descriptor1.device)
                else:
                    valid_mask1 = space_to_depth(valid_mask1, self.config['space_to_depth_ratio'])
                    valid_mask1 = torch.prod(valid_mask1, dim=1)

                if valid_mask2 is None:
                    valid_mask2 = torch.ones(batch_size, 1, Hc, Wc).to(descriptor1.device)
                else:
                    valid_mask2 = space_to_depth(valid_mask2, self.config['space_to_depth_ratio'])
                    valid_mask2 = torch.prod(valid_mask2, dim=1)

                valid_mask = torch.matmul(
                    valid_mask2.view(batch_size, -1, 1).float(),
                    valid_mask1.view(batch_size, 1, -1).float()
                ).view(batch_size, Hc, Wc, Hc, Wc)

                loss *= valid_mask
                positive_dist *= valid_mask
                negative_dist *= valid_mask
                normalization = valid_mask.sum(-1).sum(-1).sum(-1).sum(-1)  # Shape: (batch_size,)
            else:
                normalization = Hc * Wc * Hc * Wc  # Scalar

            loss = (loss.sum(-1).sum(-1).sum(-1).sum(-1) / normalization).mean()
            positive_dist = (positive_dist.sum(-1).sum(-1).sum(-1).sum(-1) / normalization).mean()
            negative_dist = (negative_dist.sum(-1).sum(-1).sum(-1).sum(-1) / normalization).mean()

        return loss, positive_dist, negative_dist
    
    def print_loss_details(self) -> None:
        print(f"""
        === XPointLoss Configuration ===

        Detector Loss:
        - Enabled: {self.config['detector_loss']}
        - Loss Function: {self.config['detector_loss_function']}
        - Handle Multiple Keypoints: {self.config['detector_handle_multiple_keypoints']}
        - Label Normalization: {self.config['detector_label_normalization']}
        - Class Weights: {self.cross_entropy_weights}
        - Dustbin Loss Weight: {self.config['detector_dustbin_loss_weight']}
        
        Focal Loss Configuration:
            * Use Focal Loss: {self.config['detector_focal_loss']['use']}
            * Alpha: {self.config['detector_focal_loss']['alpha']}
            * Gamma: {self.config['detector_focal_loss']['gamma']}

        Descriptor Loss:
        - Enabled: {self.config['descriptor_loss']}
        - Threshold: {self.config['descriptor_loss_threshold']}
        - Sparse Descriptor Loss: {self.config['sparse_descriptor_loss']}
        - Sparse Descriptor Loss Cell Divisor: {self.config['sparse_descriptor_loss_num_cell_divisor']}
        - Use Mask: {self.config['descriptor_loss_use_mask']}
        - Positive Margin: {self.config['positive_margin']}
        - Negative Margin: {self.config['negative_margin']}
        - Lambda_d: {self.config['lambda_d']}
        - Lambda: {self.config['lambda']}

        Space to Depth:
        - Ratio: {self.config['space_to_depth_ratio']}

        Encoder Similarity:
        - Use Encoder Similarity: {self.config['use_encoder_similarity']}

        Homography Regression Loss:
        - Enabled: {self.config['homography_regression_loss']['check']}
        - Gamma: {self.config['homography_regression_loss']['gamma']}

        ===============================

        """)
