import torch
import numpy as np
from tqdm import tqdm
import cv2
import time

import os
from .utils import box_nms, data_to_device, interpolate_descriptors,div0,data_unsqueeze
from .homographies import warp_keypoints, filter_points
from .matching import get_matches

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def desc_process_and_display_sample(net, dataset, device, config, args):
    time_dict_seconds = {"two_forward":[], "nms":[],"interpolate":[]}

    for index in args.index:
        # get the sample and move it to the right device
        synchronize()
        t_start = time.time()
        data = dataset[index]
        data = data_to_device(data, device)
        data = data_unsqueeze(data, 0)
        
        
        synchronize()
        t_1 = time.time()
        if not net.takes_pair():
            out_optical = net(data['optical'])
            out_thermal = net(data['thermal'])
        else :
            out_optical,out_thermal,out_hm = net(data) # give both

        synchronize()
        t_2 = time.time()


        # mydata = data['optical']['image'].squeeze().cpu().numpy()
        # #convert image from bgr to rgb
        # mydata = cv2.cvtColor(mydata, cv2.COLOR_BGR2RGB)
        # from matplotlib import pyplot as plt
        # plt.imshow(mydata)
        # plt.show()
        # exit()

        time_dict_seconds["two_forward"].append(t_2-t_1)

        # if len(args.index) < 4: # just to measure the HZ precisely
        #     for i in range(3):
        #         synchronize()
        #         t_1_local = time.time()
        #         if not net.takes_pair():
        #             out_optical = net(data['optical'])
        #             out_thermal = net(data['thermal'])
        #         else :
        #             out_optical,out_thermal,out_hm = net(data) # give both
        #         synchronize()
        #         t_2_local = time.time()
            
        #         avg_forward_sec.append(1/(t_2_local-t_1_local))

        # compute the nms probablity
        if config['prediction']['nms'] > 0:
            out_optical['prob'] = box_nms(out_optical['prob'] * data['optical']['valid_mask'],
                                                config['prediction']['nms'],
                                                config['prediction']['detection_threshold'],
                                                keep_top_k=config['prediction']['topk'],
                                                on_cpu=config['prediction']['cpu_nms'])
            out_thermal['prob'] = box_nms(out_thermal['prob'] * data['thermal']['valid_mask'],
                                                config['prediction']['nms'],
                                                config['prediction']['detection_threshold'],
                                                keep_top_k=config['prediction']['topk'],
                                                on_cpu=config['prediction']['cpu_nms'])

        synchronize()
        t_3 = time.time()
        #print('Loading the data took: {} s'.format(t_1 - t_start))
        #print('Two forward passes took: {} s, {} Hz'.format(t_2 - t_1,1/(t_2-t_1)))
        #print('Box nms: {} s'.format(t_3 - t_2))

        time_dict_seconds["nms"].append(t_3-t_2)


        

        
        # add homography to data if not available
        if 'homography' not in data['optical'].keys():
            data['optical']['homography'] =  torch.eye(3, dtype=torch.float32).to(device).view(data['optical']['image'].shape[0],3,3)

        if 'homography' not in data['thermal'].keys():
            data['thermal']['homography'] =  torch.eye(3, dtype=torch.float32).to(device).view(data['optical']['image'].shape[0],3,3)

        for i, (optical, thermal,
                prob_optical, prob_thermal,
                mask_optical, mask_thermal,
                H_optical, H_thermal,
                desc_optical, desc_thermal) in enumerate(zip(data['optical']['image'],
                                                                data['thermal']['image'],
                                                                out_optical['prob'],
                                                                out_thermal['prob'],
                                                                data['optical']['valid_mask'],
                                                                data['thermal']['valid_mask'],
                                                                data['optical']['homography'],
                                                                data['thermal']['homography'],
                                                                out_optical['desc'],
                                                                out_thermal['desc'],)):

            # get the keypoints
            pred_optical = torch.nonzero((prob_optical.squeeze() > config['prediction']['detection_threshold']).float())
            pred_thermal = torch.nonzero((prob_thermal.squeeze() > config['prediction']['detection_threshold']).float())
            kp_optical = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred_optical.cpu().numpy().astype(np.float32)]
            kp_thermal = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred_thermal.cpu().numpy().astype(np.float32)]


            synchronize()
            t_4 = time.time()
            # get the descriptors
            if desc_optical.shape[1:] == prob_optical.shape[1:]:
                # classic descriptors, directly take values
                desc_optical_sampled = desc_optical[:, pred_optical[:,0], pred_optical[:,1]].transpose(0,1)
                desc_thermal_sampled = desc_thermal[:, pred_thermal[:,0], pred_thermal[:,1]].transpose(0,1)
            else:
                H, W = data['optical']['image'].shape[2:]

                desc_optical_sampled = interpolate_descriptors(pred_optical, desc_optical, H, W)
                desc_thermal_sampled = interpolate_descriptors(pred_thermal, desc_thermal, H, W)

            synchronize()
            t_5 = time.time()            
            #print('bi-cubic + l2 normalization took : {} s'.format(t_5 - t_4))
            time_dict_seconds["interpolate"].append(t_5 - t_4)
            
            if args.plot:
                # match the keypoints
                matches = get_matches(desc_optical_sampled.cpu().numpy(),
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
                #cv2.imshow('matches', out_image)

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
                                            confidence=0.999,
                                            maxIters=10000,
                                        )
                    else:
                        print("Using RANSAC")
                        H_est, mask = cv2.findHomography(optical_pts, thermal_pts, cv2.RANSAC, ransacReprojThreshold=config['prediction']['reprojection_threshold'])
                    matchesMask = mask.ravel().tolist()
                
                if H_est is  not None:
                    warped_image = cv2.warpPerspective(im_optical, H_est, im_optical.shape[:2][::-1], borderMode=cv2.BORDER_CONSTANT)
                #cv2.imshow('warped optical with estimated homography', warped_image)


                # correct matches mask
                H_gt = np.matmul(H_thermal.cpu().numpy(), np.linalg.inv(H_optical.cpu().numpy()))
                warped_optical = warp_keypoints(optical_pts.squeeze(1)[:,::-1], H_gt)[:,::-1]
                diff = thermal_pts.squeeze(1) - warped_optical
                diff = np.linalg.norm(diff, axis=1)
                matchesMask = (diff < config['prediction']['reprojection_threshold']).tolist() # 4 is reprojection threshold i guess??

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
                #cv2.imshow('refined_matches', out_image_refined)
                #print(index)
                save_dir = os.path.join(args.output_dir,'images','supp','i'+str(index))
                print(save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                image_save_path  = "{}_{}_i{}_s{}".format(os.path.join(save_dir,args.model_dir.split("/")[-1]),args.version,index,args.seed)+".png"
                
                cv2.imwrite(image_save_path,out_image_refined)
                # compare estimated and computed homography
                # print('--------------------------------------------------------')
                # print('Estimated Homography:')
                # print(H_est)
                # print('Ground Truth Homography:')
                # print(np.matmul(H_thermal.cpu().numpy(), np.linalg.inv(H_optical.cpu().numpy())))
                # print('--------------------------------------------------------')

            #cv2.waitKey(0)
    return time_dict_seconds
def repeatability_process_and_display_sample(net, dataset, device, config, args):
        # get the sample and move it to the right device
        t_start = time.time()
        data = dataset[args.index[0]]

        data = data_to_device(data, device)
        data = data_unsqueeze(data, 0)

        # predict 
        if dataset.returns_pair():
            batch_size = data['optical']['image'].shape[0]
            if not net.takes_pair():
                out_optical = net(data['optical'])
                out_thermal = net(data['thermal'])
            else :
                out_optical,out_thermal,out_hm = net(data) # give both

            # compute the nms probablity
            if config['prediction']['nms'] > 0:
                out_optical['prob'] = box_nms(out_optical['prob'],
                                                    4,#config['prediction']['nms'],
                                                    config['prediction']['detection_threshold'],
                                                    keep_top_k=config['prediction']['topk'],
                                                    on_cpu=config['prediction']['cpu_nms'])
                out_thermal['prob'] = box_nms(out_thermal['prob'],
                                                    4,#config['prediction']['nms'],
                                                    config['prediction']['detection_threshold'],
                                                    keep_top_k=config['prediction']['topk'],
                                                    on_cpu=config['prediction']['cpu_nms'])
        else:
            batch_size = data['image'].shape[0]
            out = net(data)

            # compute the nms probablity
            if config['prediction']['nms'] > 0:
                out['prob'] = box_nms(out['prob'],
                                            config['prediction']['nms'],
                                            config['prediction']['detection_threshold'],
                                            keep_top_k=config['prediction']['topk'],
                                            on_cpu=config['prediction']['cpu_nms'])


        # display a sample
        if args.plot:
            if dataset.returns_pair():
                for i, (optical, thermal,
                        prob_optical, prob_thermal,
                        mask_optical, mask_thermal) in enumerate(zip(data['optical']['image'],
                                                                     data['thermal']['image'],
                                                                     out_optical['prob'],
                                                                     out_thermal['prob'],
                                                                     data['optical']['valid_mask'],
                                                                     data['thermal']['valid_mask'],)):
                    optical = optical.squeeze().cpu()
                    thermal = thermal.squeeze().cpu()
                    prob_optical = prob_optical.squeeze().cpu()
                    prob_thermal = prob_thermal.squeeze().cpu()
                    mask_optical = mask_optical.squeeze().cpu()
                    mask_thermal = mask_thermal.squeeze().cpu()

                    if args.mask:
                        optical *= mask_optical
                        thermal *= mask_thermal

                    # convert the predictions to keypoints
                    pred_optical = torch.nonzero((prob_optical > config['prediction']['detection_threshold']).float() * mask_optical)
                    kp_optical = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred_optical.numpy().astype(np.float32)]
                    pred_thermal = torch.nonzero((prob_thermal > config['prediction']['detection_threshold']).float() * mask_thermal)
                    kp_thermal = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred_thermal.numpy().astype(np.float32)]

                    # draw predictions and ground truth on image
                    out_optical = cv2.cvtColor((np.clip(optical.numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
                    out_thermal = cv2.cvtColor((np.clip(thermal.numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)

                    out_optical = cv2.drawKeypoints(out_optical,
                                                    kp_optical,
                                                    outImage=np.array([]),
                                                    color=(0, 255, 0),
                                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    out_thermal = cv2.drawKeypoints(out_thermal,
                                                    kp_thermal,
                                                    outImage=np.array([]),
                                                    color=(0, 255, 0),
                                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    if 'keypoints' in data['optical'].keys() and 'keypoints' in data['thermal'].keys():
                        if data['optical']['keypoints'] is not None and data['thermal']['keypoints'] is not None:
                            kp = data['optical']['keypoints'][i].squeeze().cpu()

                            # convert the ground truth keypoints
                            if kp.shape == optical.shape:
                                kp = torch.nonzero(kp)

                            keypoints = [cv2.KeyPoint(c[1], c[0], args.radius + 2) for c in kp.numpy().astype(np.float32)]
                            out_optical = cv2.drawKeypoints(out_optical,
                                                            keypoints,
                                                            outImage=np.array([]),
                                                            color=(0, 0, 255),
                                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                            kp = data['thermal']['keypoints'][i].squeeze().cpu()

                            # convert the ground truth keypoints
                            if kp.shape == thermal.shape:
                                kp = torch.nonzero(kp)

                            keypoints = [cv2.KeyPoint(c[1], c[0], args.radius + 2) for c in kp.numpy().astype(np.float32)]
                            out_thermal = cv2.drawKeypoints(out_thermal,
                                                            keypoints,
                                                            outImage=np.array([]),
                                                            color=(0, 0, 255),
                                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    # plot the raw image
                    # cv2.imshow(str(i) + ' image optical', out_optical)
                    # cv2.imshow(str(i) + ' prob optical', (prob_optical).numpy() * 0.9 / config['prediction']['detection_threshold'])
                    # cv2.imshow(str(i) + ' prob masked optical', (prob_optical * mask_optical).numpy() * 0.9 / config['prediction']['detection_threshold'])
                    # cv2.imshow(str(i) + ' image thermal', out_thermal)
                    # cv2.imshow(str(i) + ' prob thermal', (prob_thermal).numpy() * 0.9 / config['prediction']['detection_threshold'])
                    # cv2.imshow(str(i) + ' prob masked thermal', (prob_thermal * mask_thermal).numpy() * 0.9 / config['prediction']['detection_threshold'])

            else:
                for i, (image, prob, mask) in enumerate(zip(data['image'], out['prob'], data['valid_mask'])):
                    image = image.squeeze().cpu()
                    prob = prob.squeeze().cpu()
                    mask = mask.squeeze().cpu()

                    if args.mask:
                        image *= mask

                    # convert the predictions to keypoints
                    pred = torch.nonzero((prob > config['prediction']['detection_threshold']).float() * mask)
                    predictions = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred.numpy().astype(np.float32)]

                    # draw predictions and ground truth on image
                    out_image = cv2.cvtColor((np.clip(image.numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)

                    out_image = cv2.drawKeypoints(out_image,
                                                  predictions,
                                                  outImage=np.array([]),
                                                  color=(0, 255, 0),
                                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    if 'keypoints' in data.keys():
                        if data['keypoints'] is not None:
                            kp = data['keypoints'][i].squeeze().cpu()

                            # convert the ground truth keypoints
                            if kp.shape == image.shape:
                                kp = torch.nonzero(kp)

                            keypoints = [cv2.KeyPoint(c[1], c[0], args.radius + 2) for c in kp.numpy().astype(np.float32)]
                            out_image = cv2.drawKeypoints(out_image,
                                                          keypoints,
                                                          outImage=np.array([]),
                                                          color=(0, 0, 255),
                                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    print(str(i) + ' is_optical: ' + str(data['is_optical'][i,0].cpu().numpy()))

                    # plot the raw image
                    # cv2.imshow(str(i) + ' image', out_image)
                    # cv2.imshow(str(i) + ' prob', (prob).numpy() * 0.9 / config['prediction']['detection_threshold'])
                    # cv2.imshow(str(i) + ' prob masked', (prob * mask).numpy() * 0.9 / config['prediction']['detection_threshold'])
            #cv2.waitKey(0)


def compute_repeatability_for_sample(out_optical, out_thermal, data, H_optical, H_thermal, detection_threshold, distance_thresh):

    n_kp_optical_member = []
    n_kp_thermal_member = []
    repeatability_member_dict = {} #repeatability for each distance threshold
    

    for (prob_o, prob_t, mask_o, mask_t, h_o, h_t) in zip(out_optical['prob'].split(1),
                                                          out_thermal['prob'].split(1),
                                                          data['optical']['valid_mask'].split(1),
                                                          data['thermal']['valid_mask'].split(1),
                                                          H_optical.split(1),
                                                          H_thermal.split(1)):

        # Extract keypoints
        kp_optical = torch.nonzero((prob_o.squeeze() > detection_threshold).float() * mask_o.squeeze())
        kp_thermal = torch.nonzero((prob_t.squeeze() > detection_threshold).float() * mask_t.squeeze())

        n_kp_optical_member.append(kp_optical.shape[0])
        n_kp_thermal_member.append(kp_thermal.shape[0])

        # Convert to numpy
        kp_thermal = kp_thermal.cpu().numpy()
        kp_optical = kp_optical.cpu().numpy()
        image_shape = prob_o.squeeze().cpu().numpy().shape

        # Warp optical images to the thermal frame and filter points
        warped_optical = warp_keypoints(kp_optical, h_o.squeeze().inverse().cpu().numpy())
        warped_optical = warp_keypoints(warped_optical, h_t.squeeze().cpu().numpy())
        warped_optical = filter_points(warped_optical, image_shape)

        # Warp thermal images to the optical frame and filter points
        warped_thermal = warp_keypoints(kp_thermal, h_t.squeeze().inverse().cpu().numpy())
        warped_thermal = warp_keypoints(warped_thermal, h_o.squeeze().cpu().numpy())
        warped_thermal = filter_points(warped_thermal, image_shape)

        # Compute repeatability
        N_thermal = warped_thermal.shape[0]
        N_optical = warped_optical.shape[0]

        warped_thermal = np.expand_dims(warped_thermal, 1)
        warped_optical = np.expand_dims(warped_optical, 1)
        kp_optical = np.expand_dims(kp_optical, 0)
        kp_thermal = np.expand_dims(kp_thermal, 0)

        dist1 = np.linalg.norm(warped_thermal - kp_optical, ord=None, axis=2)
        dist2 = np.linalg.norm(warped_optical - kp_thermal, ord=None, axis=2)


        if type(distance_thresh) is list:
            for th in distance_thresh:
                repeatability_member = []
                count1 = np.sum(np.min(dist1, axis=1) <= th) if kp_optical.shape[1] != 0 else 0
                count2 = np.sum(np.min(dist2, axis=1) <= th) if kp_thermal.shape[1] != 0 else 0

                if N_thermal + N_optical > 0:
                    repeatability_member.append((count1 + count2) / (N_thermal + N_optical))
                repeatability_member_dict[th] = repeatability_member
        else:
            repeatability_member = []
            count1 = np.sum(np.min(dist1, axis=1) <= distance_thresh) if kp_optical.shape[1] != 0 else 0
            count2 = np.sum(np.min(dist2, axis=1) <= distance_thresh) if kp_thermal.shape[1] != 0 else 0

            if N_thermal + N_optical > 0:
                repeatability_member.append((count1 + count2) / (N_thermal + N_optical))
            repeatability_member_dict[distance_thresh] = repeatability_member





    return repeatability_member_dict,n_kp_optical_member, n_kp_thermal_member

def compute_mAP(precision, recall):
    """
    Compute average precision.
    """
    return np.sum(precision[1:] * (recall[1:] - recall[:-1]))


def compute_desc_dict(descriptor_metrics_dict):
    descriptor_metrics_results_dict = {}
    for th_keypoints in descriptor_metrics_dict.keys():
        # convert to numpy arrays
        tp_optical = np.array(descriptor_metrics_dict[th_keypoints]['tp_optical'])
        distance_optical = np.array(descriptor_metrics_dict[th_keypoints]['distance_optical'])
        m_score_optical = np.array(descriptor_metrics_dict[th_keypoints]['m_score_optical'])

        tp_thermal = np.array(descriptor_metrics_dict[th_keypoints]['tp_thermal'])
        distance_thermal = np.array(descriptor_metrics_dict[th_keypoints]['distance_thermal'])
        m_score_thermal = np.array(descriptor_metrics_dict[th_keypoints]['m_score_thermal'])

        n_gt_optical = descriptor_metrics_dict[th_keypoints]['n_gt_optical']
        n_gt_thermal = descriptor_metrics_dict[th_keypoints]['n_gt_thermal']

        # sort in ascending order of distance
        sort_idx_optical = np.argsort(distance_optical)
        tp_optical = tp_optical[sort_idx_optical]
        fp_optical = np.logical_not(tp_optical)
        distance_optical = distance_optical[sort_idx_optical]

        sort_idx_thermal = np.argsort(distance_thermal)
        tp_thermal = tp_thermal[sort_idx_thermal]
        fp_thermal = np.logical_not(tp_thermal)
        distance_thermal = distance_thermal[sort_idx_thermal]

        # compute the precision and recall
        tp_optical_cum = np.cumsum(tp_optical)
        tp_thermal_cum = np.cumsum(tp_thermal)
        fp_optical_cum = np.cumsum(fp_optical)
        fp_thermal_cum = np.cumsum(fp_thermal)

        recall_optical = div0(tp_optical_cum, n_gt_optical)
        recall_thermal = div0(tp_thermal_cum, n_gt_thermal)

        precision_optical = div0(tp_optical_cum, tp_optical_cum + fp_optical_cum)
        precision_thermal = div0(tp_thermal_cum, tp_thermal_cum + fp_thermal_cum)

        recall_optical = np.concatenate([[0], recall_optical, [1]])
        precision_optical = np.concatenate([[0], precision_optical, [0]])
        precision_optical = np.maximum.accumulate(precision_optical[::-1])[::-1]

        recall_thermal = np.concatenate([[0], recall_thermal, [1]])
        precision_thermal = np.concatenate([[0], precision_thermal, [0]])
        precision_thermal = np.maximum.accumulate(precision_thermal[::-1])[::-1]

        # compute nearest neighbor mean average precision
        nn_map_optical = compute_mAP(precision_optical, recall_optical)
        nn_map_thermal = compute_mAP(precision_thermal, recall_thermal)
        nn_map = (nn_map_optical + nn_map_thermal) * 0.5

        # compute the matching score
        m_score = (m_score_optical.mean() + m_score_thermal.mean()) * 0.5



        # create out dictionary
        out = {
            'tp_optical': tp_optical,
            'tp_thermal': tp_thermal,
            'fp_optical': fp_optical,
            'fp_thermal': fp_thermal,
            'distance_optical': distance_optical,
            'distance_thermal': distance_thermal,
            'recall_optical': recall_optical,
            'recall_thermal': recall_thermal,
            'precision_optical': precision_optical,
            'precision_thermal': precision_thermal,
            'nn_map_optical': nn_map_optical,
            'nn_map_thermal': nn_map_thermal,
            'nn_map': nn_map,
            'm_score_optical': m_score_optical,
            'm_score_thermal': m_score_thermal,
            'm_score': m_score,
            #'pts_dist': pts_dist_orig,
            # 'average_h_error': average_h_error,
            #'h_correctness': h_correctness,
            #'matching_kp_numbers' : matching_kp_numbers
            }
        descriptor_metrics_results_dict[th_keypoints] = out

    return descriptor_metrics_results_dict    


def compute_homography_dict(overall_pts_dist_dict,threshold_warp):
    homography_results_dict = { th : {th_warp:0 for th_warp in threshold_warp} for th in overall_pts_dist_dict.keys() }
    for th_ransac in overall_pts_dist_dict.keys():

        pts_dist_orig = overall_pts_dist_dict[th_ransac]
        pts_dist = np.array(overall_pts_dist_dict[th_ransac])


        # compute homography estimation accuracy
        average_h_error = pts_dist.mean()
        
        for th_warp in threshold_warp:
            homography_results_dict[th_ransac][th_warp] = (pts_dist < th_warp).sum() / len(pts_dist)



        # create out dictionary
        out = {
            #'pts_dist': pts_dist_orig,
            'average_h_error': average_h_error,
            "h_correctness": {}
            }
        for key,value in homography_results_dict[th_ransac].items():
            out["h_correctness"]['epsilon_warp_th'+str(key)] = value
        homography_results_dict[th_ransac] = out

    return homography_results_dict  

def compute_descriptor_for_sample(prob_optical, prob_thermal,desc_optical,desc_thermal, data, config,keypoint_detection_threshold, threshold_keypoints):
    """
    Compute various descriptor metrics for optical and thermal images.
    """

    descriptor_dict = {th:{} for th in threshold_keypoints } if type(threshold_keypoints) is list else {threshold_keypoints:{}}

    for th in descriptor_dict.keys():
        tp_optical_member = []
        tp_thermal_member = []
        distance_optical_member = []
        distance_thermal_member = []
        m_score_optical_member = []
        m_score_thermal_member = []
        #pts_dist_member = []
        matching_kp_numbers_member = []
        n_gt_optical_member = 0
        n_gt_thermal_member = 0

        #print("Computing descriptor metrics for threshold: ",th)
        for (prob_o, prob_t, h_o, h_t, desc_o, desc_t) in zip(prob_optical, prob_thermal, 
                                                            data['optical']['homography'], data['thermal']['homography'], 
                                                            desc_optical, desc_thermal):
            # get the combined homography from optical to thermal
            gt_homography = torch.mm(h_t,h_o.inverse())

            # compute keypoints
            pred_optical = torch.nonzero((prob_o.squeeze() > keypoint_detection_threshold).float())
            pred_thermal = torch.nonzero((prob_t.squeeze() > keypoint_detection_threshold).float())

            # get the descriptors
            H_o, W_o = data['optical']['image'].shape[2:]
            H_t, W_t = data['thermal']['image'].shape[2:]
            desc_o= interpolate_descriptors(pred_optical, desc_o, H_o, W_o)
            desc_t = interpolate_descriptors(pred_thermal, desc_t, H_t, W_t)

            # match the keypoints
            if desc_o.shape[0] > 0 and desc_t.shape[0] > 0:
                matches_thermal = get_matches(desc_t.cpu().numpy(),
                                                desc_o.cpu().numpy(),
                                                config['prediction']['matching']['method'],
                                                config['prediction']['matching']['knn_matches'],
                                                **config['prediction']['matching']['method_kwargs'])
                                                # 'bfmatcher',
                                                # False,
                                                # crossCheck = True)
                matches_optical = get_matches(desc_o.cpu().numpy(),
                                                desc_t.cpu().numpy(),
                                                config['prediction']['matching']['method'],
                                                config['prediction']['matching']['knn_matches'],
                                                **config['prediction']['matching']['method_kwargs'])
                                                # 'bfmatcher',
                                                # False,
                                                # crossCheck = True)
            else:
                matches_thermal = []
                matches_optical = []


            matches_optical = sorted(matches_optical, key = lambda x:x.distance)
            matches_thermal = sorted(matches_thermal, key = lambda x:x.distance)

            # warp the keypoints to get the ground truth position
            warped_optical = warp_keypoints(pred_optical.cpu().float().numpy(), gt_homography.cpu().numpy(), float)
            warped_thermal = warp_keypoints(pred_thermal.cpu().float().numpy(), gt_homography.inverse().cpu().numpy(), float)

            # compute the correct matches matrix
            dist = torch.from_numpy(warped_optical).to(pred_thermal.device).unsqueeze(1) - pred_thermal.unsqueeze(0)
            correct_optical = torch.norm(dist.float(), dim=-1) <= th
            dist = torch.from_numpy(warped_thermal).to(pred_thermal.device).unsqueeze(1) - pred_optical.unsqueeze(0)
            correct_thermal = torch.norm(dist.float(), dim=-1) <= th

            # number of possible matches (at least one valid kp in the thermal spectrum for each one from the optical spectrum)
            n_gt_optical_member = correct_optical.sum(1).nonzero().shape[0]
            n_gt_thermal_member = correct_thermal.sum(1).nonzero().shape[0]

            # check if the matches from the matcher are true or false positives
            num_matched_optical = 0
            for m in matches_optical:
                num_matched_optical += correct_optical[m.queryIdx, m.trainIdx].item()
                tp_optical_member.append(correct_optical[m.queryIdx, m.trainIdx].item())
                distance_optical_member.append(m.distance)

            num_matched_thermal = 0
            for m in matches_thermal:
                num_matched_thermal += correct_thermal[m.queryIdx, m.trainIdx].item()
                tp_thermal_member.append(correct_thermal[m.queryIdx, m.trainIdx].item())
                distance_thermal_member.append(m.distance)

            # compute the m-score (number of recovered keypoints over possible keypoints)
            image_shape = prob_o.squeeze().cpu().numpy().shape
            N_optical = filter_points(warped_optical, image_shape).shape[0]
            N_thermal = filter_points(warped_thermal, image_shape).shape[0]
            if N_optical > 0:
                m_score_optical_member.append(float(num_matched_optical)/N_optical)
            else:
                m_score_optical_member.append(0.0)
            if N_thermal > 0:
                m_score_thermal_member.append(float(num_matched_thermal)/N_thermal)
            else:
                m_score_thermal_member.append(0.0)
            
            matching_kp_numbers_member.append((num_matched_optical + num_matched_thermal) // 2)

            # # estimate the homography
            # if desc_o.shape[0] > 0 and desc_t.shape[0] > 0:
            #     matches = get_matches(desc_o.cpu().numpy(),
            #                             desc_t.cpu().numpy(),
            #                             config['prediction']['matching']['method'],
            #                             config['prediction']['matching']['knn_matches'],
            #                             **config['prediction']['matching']['method_kwargs'])
            # else:
            #     matches = []

            # kp_optical = [cv2.KeyPoint(c[1], c[0], 1) for c in pred_optical.cpu().numpy().astype(np.float32)]
            # kp_thermal = [cv2.KeyPoint(c[1], c[0], 1) for c in pred_thermal.cpu().numpy().astype(np.float32)]
            # optical_pts = np.float32([kp_optical[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            # thermal_pts = np.float32([kp_thermal[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

            # if optical_pts.shape[0] < 4 or thermal_pts.shape[0] < 4:
            #     H_est = None
            #     matchesMask = []
            # else:
            #     if tuple(map(int, cv2.__version__.split('.')[:2])) >= (4, 5):
            #         #LONG LIVE MAGSAC!
            #         #print("Using MAGSAC!")
            #         H_est, mask = cv2.findHomography(
            #                             optical_pts,
            #                             thermal_pts,
            #                             method=cv2.USAC_MAGSAC,
            #                             ransacReprojThreshold=config['prediction']['reprojection_threshold'],
            #                             confidence=0.9999,
            #                             maxIters=10000,
            #                         )
            #     else:
            #         H_est, mask = cv2.findHomography(optical_pts, thermal_pts, cv2.RANSAC, ransacReprojThreshold=config['prediction']['reprojection_threshold'])

            # #print("hm from classical : ",H_est)
            # #print("------")
            # # compute the homography correctness
            # #H_est = pred_hm
            
            # if H_est is not None:
            #     pts = np.array([[0, 0], [H_o, 0], [0, W_o], [H_o, H_o]])
            #     pts_warped_gt = warp_keypoints(pts, gt_homography.cpu().numpy(), float)
            #     pts_warped_est = warp_keypoints(pts, H_est, float)
            #     pts_dist_member.append(np.linalg.norm(pts_warped_est - pts_warped_gt, axis=1).sum()/4)
            # else:
            #     pts_dist_member.append(999.0)



        descriptor_dict[th]["tp_optical"] = tp_optical_member
        descriptor_dict[th]["tp_thermal"] = tp_thermal_member
        descriptor_dict[th]["distance_optical"] = distance_optical_member
        descriptor_dict[th]["distance_thermal"] = distance_thermal_member
        descriptor_dict[th]["m_score_optical"] = m_score_optical_member
        descriptor_dict[th]["m_score_thermal"] = m_score_thermal_member
        #descriptor_dict[th]["pts_dist"] = pts_dist_member
        descriptor_dict[th]["matching_kp_numbers"] = matching_kp_numbers_member
        descriptor_dict[th]["n_gt_optical"] = n_gt_optical_member
        descriptor_dict[th]["n_gt_thermal"] = n_gt_thermal_member

    return descriptor_dict



def compute_pts_dist_for_sample(prob_optical, prob_thermal,desc_optical,desc_thermal, data, config,keypoint_detection_threshold, ransac_reporjection_thresholds):
    """
    Compute various descriptor metrics for optical and thermal images.
    """

    pts_dist_member_dicts = {th:{} for th in ransac_reporjection_thresholds } if type(ransac_reporjection_thresholds) is list else {ransac_reporjection_thresholds:{}}

    for ransac_reporjection_threshold in ransac_reporjection_thresholds:
        pts_dist_member = []
        for (prob_o, prob_t, h_o, h_t, desc_o, desc_t) in zip(prob_optical, prob_thermal, 
                                                            data['optical']['homography'], data['thermal']['homography'], 
                                                            desc_optical, desc_thermal):
            # get the combined homography from optical to thermal
            gt_homography = torch.mm(h_t,h_o.inverse())

            # compute keypoints
            pred_optical = torch.nonzero((prob_o.squeeze() > keypoint_detection_threshold).float())
            pred_thermal = torch.nonzero((prob_t.squeeze() > keypoint_detection_threshold).float())

            # get the descriptors
            H_o, W_o = data['optical']['image'].shape[2:]
            H_t, W_t = data['thermal']['image'].shape[2:]
            desc_o= interpolate_descriptors(pred_optical, desc_o, H_o, W_o)
            desc_t = interpolate_descriptors(pred_thermal, desc_t, H_t, W_t)


            # estimate the homography
            if desc_o.shape[0] > 0 and desc_t.shape[0] > 0:
                matches = get_matches(desc_o.cpu().numpy(),
                                        desc_t.cpu().numpy(),
                                        config['prediction']['matching']['method'],
                                        config['prediction']['matching']['knn_matches'],
                                        **config['prediction']['matching']['method_kwargs'])
            else:
                matches = []

            kp_optical = [cv2.KeyPoint(c[1], c[0], 1) for c in pred_optical.cpu().numpy().astype(np.float32)]
            kp_thermal = [cv2.KeyPoint(c[1], c[0], 1) for c in pred_thermal.cpu().numpy().astype(np.float32)]
            optical_pts = np.float32([kp_optical[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            thermal_pts = np.float32([kp_thermal[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

            if optical_pts.shape[0] < 4 or thermal_pts.shape[0] < 4:
                H_est = None
                matchesMask = []
            else:
                if tuple(map(int, cv2.__version__.split('.')[:2])) >= (4, 5):
                    #LONG LIVE MAGSAC!
                    #print("Using MAGSAC!")
                    H_est, mask = cv2.findHomography(
                                        optical_pts,
                                        thermal_pts,
                                        method=cv2.USAC_MAGSAC,
                                        ransacReprojThreshold=ransac_reporjection_threshold,
                                        confidence=0.9999,
                                        maxIters=10000,
                                    )
                else:
                    H_est, mask = cv2.findHomography(optical_pts, thermal_pts, cv2.RANSAC, ransacReprojThreshold=ransac_reporjection_threshold)

            #print("hm from classical : ",H_est)
            #print("------")
            # compute the homography correctness
            #H_est = pred_hm
            
            if H_est is not None:
                pts = np.array([[0, 0], [H_o, 0], [0, W_o], [H_o, H_o]])
                pts_warped_gt = warp_keypoints(pts, gt_homography.cpu().numpy(), float)
                pts_warped_est = warp_keypoints(pts, H_est, float)
                pts_dist_member.append(np.linalg.norm(pts_warped_est - pts_warped_gt, axis=1).sum()/4)
            else:
                pts_dist_member.append(999.0)
        pts_dist_member_dicts[ransac_reporjection_threshold] = pts_dist_member

    return pts_dist_member_dicts



def compute_metrics(net, dataloader, device, config,keypoint_detection_threshold=0.015  ,thresh_repeatability=3, thresh_keypoints=2, thresh_warp=2,ransac_reproj_thresholds=3):
    """
    Compute both repeatability and descriptor metrics.
    """
    # Initialization for Repeatability Metrics
    repeatability = {th:[] for th in thresh_repeatability} if type(thresh_repeatability) is list else {thresh_repeatability:[]}
    n_kp_optical, n_kp_thermal = [], []

    descriptor_metrics_dict = {th:{} for th in thresh_keypoints} if type(thresh_keypoints) is list else {thresh_keypoints:{}}

    overall_pts_dist_dict = {th:[] for th in ransac_reproj_thresholds} if type(ransac_reproj_thresholds) is list else {ransac_reproj_thresholds:[]}


    for data in tqdm(dataloader):

        # # extract homographies
        # if 'homography' in data['optical'].keys():
        #     H_optical = data['optical']['homography']
        # else:
        #     H_optical = torch.eye(3,3).repeat(data['optical']['image'].shape[0],1,1)

        # if 'homography' in data['thermal'].keys():
        #     H_thermal = data['thermal']['homography']
        # else:
        #     H_thermal = torch.eye(3,3).repeat(data['thermal']['image'].shape[0],1,1)

        # add identity homography to data if not present
        if 'homography' not in data['optical'].keys():
            data['optical']['homography']  =  torch.eye(3, dtype=torch.float32).to(device).repeat(data['optical']['image'].shape[0],1,1)
            
        if 'homography' not in data['thermal'].keys():
            data['thermal']['homography'] =  torch.eye(3, dtype=torch.float32).to(device).repeat(data['optical']['image'].shape[0],1,1)
        H_optical = data['optical']['homography']
        H_thermal = data['thermal']['homography']


        data = data_to_device(data, device)  # Transfer data to the device
        # Predictions for both optical and thermal data
        if not net.takes_pair():
            out_optical = net(data['optical'])
            out_thermal = net(data['thermal'])
        else:
            out_optical, out_thermal, out_hm = net(data)



        ##these are for descriptor metrics     START
        # mask the output
        prob_optical = out_optical['prob'] * data['optical']['valid_mask']
        prob_thermal = out_thermal['prob'] * data['thermal']['valid_mask']
        if config["prediction"]['nms'] > 0:
            prob_thermal = box_nms(prob_thermal,
                                   config['prediction']['nms'],
                                   keypoint_detection_threshold,
                                   keep_top_k = config['prediction']['topk'],
                                   on_cpu=config['prediction']['cpu_nms'])
            prob_optical = box_nms(prob_optical,
                                   config['prediction']['nms'],
                                   keypoint_detection_threshold,
                                   keep_top_k = config['prediction']['topk'],
                                   on_cpu=config['prediction']['cpu_nms'])

        ##these are for descriptor metrics  END   
            

        # calculate nms prob if requested
        if config['prediction']['nms'] > 0:
            out_optical['prob'] = box_nms(out_optical['prob'],
                                          config['prediction']['nms'], #config['prediction']['nms'],
                                          keypoint_detection_threshold,
                                          keep_top_k=config['prediction']['topk'],
                                          on_cpu=config['prediction']['cpu_nms'])

            out_thermal['prob'] = box_nms(out_thermal['prob'],
                                          config['prediction']['nms'], #config['prediction']['nms'],
                                          keypoint_detection_threshold,
                                          keep_top_k=config['prediction']['topk'],
                                          on_cpu=config['prediction']['cpu_nms'])



        #LETS GO FOR REPEATABILITY
        repeatability_member_dict, n_kp_optical_member, n_kp_thermal_member = compute_repeatability_for_sample(out_optical, out_thermal, data, 
                                                                                                               H_optical, H_thermal, 
                                                                                                               keypoint_detection_threshold,
                                                                                                                 thresh_repeatability)
        
        for key,value in repeatability_member_dict.items():
            repeatability[key].extend(value)
            #print("Repeatability is updated : ",repeatability[key])

        n_kp_optical += n_kp_optical_member
        n_kp_thermal += n_kp_thermal_member

        #LETS GO FOR DESCRIPTOR METRICS
        descriptor_dict = compute_descriptor_for_sample(prob_optical, prob_thermal, out_optical['desc'], out_thermal['desc'], 
                                                        data, config,keypoint_detection_threshold, thresh_keypoints)

        pts_dist_dict = compute_pts_dist_for_sample(prob_optical, prob_thermal, out_optical['desc'], out_thermal['desc'],data, 
                                                    config,keypoint_detection_threshold, ransac_reporjection_thresholds=ransac_reproj_thresholds)
    
        for key, value in descriptor_dict.items():
            for key2, value2 in value.items():
                if key2.startswith("n_gt"):
                    descriptor_metrics_dict[key][key2] = descriptor_metrics_dict[key].get(key2, 0) + value2
                else:
                    descriptor_metrics_dict[key][key2] = descriptor_metrics_dict[key].get(key2, []) + value2

        for key, value in pts_dist_dict.items():
            overall_pts_dist_dict[key] += value #append the list of pts_dist for each threshold




    out = {"repeatability": {}, "descriptor": {}, "homography": {}}
    #descriptor dict
    out_desc = compute_desc_dict(descriptor_metrics_dict)
    out_homography = compute_homography_dict(overall_pts_dist_dict,thresh_warp)
    
    #repeatability dict
    out_repeatability = {}
    repeatability_dict_mean = {key: np.mean(value) for key,value in repeatability.items()}
    out_repeatability['repeatability_mean'] = repeatability_dict_mean #np.mean(repeatability)
    #out_repeatability['repeatability'] = repeatability
    out_repeatability['n_kp_optical'] = np.mean(n_kp_optical)
    out_repeatability['n_kp_thermal'] = np.mean(n_kp_thermal)
    out_repeatability['n_kp_avg'] = (out_repeatability['n_kp_optical'] +out_repeatability['n_kp_thermal'] ) /2.0

    out['descriptor'] = out_desc
    out['repeatability'] = out_repeatability
    out['homography'] = out_homography
    
    return out