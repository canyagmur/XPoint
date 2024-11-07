import cv2
import numpy as np
from matplotlib import pyplot as plt

def match_keypoints_nn(des1, des2, kp1, kp2, lowes_ratio=0.75, mutual=True):


    # BFMatcher with default params
    bf = cv2.BFMatcher()

    # Mutual Nearest Neighbor Matching
    matches1 = bf.knnMatch(des1, des2, k=2)

    if mutual:
        matches2 = bf.knnMatch(des2, des1, k=2)
        # Apply ratio test version 2 (mutual nearest neighbor check)
        good_matches1 = []
        for m, n in matches1:
            if m.distance < lowes_ratio * n.distance:
                good_matches1.append(m)

        good_matches2 = []
        for m, n in matches2:
            if m.distance < lowes_ratio * n.distance:
                good_matches2.append(m)

        # Mutual Nearest Neighbor
        mutual_matches = []
        for m in good_matches1:
            if any(m.queryIdx == n.trainIdx and m.trainIdx == n.queryIdx for n in good_matches2):
                mutual_matches.append(m)
    else:
        # Apply ratio test version 1
        mutual_matches = []
        for m, n in matches1:
            if m.distance < lowes_ratio * n.distance:
                mutual_matches.append(m)

    # Extract location of good matches
    points1 = np.float32([kp1[m.queryIdx].pt for m in mutual_matches]).reshape(-1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in mutual_matches]).reshape(-1, 2)

    return points1, points2, mutual_matches



def outlier_removal(points1,points2):
    # Outlier removal using MAGSAC
    H, mask = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, 5.0,confidence=0.9999,maxIters=10000)
    matchesMask = mask.ravel().tolist()

    #select inlier keypoints 
    inliers1 = [points1[i] for i in range(len(points1)) if matchesMask[i]]
    inliers2 = [points2[i] for i in range(len(points2)) if matchesMask[i]]

    return inliers1, inliers2, matchesMask

def draw_matches(img1, img2, kp1, kp2, mutual_matches, matchesMask):
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, mutual_matches, None, **draw_params)
    # Display the number of inliers and outliers
    num_inliers = np.sum(matchesMask)
    num_outliers = len(mutual_matches) - num_inliers
    print(f'Number of kp1: {len(kp1)}')
    print(f'Number of kp2: {len(kp2)}')
    print(f'Number of matches with N.N : {len(mutual_matches)}')
    print(f'Number of inliers after MAGSAC: {num_inliers}')
    print(f'Number of outliers after MAGSAC: {num_outliers}')
    plt.imshow(img3), plt.show()


