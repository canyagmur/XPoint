import os
import cv2
from src.RIFT2 import RIFT2
from src.matcher_functions import match_keypoints_nn,draw_matches,outlier_removal
import time

image_folder_path = "images/sar-optical"
img1_path = os.path.join(image_folder_path, 'pair1.jpg')
img2_path = os.path.join(image_folder_path, 'pair2.jpg')

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

start_time = time.time()
rift2_pipeline = RIFT2()
kp1, des1, kp2, des2 = rift2_pipeline(img1, img2)
end_time = time.time()
#print information
print("RIFT2 pipeline time elapsed {:.3f} seconds".format(end_time - start_time))

# Perform keypoint matching
time1 = time.time()
points1, points2, mutual_matches = match_keypoints_nn(des1, des2, kp1, kp2, lowes_ratio=0.95, mutual=False)
time2 = time.time()
print("Matching time elapsed {:.3f} seconds".format(time2 - time1))

# Outlier removal using MAGSAC
time1 = time.time()
inliers1, inliers2, matchesMask = outlier_removal(points1, points2)
time2 = time.time()
print("Outlier removal time elapsed {:.3f} seconds".format(time2 - time1))
print("Total time elapsed {:.3f} seconds".format(time2 - start_time))
# Draw matches
draw_matches(img1, img2, kp1, kp2, mutual_matches, matchesMask)


