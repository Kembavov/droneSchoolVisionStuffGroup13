import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("images_manual_outside_2/img_15.jpg")
(B,G,R) = cv2.split(img)
(minR,maxR,x,y) = cv2.minMaxLoc(R)
(minG,maxG,x,y) = cv2.minMaxLoc(G)
(minB,maxB,x,y) = cv2.minMaxLoc(B)

lower_threshold = (minR, minG, minB)
upper_threshold = (maxR, maxG, maxB)
maskedImg = cv2.inRange(img, lower_threshold, upper_threshold)
cv2.imwrite("grassMaskTest.png",maskedImg)
print(lower_threshold)
print(upper_threshold)

calc_min = (0,10,0)
calc_max = (150,224,220)

for x in range(15):
    imgnum = x

    new_img = cv2.imread(f"images_manual_outside_3/img_{imgnum}.jpg")

    mask = cv2.inRange(new_img, calc_min, calc_max)

    cv2.imwrite(f"images_manual_outside_3_output/mask_{imgnum}_out.png",mask)

    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 0


    params.minDistBetweenBlobs = 500

    params.filterByArea = True
    params.minArea = 1000
    params.maxArea = 10000000000000000000000
    
    params.filterByCircularity = False
    params.minCircularity = 0.01
    
    params.filterByConvexity = False
    params.minConvexity = 0.01
    
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(mask)

    im_keypoints = cv2.drawKeypoints(new_img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite(f"images_manual_outside_3_output/img_{imgnum}_out.png",im_keypoints)