import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("img_3_outside.jpg")
(B,G,R) = cv2.split(img)
(minR,maxR,x,y) = cv2.minMaxLoc(R)
(minG,maxG,x,y) = cv2.minMaxLoc(G)
(minB,maxB,x,y) = cv2.minMaxLoc(B)

lower_threshold = (minR, minG, minB)
upper_threshold = (maxR, maxG, maxB)
maskedImg = cv2.inRange(img, lower_threshold, upper_threshold)
cv2.imwrite("grassMaskTest.png",maskedImg)

calc_min = (50,50,50)
calc_max = (230,255,255)
for x in range(7):
    imgnum = x

    new_img = cv2.imread(f"images_manual_outside_1/img_{imgnum}.jpg")

    mask = cv2.inRange(new_img, calc_min, calc_max)

    cv2.imwrite(f"images_manual_outside_1_output/mask_{imgnum}_out.png",mask)

    params = cv2.SimpleBlobDetector_Params()

    #params.minThreshold = 0
    #params.maxThreshold = 255
    params.filterByColor = True
    params.blobColor = 0


    params.minDistBetweenBlobs = 50



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

    cv2.imwrite(f"images_manual_outside_1_output/img_{imgnum}_out.png",im_keypoints)