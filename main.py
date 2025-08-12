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

calc_min = (51,75,57)
calc_max = (238,250,250)

new_img = cv2.imread("image.png")

mask = cv2.inRange(new_img, calc_min, calc_max)

cv2.imwrite("grassMask.png",mask)

params = cv2.SimpleBlobDetector_Params()
 
params.filterByColor = True
params.blobColor = 255

params.filterByArea = True
params.minArea = 100
 
params.filterByCircularity = True
params.minCircularity = 0.1
 
params.filterByConvexity = True
params.minConvexity = 0.01
 
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(mask)

im_keypoints = cv2.drawKeypoints(new_img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite("im_keypoints.png",im_keypoints)