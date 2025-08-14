import cv2
import numpy as np

img = cv2.imread("images_manual_outside_2/img_15.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
(H, S, V) = cv2.split(img)

cv2.imwrite("HTest.png", H)
cv2.imwrite("STest.png", S)
cv2.imwrite("VTest.png", V)

(minH, maxH, x,y) = cv2.minMaxLoc(H)

low = (0, 0, 0)
high = (100, 255, 255)

new_img = cv2.imread("images_manual_outside_3/img_9.jpg")
(H, S, V) = cv2.split(new_img)

cv2.imwrite("AHTest.png", H)
cv2.imwrite("ASTest.png", S)
cv2.imwrite("AVTest.png", V)
new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
new_img = cv2.inRange(new_img, low, high)
cv2.imwrite("HMaskTest.png", new_img)

for x in range(15):
    imgnum = x

    new_img = cv2.imread(f"images_manual_outside_3/img_{imgnum}.jpg")
    temp_img = new_img
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
    new_img = cv2.inRange(new_img, low, high)

    cv2.imwrite(f"images_manual_outside_3_output_hsv/mask_{imgnum}_out.png",new_img)

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

    keypoints = detector.detect(new_img)

    im_keypoints = cv2.drawKeypoints(temp_img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite(f"images_manual_outside_3_output_hsv/img_{imgnum}_out.png",im_keypoints)