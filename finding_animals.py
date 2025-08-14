import numpy as np
import cv2
import matplotlib.pyplot as plt


def main(): 
  # Setting up fonts
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 6
  color = (255, 255, 255)
  thickness = 6

  # Find parameters for Mahalanobis
  reference_color, covariance_matrix = findGrassValues("animal_contour/img_5_grass.jpg")

  # Find countours of known Animals
  animal_files = {
        "animal_contour/rhino.png": "Rhino",    
        "animal_contour/lion.png": "Lion",   
        "animal_contour/hippo.png": "Hippo",            
        "animal_contour/gazelle.png": "Gazelle",            
        "animal_contour/elephant.png": "elephant",
        "animal_contour/zebra.png": "zebra",
        "animal_contour/lion_weird.png": "Lion",

        #"animal_contour/rhino.png": "Rhino",    
        "animal_contour/img_20_lion.jpg": "Lion",            
        "animal_contour/img_5_gazzel.jpg": "Gazelle",            
        "animal_contour/img_5_elephant.jpg": "elephant",
        "animal_contour/img_5_zebra.jpg": "zebra",
    }
  animal_contours = {}  # {filename: [contours]}

  badcode = 0
  for file in animal_files:
    badcode += 1
    if badcode < 8:
        contours, img, drawing = findCountoursInFile(file, reference_color, covariance_matrix, 1)
        animal_contours[file] = contours
    else:
        contours, img, drawing = findCountoursInFile(file, reference_color, covariance_matrix, 3)
        animal_contours[file] = contours


  # Find Countour of unknown Animals:
  reference_color, covariance_matrix = findGrassValues("images_auto_outside_final/img_3.jpg")

  target_fileName = [
        "images_auto_outside_final/img_6.jpg",
        "images_auto_outside_final/img_7.jpg",
        "images_auto_outside_final/img_13.jpg",
        "images_auto_outside_final/img_14.jpg",
        "images_auto_outside_final/img_15.jpg",
        "images_auto_outside_final/img_17.jpg",
        "images_auto_outside_final/img_20.jpg",
        "images_auto_outside_final/img_21.jpg",
        "images_auto_outside_final/img_22.jpg",
        "images_auto_outside_final/img_23.jpg",
        "images_auto_outside_final/img_24.jpg",
        "images_auto_outside_final/img_25.jpg",
        "images_auto_outside_final/img_26.jpg",
        "images_auto_outside_final/img_27.jpg",
        "images_auto_outside_final/img_33.jpg",
    ]

  index = 0
  for file in target_fileName:
    index += 1
    target_contour, target_img, target_drawing = findCountoursInFile(file, reference_color, covariance_matrix, 3)

    for i, contour in enumerate(target_contour):
        best_score = float(9999999)
        best_match = None
        for file in animal_files:
                score = cv2.matchShapes(target_contour[i], animal_contours[file][0], 2, 0.0)
                print(score)

                if score < best_score:
                        best_score = score
                        best_match = file
        print(best_match)
        print(best_score)


        moments = cv2.moments(target_contour[i], False)
        cx = int(moments['m10']/moments['m00'])
        cy = int(moments['m01']/moments['m00'])
        
        print(cx)
        print(cy, "\n")
        cv2.putText(target_img, animal_files[best_match], (cx,cy), font, fontScale,
        color, thickness, cv2.LINE_AA)
    
    #plt.figure(figsize=(12, 8))
    #plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    #plt.title("img")
    #plt.axis("off")
    #plt.show()

    cv2.imwrite(f"finding_animals_output/2img_{index}_out.png",target_img)



def findGrassValues(fileName):
  img_grass = cv2.imread(fileName, 1)
  grass_lab = cv2.cvtColor(img_grass, cv2.COLOR_BGR2Lab)

  #Flatten images
  grass_pixels = np.reshape(grass_lab, (-1, 3))

  # Compute reference mean and covariance
  reference_color  = np.average(grass_pixels, axis=0)
  covariance_matrix = np.cov(grass_pixels.transpose())

  return reference_color, covariance_matrix




def findCountoursInFile(fileName, reference_color, covariance_matrix, objectClass):  
  # ObjectClass 1 assumes white background
  # ObjectClass 2 plots images

  # Read and convert images to LAB
  img = cv2.imread(fileName) 
  start_img = cv2.imread(fileName) 

  img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
  image_scale_factor =  1.64977299e-7 * np.size(img)
  
  #Adding blur!
  img_lab_blur = cv2.GaussianBlur(img_lab, (35, 35), 0)

  if objectClass == 1: # If it is training images Do this 
    img_blur = cv2.GaussianBlur(img, (15, 15), 0)
    #if fileName == "animal_contour/lion.png":
    #  img_blur = cv2.GaussianBlur(img_blur, (35, 35), 0)
  
  #Flatten images
  pixels = np.reshape(img_lab_blur, (-1, 3))
  shape = pixels.shape

  # Compute Mahalanobis distance
  diff = pixels - np.repeat([reference_color], shape[0], axis=0)
  inv_cov = np.linalg.inv(covariance_matrix)
  moddotproduct = diff * (diff @ inv_cov)
  mahalanobis_dist = np.sum(moddotproduct, axis=1)

  # Reshape to image        
  mahalanobis_distance_image = np.reshape(
  mahalanobis_dist, 
  (img.shape[0],
  img.shape[1]))

  # Scale the distance image and make thresshold image

  if objectClass == 1: # Change to 1 
    lower_threshold = (235,235,235)
    upper_threshold = (255,255,255)
    mask = cv2.inRange(img_blur, lower_threshold, upper_threshold)

  else: 
    mahalanobis_distance_image = 5 * 255 * mahalanobis_distance_image / np.max(mahalanobis_distance_image)
    thresholdMask = 100
    mask = mahalanobis_distance_image < thresholdMask


  # Finding Contours and Hu 
  observations = []
  mask_uint8 = (mask * 255).astype(np.uint8)
  drawing = 0 * img

  contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  for i, contour in enumerate(contours):
        area = cv2.contourArea(contours[i], True)
        if area < 1500*image_scale_factor:
          continue

        observation = []
        observation.append(contours[i])
        observations.append(observation)

        cv2.drawContours(drawing, contours, i, (255, 0, 255), 2, 8, hierarchy, 0)
        cv2.drawContours(img, contours, i, (255, 0, 255), 2, 8, hierarchy, 0)
        


  if objectClass == 2:  ## Make cool plots
    fig, axis = plt.subplots(2, 3, figsize=(24, 8))

    axis[0,0].imshow(cv2.cvtColor(start_img, cv2.COLOR_BGR2RGB))
    axis[0,0].set_title("Original")
    axis[0,0].axis('off')

    axis[0,1].imshow(cv2.cvtColor(img_lab, cv2.COLOR_BGR2RGB))
    axis[0,1].set_title("Image in LAB Color space")
    axis[0,1].axis('off')

    axis[0,2].imshow(cv2.cvtColor(img_lab_blur, cv2.COLOR_BGR2RGB))
    axis[0,2].set_title("Lab image blured")
    axis[0,2].axis('off')

    axis[1,0].imshow(mahalanobis_distance_image, cmap='gray')
    axis[1,0].set_title("mahalanobis distance")
    axis[1,0].axis('off')

    axis[1,1].imshow(mask_uint8, cmap='gray')
    axis[1,1].set_title("Animal mask")
    axis[1,1].axis('off')

    axis[1,2].imshow(cv2.cvtColor(drawing, cv2.COLOR_BGR2RGB))
    axis[1,2].set_title("contur")
    axis[1,2].axis('off')

    plt.show()

  return [obs[0] for obs in observations], img, drawing

main()