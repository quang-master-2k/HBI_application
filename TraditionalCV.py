import cv2
import numpy as np
from skimage.filters.thresholding import threshold_isodata

class TraditionalCV:
    def __init__(self, thresholdType, numCorners, crop_w = 60, crop_h = 60, positionThatHaveTwoCorners = []):
        # This is the constructor method
        # Initialize instance variables here
        self.thresholdType = thresholdType
        self.numCorners = numCorners
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.image = None
        self.threshold = None
        self.finalCorner = None
        self.finalImageColor = None
        self.finalImageBinary = None
        self.maskGeneralized = None
        self.maskAccurate = None
        self.cnt = None
        self.cntList = []
        self.positionThatHaveTwoCorners = positionThatHaveTwoCorners


    def change_crop_size(self, width, height):
        self.crop_w = width
        self.crop_h = height

    def image_RGB_Gray_mask(self, input_image):
        imgColor = input_image
        imgGray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Applying the filter
        #imgColor = cv2.GaussianBlur(imgColor,gaussian_kernel_size,0)
        #imgGray = cv2.GaussianBlur(imgGray,gaussian_kernel_size,0)

        mask = np.zeros_like(imgColor)
        return imgColor, imgGray, mask
    
    def binary_contours(self, input_imgGray, thresholdType):
        binaryThreshold = threshold_isodata(input_imgGray)
        _, threshold = cv2.threshold(input_imgGray, binaryThreshold, 255, thresholdType)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return threshold, contours

    def find_max_value(self, arr):
        max_value = max(arr, key=lambda x: x[0])
        return max_value

    def drawContours_imgColor_mask(self, contours, imgColor, mask, eps = 0.004):
        areaList = []
        for cnt in contours:
            areaList.append([cv2.contourArea(cnt), cnt])

        cnt = self.find_max_value(areaList)[1]
        
        approx = cv2.approxPolyDP(cnt, eps*cv2.arcLength(cnt, True), True)
        
        cv2.drawContours(mask, [approx], 0, (255, 255, 255), 1) 
        cv2.drawContours(imgColor, [approx], 0, (0, 0, 255), 1)

        return imgColor, mask, approx, cnt

    
    def drawCorners(self, mask, imgColor, numCorners, qualityLevel = 0.1, minDistance = 10, blockSize = 15, pointSize = 8):
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(mask, numCorners, qualityLevel = qualityLevel, minDistance = minDistance, blockSize = blockSize)
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(imgColor, (x, y), pointSize,  (0, 255, 0), -1)
        return mask, corners
        
    def sortCorners(self, corners):
        mid_corner = int(len(corners) / 2)

        for i in range(len(corners)):
            min_idx = i
            for j in range(i+1, len(corners)):
                if corners[min_idx][0][0] > corners[j][0][0]:
                    min_idx = j
            corners[i][0][0], corners[min_idx][0][0] = corners[min_idx][0][0], corners[i][0][0]
            corners[i][0][1], corners[min_idx][0][1] = corners[min_idx][0][1], corners[i][0][1]

        for i in range(0, mid_corner):
            min_idx = i
            for j in range(i+1, mid_corner):
                if corners[min_idx][0][1] > corners[j][0][1]:
                    min_idx = j
            corners[i][0][0], corners[min_idx][0][0] = corners[min_idx][0][0], corners[i][0][0]
            corners[i][0][1], corners[min_idx][0][1] = corners[min_idx][0][1], corners[i][0][1]

        for i in range(mid_corner, len(corners)):
            max_idx = i
            for j in range(i+1, len(corners)):
                if corners[max_idx][0][1] < corners[j][0][1]:
                    max_idx = j
            corners[i][0][0], corners[max_idx][0][0] = corners[max_idx][0][0], corners[i][0][0]
            corners[i][0][1], corners[max_idx][0][1] = corners[max_idx][0][1], corners[i][0][1]

        return corners


    def euclidean_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def findPositionCornerPoints(self, corner_x, contours):
        min_distance = float('inf')
        closest_index = -1

        for i, point in enumerate(contours):
            distance = self.euclidean_distance(point[0], corner_x[0])
            if distance < min_distance:
                min_distance = distance
                closest_index = i
                
        return closest_index

    
    def crop_image_from_point(self, image, center_x, center_y, crop_width, crop_height, blur = False):
        width , height = image.shape[1], image.shape[0]

        # Calculate the top-left and bottom-right coordinates of the crop region
        top_left_x = min(width, max(0, int(center_x - crop_width / 2)))
        top_left_y = min(height, max(0, int(center_y - crop_height / 2)))
        bottom_right_x = min(width, max(0, int(center_x + crop_width / 2)))
        bottom_right_y = min(height, max(0, int(center_y + crop_height / 2)))

        # Crop the image based on the specified region
        cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        if blur == True:
            cropped_image = cv2.GaussianBlur(cropped_image, (9,9),0)

        return cropped_image
    
        
    def findSpecCorner(self, cornerIndex, numberOfCorners,  threshold, mask, tempCorner, show = False, qualityOfLevel = 0.3, minOfDistance = 20, blockOfSize = 5, applyBlur = False):
        center_x = tempCorner[cornerIndex][0][0][0]  # X-coordinate of the center point
        center_y = tempCorner[cornerIndex][0][0][1]  # Y-coordinate of the center point
        crop_width = self.crop_w  # Width of the cropped region
        crop_height = self.crop_h  # Height of the cropped region

        imgGray_temp = self.crop_image_from_point(threshold, center_x, center_y, crop_width, crop_height, blur = applyBlur)
        imgColor_temp = cv2.cvtColor(imgGray_temp, cv2.COLOR_GRAY2BGR)

        mask_temp = self.crop_image_from_point(mask, center_x, center_y, crop_width, crop_height)
        #cv2_imshow(mask_temp)
        #mask_temp = cv2.cvtColor(mask_temp, cv2.COLOR_GRAY2BGR)

        #if show == True:
        #    cv2_imshow(mask_temp)

        mask_temp, corners_temp = self.drawCorners(mask_temp, imgColor_temp, numCorners = numberOfCorners, qualityLevel = qualityOfLevel, minDistance = minOfDistance, blockSize = blockOfSize, pointSize = 3)

        #cv2.imshow( "Corner {}".format(cornerIndex), imgColor_temp)

        return corners_temp, [center_x, center_y], [crop_width, crop_height], cornerIndex
    
        
    def draw_point_on_image(self, image, point):
        # Draw a point on the image

        image = cv2.circle(image, point, 5, (0, 0, 255), -1)
        return image

    def drawSpecCorner(self, corners_temp, center, crop, imgColor, imgGray, cornerList_temp, cornerIndex, show = False):
        center_x = center[0]
        center_y = center[1]
        crop_width = crop[0]
        crop_height = crop[1]

        x = center_x - crop_width // 2
        y = center_y - crop_height // 2

        if len(corners_temp) == 1:
            index = 0
            point_in_original = (x + corners_temp[index][0][0], y + corners_temp[index][0][1])
            #print(point_in_original)
            resultColor = self.draw_point_on_image(imgColor, point_in_original)
            resultGray = self.draw_point_on_image(imgGray, point_in_original)

            cornerList_temp.append([point_in_original, [cornerIndex]])

        else:
            higher_point = (x + corners_temp[0][0][0], y + corners_temp[0][0][1])
            resultColor = self.draw_point_on_image(imgColor, higher_point)
            resultGray = self.draw_point_on_image(imgGray, higher_point)
            lower_point = (x + corners_temp[1][0][0], y + corners_temp[1][0][1])
            resultColor = self.draw_point_on_image(imgColor, lower_point)
            resultGray = self.draw_point_on_image(imgGray, lower_point)

            if(higher_point[1] < lower_point[1]):
                temp = higher_point
                higher_point = lower_point
                lower_point = temp

            cornerList_temp.append([higher_point, [cornerIndex, 0]])
            cornerList_temp.append([lower_point, [cornerIndex, 1]])

        return resultColor, resultGray, cornerList_temp

    def finalSpecCorner(self, threshold, mask, imgColor, imgGray, tempCorner, show = False):
        cornerList_temp = []
        for i in range(self.numCorners):
            if i in self.positionThatHaveTwoCorners:
                corners_temp, center_temp, crop_temp, cornerIndex = self.findSpecCorner(i, 2, threshold, mask, tempCorner, show = False, qualityOfLevel = 0.1, minOfDistance = 5, blockOfSize = 10)
                resultColor, resultGray, cornerList_temp = self.drawSpecCorner(corners_temp, center_temp, crop_temp, imgColor, imgGray, cornerList_temp, cornerIndex)
            else:
                corners_temp, center_temp, crop_temp, cornerIndex = self.findSpecCorner(i, 1, threshold, mask, tempCorner, show = False, qualityOfLevel = 0.3, minOfDistance = 20, blockOfSize = 10)
                resultColor, resultGray, cornerList_temp = self.drawSpecCorner(corners_temp, center_temp, crop_temp, imgColor, imgGray, cornerList_temp, cornerIndex)
        #if show == True:
        #    cv2_imshow(finalResult)

 

        return resultColor, resultGray, cornerList_temp

    def process(self, image_path):
        # This is a method of the class
        # You can perform actions using instance variables and parameters
        # For example:
        imgColor, imgGray, mask = self.image_RGB_Gray_mask(image_path)
        self.image = imgColor.copy()

        self.threshold, contours = self.binary_contours(imgGray, self.thresholdType)
        

        imgColor_2, mask_2, cutpartContours_2, cnt1 = self.drawContours_imgColor_mask(contours, imgColor.copy(), mask.copy(), eps = 0.000001)
        self.cnt = cnt1
        cv2.imwrite('mask.png', mask_2)
        imgColor_1, mask_1, cutpartContours_1, cnt2 = self.drawContours_imgColor_mask(contours, imgColor.copy(), mask.copy(), eps = 0.003)

       
        mask_final, corners = self.drawCorners(mask_1, imgColor_1, self.numCorners)

        corners = self.sortCorners(corners)
        tempCorner = []
        positionCornerList = []

        for i in range(len(corners)):
            positionCornerList.append(self.findPositionCornerPoints(corners[i], cutpartContours_2))
            tempCorner.append([corners[i], self.findPositionCornerPoints(corners[i], cutpartContours_2)])

        self.finalImageColor, self.finalImageBinary, self.finalCorner = self.finalSpecCorner(self.threshold, mask_2, imgColor.copy(),  cv2.cvtColor(self.threshold, cv2.COLOR_GRAY2RGB).copy(), tempCorner, False)
        self.maskGeneralized = mask_1
        self.maskAccurate = mask_2



    

