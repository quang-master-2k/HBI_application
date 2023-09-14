import cv2
from sympy import symbols, Eq, solve
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq

class CombineModel:
    def __init__(self, yolo_corner, opencv_corner, thresholdImage, mask):
        # This is the constructor method
        # Initialize instance variables here
        self.thresholdImage = thresholdImage
        self.yolo_corner = yolo_corner
        self.opencv_corner = opencv_corner
        self.mask = mask
        self.corners_with_mode = None
        self.combineCorners = None
        self.imageWithMode = None
        self.imageOnlyCorners = None

    def euclidean_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def find_nearest_element(self, A, B):
        min_distance = float('inf')
        nearest_index = None

        for idx, element in enumerate(B):
            if len(element[1]) == 1:
                distance = self.euclidean_distance(A, element[0])
                if distance < min_distance:
                    min_distance = distance
                    nearest_index = idx

        return nearest_index
    
    def checkCircleindex(self, firstNum, nextNum, length):
        # Checking whether the indices of elements in an array form a consecutive sequence in a circular manner
        if firstNum == nextNum - 1:
            return True
        else:
            if firstNum == length - 1 and nextNum == 0:
                return True
            else:
                return False
            
    def changeTupleElementToList(self, oldList):
        newList = []
        for i in oldList:
            newList.append(list(i))

        return newList

    def process(self, num_corners, mode):
        # This is another method of the class
        # You can pass arguments and perform actions
        # For example:
        
        keypoints_yolo = self.yolo_corner
        finalCorner = self.opencv_corner

        sort_comparison_pair = []
        #Finding number 2 key point
        keypoint_number2 = keypoints_yolo[0][2]
        nearest_index = self.find_nearest_element(keypoint_number2, finalCorner)
        sort_comparison_pair.append([finalCorner[nearest_index][1][0], 2])

        # Finding number 3 key point
        keypoint_number3 = keypoints_yolo[0][3]
        nearest_index = self.find_nearest_element(keypoint_number3, finalCorner)
        sort_comparison_pair.append([finalCorner[nearest_index][1][0], 3])

        # Checking if the index is wrong
        if self.checkCircleindex(sort_comparison_pair[0][0], sort_comparison_pair[1][0], num_corners):
            step = np.abs(sort_comparison_pair[0][1] - sort_comparison_pair[0][0])
            for index, value in enumerate(finalCorner):
                finalCorner[index][1][0] = (finalCorner[index][1][0] + step) % num_corners

                if len(finalCorner[index][1]) == 2:

                    if finalCorner[index][1][1] == 0:
                        finalCorner[index][1][1] = 'A'
                    else:
                        finalCorner[index][1][1] = 'B'

            self.corners_with_mode = finalCorner.copy()
        else:
            print("Error!!!")

        if mode == 'A':
            tempCornerFinal = []

            # Can code toi uu hon
            for index, value in enumerate(finalCorner):
                if len(finalCorner[index][1]) == 2:
                    if finalCorner[index][1][1] == 'A':
                        tempCornerFinal.append([value[0], value[1][0]])
                else:
                    tempCornerFinal.append([value[0], value[1][0]])

            tempCornerFinal = sorted(tempCornerFinal, key=lambda x: x[1])
            for index, value in enumerate(tempCornerFinal):
                tempCornerFinal[index] = tempCornerFinal[index][0]

        elif mode == 'B':
            tempCornerFinal = []

            # Can code toi uu hon
            for index, value in enumerate(finalCorner):
                if len(finalCorner[index][1]) == 2:
                    if finalCorner[index][1][1] == 'B':
                        tempCornerFinal.append([value[0], value[1][0]])
                else:
                    tempCornerFinal.append([value[0], value[1][0]])

            tempCornerFinal = sorted(tempCornerFinal, key=lambda x: x[1])
            for index, value in enumerate(tempCornerFinal):
                tempCornerFinal[index] = tempCornerFinal[index][0]

        tempCornerFinal = self.changeTupleElementToList(tempCornerFinal)

        self.combineCorners = tempCornerFinal
        self.imageWithMode = cv2.cvtColor(self.thresholdImage.copy(), cv2.COLOR_GRAY2RGB)
        self.imageOnlyCorners = cv2.cvtColor(self.thresholdImage.copy(), cv2.COLOR_GRAY2RGB)

        for point_indx, point in enumerate(tempCornerFinal):
                    cv2.circle(self.imageWithMode, tuple([int(point[0]), int(point[1])]), 6, (0, 0, 255), -1)
                    cv2.circle(self.imageOnlyCorners, tuple([int(point[0]), int(point[1])]), 6, (0, 0, 255), -1)
                    cv2.putText(self.imageWithMode, str(point_indx), (int(point[0]), int(point[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (148, 0, 211), 5)
                    
    def cutting_image_2ndMethod(self):
        original_img = self.mask
        cv_corners = self.combineCorners
        for index, value in enumerate(cv_corners):
            if index == 0:
                a0 = value
            if index == 5:
                a5 = value
            if index == 2:
                a2 = value
            if index == 3:
                a3 = value
        x_corner = int((((a0[0] + a5[0])/2) + ((a2[0] + a3[0])/2)) / 2)
        y_corner = int((((a0[1] + a5[1])/2) + ((a2[1] + a3[1])/2)) / 2)
        # cv2.circle(original_img, (x_corner, y_corner), 8, (255,0,0), -1)

        # Define the variables x and y as symbols
        x, y = symbols('x y')

        #Top-left and bottom-right

        points1 = [[x_corner, y_corner], a0]
        x_coords1, y_coords1 = zip(*points1)
        A = vstack([x_coords1,ones(len(x_coords1))]).T
        m1, c1 = lstsq(A, y_coords1)[0]

        # Define the equations
        eq1 = Eq(-m1*x + 1*y, c1)
        eq2 = Eq(x**2 - 2*x_corner*x + y**2 - 2*y_corner*y, (x_corner - a0[0])**2 + (y_corner - a0[1])**2 - x_corner**2 - y_corner**2 + 315**2)
        
        # Solve the system of equations
        solution1 = solve((eq1, eq2), (x, y))

        # Print the solution

        # print(solution1)
        # cv2.circle(original_img, (int(solution1[0][0]), int(solution1[0][1])), 15, (255,0,0), -1)
        # cv2.circle(original_img, (int(solution1[1][0]), int(solution1[1][1])), 15, (255,0,0), -1)

        #Bottom-left and top-right
        points2 = [[x_corner, y_corner], a5]
        x_coords2, y_coords2 = zip(*points2)
        A = vstack([x_coords2,ones(len(x_coords2))]).T
        m2, c2 = lstsq(A, y_coords2)[0]

        # Define the equations
        eq3 = Eq(-m2*x + 1*y, c2)
        eq4 = Eq(x**2 - 2*x_corner*x + y**2 - 2*y_corner*y, (x_corner - a5[0])**2 + (y_corner - a5[1])**2 - x_corner**2 - y_corner**2 + 315**2)
    
        # Solve the system of equations
        solution2 = solve((eq3, eq4), (x, y))

        # Print the solution
        # print(solution2)
        # cv2.circle(original_img, (int(solution2[0][0]), int(solution2[0][1])), 15, (255,0,0), -1)
        # cv2.circle(original_img, (int(solution2[1][0]), int(solution2[1][1])), 15, (255,0,0), -1)

        color = (0, 255, 0)  # Specify the color in BGR format (here, green)
        thickness = 5  # Specify the thickness of the border
        # cv2.line(original_img, (int(solution1[0][0]), int(solution1[0][1])), (int(solution2[0][0]), int(solution2[0][1])), color, thickness)
        # cv2.line(original_img, (int(solution2[0][0]), int(solution2[0][1])), (int(solution1[1][0]), int(solution1[1][1])), color, thickness)
        # cv2.line(original_img, (int(solution1[1][0]), int(solution1[1][1])), (int(solution2[1][0]), int(solution2[1][1])), color, thickness)
        # cv2.line(original_img, (int(solution2[1][0]), int(solution2[1][1])), (int(solution1[0][0]), int(solution1[0][1])), color, thickness)

        # maskCropped = np.zeros_like(original_img)
        # points = np.array([[int(solution1[0][0]), int(solution1[0][1])], [int(solution2[0][0]), int(solution2[0][1])], [int(solution1[1][0]), int(solution1[1][1])], [int(solution2[1][0]), int(solution2[1][1])]])
        # cv2.fillPoly(maskCropped, [points], (255, 255, 255))
        # cropped_image = cv2.bitwise_and(original_img, maskCropped)

        point1 = (int(solution1[0][0]), int(solution1[0][1]))
        point2 = (int(solution2[1][0]), int(solution2[1][1]))
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle = np.arctan2(dy, dx) * (180.0 / np.pi)
        
        # Rotates an image by the given angle and expands to avoid cropping
        height, width = original_img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated_image = cv2.warpAffine(original_img, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        new_top_left_float = np.dot(rotation_matrix, np.array([int(solution1[0][0]), int(solution1[0][1]), 1])).tolist()
        new_top_right_float = np.dot(rotation_matrix, np.array([int(solution2[1][0]), int(solution2[1][1]), 1])).tolist()
        new_bot_right_float = np.dot(rotation_matrix, np.array([int(solution1[1][0]), int(solution1[1][1]), 1])).tolist()
        new_bot_left_float = np.dot(rotation_matrix, np.array([int(solution2[0][0]), int(solution2[0][1]), 1])).tolist()

        new_top_left = [int(new_top_left_float[0]),int(new_top_left_float[1])]
        new_top_right = [int(new_top_right_float[0]),int(new_top_right_float[1])]
        new_bot_right = [int(new_bot_right_float[0]),int(new_bot_right_float[1])]
        new_bot_left = [int(new_bot_left_float[0]),int(new_bot_left_float[1])]

        # cv2.circle(rotated_image, new_top_right, 15, (255,255,0), -1)
        # print(new_bot_left, new_bot_right, new_top_left, new_top_right)

        width = new_top_right[0] - new_top_left[0]
        height = new_bot_left[1] - new_top_left[1]

        center_point = [int(width/2), int(height/2)]

        # Extract the region of interest
        roi = rotated_image[new_top_left[1]:new_top_left[1]+height, new_top_left[0]:new_top_left[0]+width]

        return roi, rotation_matrix, new_top_left, center_point