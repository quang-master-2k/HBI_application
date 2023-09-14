import cv2
import numpy as np

class secondMethod:
    def __init__(self, image, cnt, thickness):
        self.image = image
        self.cnt = cnt
        self.thickness = thickness

    def drawTorContours(self):
        mask_tor = np.zeros_like(self.image)
        approx = cv2.approxPolyDP(self.cnt, 0.000001*cv2.arcLength(self.cnt, True), True)
        cv2.drawContours(mask_tor, [approx], 0, (255, 255, 255), self.thickness)
        return mask_tor

    def getTorleranceArea(self, mask_tor):
        #Convert to HSV
        mask_tor_HSV = cv2.cvtColor(mask_tor, cv2.COLOR_BGR2HSV)

        #Define the lower and upper range for white color
        lower_white = np.array([0, 0, 255], dtype=np.uint8)
        upper_white = np.array([255, 25, 255], dtype=np.uint8)

        #Create white_mask_tor
        white_mask_tor = cv2.inRange(mask_tor_HSV, lower_white, upper_white)

        #White pixel coordinates
        white_tor_pixel_cor = cv2.findNonZero(white_mask_tor)

        return white_tor_pixel_cor