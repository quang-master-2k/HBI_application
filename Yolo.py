import cv2
from ultralytics import YOLO
from skimage.filters.thresholding import threshold_isodata

class YoloModel:
    def __init__(self, model_path = 'best640_3.pt', thresholdType = cv2.THRESH_BINARY):
        # This is the constructor method
        # Initialize instance variables here
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.thresholdType = thresholdType
        self.keypoints_yolo = None
        self.image = None
        self.imageBinary = None

    def image_gray(self, input_image):
        imgGray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        return imgGray

    def binary_contours(self, input_imgGray, thresholdType):
        binaryThreshold = threshold_isodata(input_imgGray)
        _, threshold = cv2.threshold(input_imgGray, binaryThreshold, 255, thresholdType)
        return threshold

    def process(self, image_path):
        self.imageBinary =  cv2.cvtColor(self.binary_contours(self.image_gray(image_path), self.thresholdType), cv2.COLOR_GRAY2RGB)

        self.keypoints_yolo = self.model(self.imageBinary)[0].keypoints.xy.tolist()
        self.image = image_path
    
    def create_corner_image(self):
        for point_indx, point in enumerate(self.keypoints_yolo[0]):
            cv2.circle(self.imageBinary, tuple([int(point[0]), int(point[1])]), 1, (0, 0, 255), -1)
            cv2.putText(self.imageBinary, str(point_indx), (int(point[0]), int(point[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

        cv2.imwrite('yolo.png', self.imageBinary)



  

