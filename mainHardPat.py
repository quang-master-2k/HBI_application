from Yolo import YoloModel
from TraditionalCV import TraditionalCV
from CombineModel import CombineModel
from CalculateEdge import CalculateEdgeLength
import cv2
import numpy as np
from secondMethod import secondMethod


### HARD_PATTERN FULL CODE PROCESS

def hardpattern_points_process():
    image_path_hard = 'hardpart.jpg'
    input_image_hard = cv2.imread(image_path_hard)
    Yolo = YoloModel()
    Yolo.process(input_image_hard)
    yolo_corners = Yolo.keypoints_yolo

    TradCV = TraditionalCV(thresholdType = cv2.THRESH_BINARY, numCorners = 6, positionThatHaveTwoCorners=[1,4])
    TradCV.process(input_image_hard)
    cv_corners, mask = TradCV.finalCorner, TradCV.maskAccurate

    secondMethod_hardP = secondMethod(mask, TradCV.cnt, 23)
    mask_tor = secondMethod_hardP.drawTorContours()
    white_tor_pixels_cor = secondMethod_hardP.getTorleranceArea(mask_tor)

    Combine = CombineModel(yolo_corners, cv_corners, TradCV.threshold, mask_tor)
    Combine.process(num_corners = 6, mode = 'A')
    roi_hardP, rotation_matrix, transposed_matrix, center_point_hardP = Combine.cutting_image_2ndMethod()

    Edge = CalculateEdgeLength(mask, Combine.combineCorners, TradCV.threshold)
    Edge.process()
    edges_hardP = Edge.edgePointsList
    edges_hardP_cor_newlist = []
    edge_hardP_cor_newlist = []
    for edge in edges_hardP:
        for cor in edge[0]:
            rotated_pixel_cor = np.dot(rotation_matrix, np.array([int(cor[0]), int(cor[1]), 1])).tolist()
            transposed_pixel = [int(rotated_pixel_cor[0] - transposed_matrix[0]), int(rotated_pixel_cor[1] - transposed_matrix[1])]
            edge_hardP_cor_newlist.append(transposed_pixel)
        edges_hardP_cor_newlist.append([edge_hardP_cor_newlist, edge[1]])

    white_tor_cor_newlist = []
    for pixel in white_tor_pixels_cor:
        rotated_pixel = np.dot(rotation_matrix, np.array([int(pixel[0][0]), int(pixel[0][1]), 1])).tolist()
        transposed_pixel = [int(rotated_pixel[0] - transposed_matrix[0]), int(rotated_pixel[1] - transposed_matrix[1])]
        white_tor_cor_newlist.append(transposed_pixel)

    return roi_hardP, white_tor_cor_newlist, center_point_hardP, edges_hardP_cor_newlist

def cutpart_points_process():
    image_path_cut = 'image_128.jpg'
    input_image_cut = cv2.imread(image_path_cut)
    Yolo = YoloModel()
    Yolo.process(input_image_cut)
    yolo_corners = Yolo.keypoints_yolo

    TradCV = TraditionalCV(thresholdType = cv2.THRESH_BINARY, numCorners = 6, positionThatHaveTwoCorners=[1,4])
    TradCV.process(input_image_cut)
    cv_corners, mask = TradCV.finalCorner, TradCV.maskAccurate

    Combine = CombineModel(yolo_corners, cv_corners, TradCV.threshold, mask)
    Combine.process(num_corners = 6, mode = 'A')
    roi_cut, rotation_matrix, transposed_matrix, center_point_cut = Combine.cutting_image_2ndMethod()

    Edge = CalculateEdgeLength(mask, Combine.combineCorners, TradCV.threshold)
    Edge.process()
    edges = Edge.edgePointsList

    cnts_cor_newlist = []
    for edge in edges:
        edge_cor_newlist = []
        for pixel in edge[0]:
            rotated_pixel = np.dot(rotation_matrix, np.array([int(pixel[0]), int(pixel[1]), 1])).tolist()
            transposed_pixel = [int(rotated_pixel[0] - transposed_matrix[0]), int(rotated_pixel[1] - transposed_matrix[1])]
            edge_cor_newlist.append(transposed_pixel)
        cnts_cor_newlist.append([edge_cor_newlist, edge[1]])
    return cnts_cor_newlist, center_point_cut

def merge_cutPart_to_hardP(center_point_hardP, center_point_cut, cnt_cor_newlist):
    for edge in cnts_cor_newlist:
        for cor in edge[0]:
            cor[0] = cor[0] + center_point_hardP[0] - center_point_cut[0]
            cor[1] = cor[1] + center_point_hardP[1] - center_point_cut[1] 
    return cnt_cor_newlist


roi_hardP, white_tor_cor_newlist, center_point_hardP, edges_hardP = hardpattern_points_process()
cnts_cor_newlist, center_point_cut = cutpart_points_process()
cnts_cor_newlist = merge_cutPart_to_hardP(center_point_hardP, center_point_cut, cnts_cor_newlist)

for i in edges_hardP:
    for j in i[0]:
        cv2.circle(roi_hardP, j, 1, (0, 255, 255), -1)


error_edges = []
for i in cnts_cor_newlist:
    error_points = []
    for j in i[0]:
        if j in white_tor_cor_newlist:
            cv2.circle(roi_hardP, j, 1, (255, 0, 0), -1)
        else:
            cv2.circle(roi_hardP, j, 1, (0, 0, 255), -1)
            error_points.append(j)
    error_edges.append([error_points, i[1]])

error_dis_and_point = []
for i in range(len(error_edges)):
    if len(error_edges[i][0]) != 0:
        sort_error_edge = []
        for point in error_edges[i][0]:
            min_dis = 1000000
            for cor in edges_hardP[i][0]:
                distance = ((point[0] - cor[0])**2 + (point[1] - cor[1])**2)**(1/2)
                if distance < min_dis:
                    min_dis = distance
            sort_error_edge.append([point, min_dis])
        sort_error_edge = sorted(sort_error_edge, key=lambda x:x[1], reverse=True)
        error_dis_and_point.append([sort_error_edge[0], i])
    else:
        error_dis_and_point.append([[None, None], i])

print(error_dis_and_point)

cv2.imshow("roi_hardP_128.jpg", roi_hardP)

cv2.waitKey(0)
cv2.destroyAllWindows()

