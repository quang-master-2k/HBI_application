# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QTableView
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QFont
import sqlite3
from Yolo import YoloModel
from TraditionalCV import TraditionalCV
from CombineModel import CombineModel
from CalculateEdge import CalculateEdgeLength
from secondMethod import secondMethod

class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        QtCore.QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return str(self._data.columns[section])
        return None

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1409, 927)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(110, 42, 101, 31))
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 50, 101, 17))
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(220, 50, 101, 17))
        self.label_2.setObjectName("label_2")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(320, 42, 101, 31))
        self.textEdit_2.setObjectName("textEdit_2")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(430, 50, 101, 17))
        self.label_3.setObjectName("label_3")
        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(480, 40, 101, 31))
        self.textEdit_3.setObjectName("textEdit_3")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(590, 48, 101, 17))
        self.label_4.setObjectName("label_4")
        self.textEdit_4 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_4.setGeometry(QtCore.QRect(620, 42, 101, 31))
        self.textEdit_4.setObjectName("textEdit_4")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(740, 50, 89, 25))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.set_start)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(850, 50, 89, 25))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setEnabled(False)
        self.pushButton_2.clicked.connect(self.display_image_measurement)

        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(10, 90, 722, 482))
        self.graphicsView.setObjectName("graphicsView")

        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(740, 90, 351, 481))
        self.tableView.setObjectName("tableView")


        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 10, 141, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(960, 50, 89, 25))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setEnabled(False)
        self.pushButton_3.clicked.connect(self.end_measurement)

        self.tableView_2 = QtWidgets.QTableView(self.centralwidget)
        self.tableView_2.setGeometry(QtCore.QRect(10, 580, 291, 281))
        self.tableView_2.setObjectName("tableView_2")

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(590, 600, 191, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")

        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(1110, 90, 200, 17))
        self.label_7.setObjectName("label_7")

        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(1110, 120, 200, 17))
        self.label_8.setObjectName("label_8")

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(650, 830, 89, 25))
        self.pushButton_4.setObjectName("pushButton_4")


        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1409, 22))
        self.menuBar.setObjectName("menuBar")
        self.menuSecond = QtWidgets.QMenu(self.menuBar)
        self.menuSecond.setObjectName("menuSecond")
        self.menuSetting = QtWidgets.QMenu(self.menuBar)
        self.menuSetting.setObjectName("menuSetting")
        MainWindow.setMenuBar(self.menuBar)
        self.actionDimension = QtWidgets.QAction(MainWindow)
        self.actionDimension.setObjectName("actionDimension")
        self.actionSecond = QtWidgets.QAction(MainWindow)
        self.actionSecond.setObjectName("actionSecond")
        self.menuSecond.addAction(self.actionDimension)
        self.menuSecond.addAction(self.actionSecond)
        self.menuBar.addAction(self.menuSecond.menuAction())
        self.menuBar.addAction(self.menuSetting.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Garment Style"))
        self.label_2.setText(_translate("MainWindow", "Pattern code"))
        self.label_3.setText(_translate("MainWindow", "Name"))
        self.label_4.setText(_translate("MainWindow", "Size"))
        self.pushButton.setText(_translate("MainWindow", "Start"))
        self.pushButton_2.setText(_translate("MainWindow", "Measure"))
        self.label_5.setText(_translate("MainWindow", "Measurement"))
        self.pushButton_3.setText(_translate("MainWindow", "End"))
        self.label_6.setText(_translate("MainWindow", "Final result report"))
        self.label_7.setText(_translate("MainWindow", "Torelance: 0"))
        self.label_8.setText(_translate("MainWindow", "Checking: 0"))
        self.pushButton_4.setText(_translate("MainWindow", "Export"))
        self.menuSecond.setTitle(_translate("MainWindow", "Measurement"))
        self.menuSetting.setTitle(_translate("MainWindow", "Setting"))
        self.actionDimension.setText(_translate("MainWindow", "Dimension"))
        self.actionSecond.setText(_translate("MainWindow", "Second"))

    ### Camera function
    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(0)  # Update frame every 30 milliseconds

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            image = QImage(
                frame_rgb.data, w, h, ch * w, QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(image)
            pixmap = pixmap.scaled(720, 480)
            self.scene.clear()
            self.scene.addPixmap(pixmap)

    def video_streaming(self):
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.timer = QtCore.QTimer(self.graphicsView)
        self.timer.timeout.connect(self.update_frame)
        self.start_webcam()


    ### Visualization function
    error_dis_and_point = []
    def display(self):
        self.scene1 = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.scene1)
        img, self.error_dis_and_point = self.cam_process()
        imgshow = np.array(img)
        height, width, channel = imgshow.shape
        bytes_per_line = channel * width    
        image = QImage(imgshow.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(720, 480)
        self.scene1.clear()
        self.scene1.addPixmap(pixmap)

    def display_image_measurement(self):
        self.display()
        self.show_measurement()
        self.AQL_table()

    def show_measurement(self):
        error_list = self.error_dis_and_point
        error_edges = []
        error_dis = []
        error_points = []
        dfMea = pd.DataFrame()
        for i in error_list:
            error_edges.append(i[1])
            error_dis.append(i[0][1])
            error_points.append(i[0][0])
        dfMea['Edge'] = error_edges
        dfMea['Distance'] = error_dis
        dfMea['Coordinates'] = error_points

        headers_Mea = list(dfMea.head(0))
        modelMea = PandasModel(dfMea)
        self.tableView.setModel(modelMea)


    AQL_count = 1
    OrdNum = []
    ResultAQL = []
    run = 0
    def AQL_table(self):
        dfAQL = pd.DataFrame()
        if self.run == 0:
            self.label_8.setText("Checking: " + str(self.AQL_count))

            self.OrdNum.append(self.AQL_count)
            
            # Name_AQL = []
            text = self.get_specs_info()
            # textAQL = text[1] + text[3]
            # Name_AQL.append[textAQL]
            

            self.ResultAQL.append('Accepted')

            dfAQL['OrdNum'] = self.OrdNum
            # dfAQL['NameAQL'] = Name_AQL
            dfAQL['Result'] = self.ResultAQL

            headers = list(dfAQL.head(0))
            modelAQL = PandasModel(dfAQL)
            self.tableView_2.setModel(modelAQL)
            self.AQL_count += 1

            if text[1] == "BB":
                self.label_7.setText("Tolerance: 1/8")
            else:
                self.label_7.setText("Tolerance: 1/16")
        else:
            dfAQL['OrdNum'] = self.OrdNum
            # dfAQL['NameAQL'] = Name_AQL
            dfAQL['Result'] = self.ResultAQL
            return dfAQL

    
    ### Start function
    def set_start(self):
        self.textEdit.setText("")
        self.textEdit_2.setText("")
        self.textEdit_3.setText("")
        self.textEdit_4.setText("")
        self.tableView.setModel(None)
        self.tableView_2.setModel(None)
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.timer = QtCore.QTimer(self.graphicsView)
        self.timer.timeout.connect(self.update_frame)
        self.start_webcam()
        self.AQL_count = 1
        self.OrdNum = []
        self.ResultAQL = []
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(True)
        self.pushButton_3.setEnabled(True)
        self.run = 0
        self.label_7.setText("Tolerance: 0")
        self.label_8.setText("Checking: 0")


    ### Camera processing
    def hardpattern_points_process(self):
        image_path_hard = 'hardpart.jpg'
        input_image_hard = cv2.imread(image_path_hard)
        Yolo = YoloModel()
        Yolo.process(input_image_hard)
        yolo_corners = Yolo.keypoints_yolo

        TradCV = TraditionalCV(thresholdType = cv2.THRESH_BINARY, numCorners = 6, positionThatHaveTwoCorners=[1,4])
        TradCV.process(input_image_hard)
        cv_corners, mask = TradCV.finalCorner, TradCV.maskAccurate

        ### Thickness is 23. Modify it to suitable with torelance
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

    def cutpart_points_process(self):
        ### Modify to frame from camera
        image_path_cut = 'image_124.jpg'
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

    def cam_process(self):
        roi_hardP, white_tor_cor_newlist, center_point_hardP, edges_hardP = self.hardpattern_points_process()
        cnts_cor_newlist, center_point_cut = self.cutpart_points_process()
        for edge in cnts_cor_newlist:
            for cor in edge[0]:
                cor[0] = cor[0] + center_point_hardP[0] - center_point_cut[0]
                cor[1] = cor[1] + center_point_hardP[1] - center_point_cut[1]
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
        return roi_hardP, error_dis_and_point

    ### End function
    countPass = 0
    def end_measurement(self):
        self.cap.release()
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        
        self.run = 1 
        dfEnd = self.AQL_table()
        for i in range(len(dfEnd['Result'])):
            if dfEnd['Result'][i] == 'Accepted':
                self.countPass += 1

        P_pass = self.countPass / (self.AQL_count-1)
        self.label_7.setText("Pass: " + str(P_pass*100) + '%')
        self.label_8.setText("Reject: " + str((1-P_pass)*100) + '%')

    
    ### Additional function
    def get_specs_info(self):
        textEdits = [
            self.textEdit,
            self.textEdit_2,
            self.textEdit_3,
            self.textEdit_4
        ]

        texts = []
        for textEdit in textEdits:
            text = textEdit.toPlainText()
            texts.append(text)
   
        return texts

    def take_photo(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
