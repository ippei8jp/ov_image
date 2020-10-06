import os
import sys
import time
import math
import logging as log

# importのサーチパスを追加
sys.path.insert(0,"../common")

import cv2
import numpy as np

from visualizer import Visualizer

class FaceVisualizer(Visualizer):
    def __init__(self, args):
        super(FaceVisualizer, self).__init__(args)
    
    # 特徴点描画
    def draw_detection_landmarks(self, frame, roi, landmarks):
        keypoints = [landmarks.left_eye,
                     landmarks.right_eye,
                     landmarks.nose_tip,
                     landmarks.left_lip_corner,
                     landmarks.right_lip_corner]
        
        for point in keypoints:
            center = roi.position + roi.size * point
            cv2.circle(frame, tuple(center.astype(int)), 2, (0, 255, 255), 2)
    
    # 向き描画
    def draw_detection_headpose(self, frame, roi, headpose):
        cpoint = roi.position + roi.size / 2
        yaw   = headpose.yaw   * np.pi / 180.0
        pitch = headpose.pitch * np.pi / 180.0
        roll  = headpose.roll  * np.pi / 180.0
        
        yawMatrix = np.matrix([[math.cos(yaw), 0, -math.sin(yaw)], [0, 1, 0], [math.sin(yaw), 0, math.cos(yaw)]])                    
        pitchMatrix = np.matrix([[1, 0, 0],[0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
        rollMatrix = np.matrix([[math.cos(roll), -math.sin(roll), 0],[math.sin(roll), math.cos(roll), 0], [0, 0, 1]])                    
        
        #Rotational Matrix
        R = yawMatrix * pitchMatrix * rollMatrix
        rows=frame.shape[0]
        cols=frame.shape[1]
        
        cameraMatrix=np.zeros((3,3), dtype=np.float32)
        cameraMatrix[0][0]= 950.0
        cameraMatrix[0][2]= cols/2
        cameraMatrix[1][0]= 950.0
        cameraMatrix[1][1]= rows/2
        cameraMatrix[2][1]= 1
        
        xAxis=np.zeros((3,1), dtype=np.float32)
        xAxis[0]=50
        xAxis[1]=0
        xAxis[2]=0
        
        yAxis=np.zeros((3,1), dtype=np.float32)
        yAxis[0]=0
        yAxis[1]=-50
        yAxis[2]=0
        
        zAxis=np.zeros((3,1), dtype=np.float32)
        zAxis[0]=0
        zAxis[1]=0
        zAxis[2]=-50
        
        zAxis1=np.zeros((3,1), dtype=np.float32)
        zAxis1[0]=0
        zAxis1[1]=0
        zAxis1[2]=50
        
        o=np.zeros((3,1), dtype=np.float32)
        o[2]=cameraMatrix[0][0]
        
        xAxis=R*xAxis+o
        yAxis=R*yAxis+o
        zAxis=R*zAxis+o
        zAxis1=R*zAxis1+o
        
        p2x=int((xAxis[0]/xAxis[2]*cameraMatrix[0][0])+cpoint[0])
        p2y=int((xAxis[1]/xAxis[2]*cameraMatrix[1][0])+cpoint[1])
        cv2.line(frame,(cpoint[0],cpoint[1]),(p2x,p2y),(0,0,255),2)
        
        p2x=int((yAxis[0]/yAxis[2]*cameraMatrix[0][0])+cpoint[0])
        p2y=int((yAxis[1]/yAxis[2]*cameraMatrix[1][0])+cpoint[1])
        cv2.line(frame,(cpoint[0],cpoint[1]),(p2x,p2y),(0,255,0),2)
        
        p1x=int((zAxis1[0]/zAxis1[2]*cameraMatrix[0][0])+cpoint[0])
        p1y=int((zAxis1[1]/zAxis1[2]*cameraMatrix[1][0])+cpoint[1])
        
        p2x=int((zAxis[0]/zAxis[2]*cameraMatrix[0][0])+cpoint[0])
        p2y=int((zAxis[1]/zAxis[2]*cameraMatrix[1][0])+cpoint[1])
        
        cv2.line(frame,(p1x,p1y),(p2x,p2y),(255,0,0),2)
        cv2.circle(frame,(p2x,p2y),3,(255,0,0))
    
    # 検出結果描画
    def draw_detections(self, frame, rois, landmarks, headposes):
        for roi, landmark, headpose in zip(rois, landmarks, headposes) :
            # 検出枠描画
            label    = None
            bgcolor  = (  0,   0, 220)
            txtcolor = (255, 255, 255)
            self.draw_detection_roi(frame, roi, label, bgcolor, txtcolor)
            # 特徴点描画
            if landmark :
                self.draw_detection_landmarks(frame, roi, landmark)
            # 向き描画
            if headpose :
                self.draw_detection_headpose(frame, roi, headpose)

