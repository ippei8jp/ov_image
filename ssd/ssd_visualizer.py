import os
import sys
import time
import math
import logging as log

# importのサーチパスを追加
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"../common"))

import cv2
import numpy as np

from visualizer import Visualizer

class SsdVisualizer(Visualizer):
    # カラーパレット(8bitマシン風。ちょっと薄目)
    COLOR_PALETTE = [   #   B    G    R 
                    ( 128, 128, 128),         # 0 (灰)
                    ( 255, 128, 128),         # 1 (青)
                    ( 128, 128, 255),         # 2 (赤)
                    ( 255, 128, 255),         # 3 (マゼンタ)
                    ( 128, 255, 128),         # 4 (緑)
                    ( 255, 255, 128),         # 5 (水色)
                    ( 128, 255, 255),         # 6 (黄)
                    ( 255, 255, 255)          # 7 (白)
                ]
    
    def __init__(self, args):
        super(SsdVisualizer, self).__init__(args)
    
    # 検出結果描画
    def draw_detections(self, frame, rois):
        for roi in rois :
            # 検出枠描画
            label    = roi.label
            bgcolor  = self.COLOR_PALETTE[roi.class_id & 0x7]   # 表示色(IDの下一桁でカラーパレットを切り替える)
            txtcolor = (0, 0, 0)
            self.draw_detection_roi(frame, roi, label, bgcolor, txtcolor)
