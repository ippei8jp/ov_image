import os
import sys
import time
import logging as log

# importのサーチパスを追加
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"../common"))

import numpy as np

from openvino.inference_engine import IECore

# ユーザ定義モジュール
from frame_processor import FrameProcessor
from yolo_detector import YoloDetector

class YoloFrameProcessor(FrameProcessor) :
    def __init__(self, args):
        # 推論に使用するデバイス一覧
        used_devices = set([args.d_yolo])
        
        super(YoloFrameProcessor, self).__init__(args, used_devices)
        
        # モデルのロード
        log.info("Loading models")
        if (args.m_yolo) :
            log.info("    YOLO model")
            yolo_detector_net = self.load_model(args.m_yolo)
            self.yolo_detector = YoloDetector(yolo_detector_net,
                                              confidence_threshold=args.t_yolo, iou_threshold=args.iou_threshold)
            self.yolo_detector.deploy(args.d_yolo, self.iecore)
            
            # ラベルのロード
            label_path = os.path.splitext(args.m_yolo)[0] + ".labels"
            self.yolo_detector.labels_map = self.load_label(label_path)
        
        else :
            # YOLOモデルが指定されていなければエラー
            log.error("--m_yolo option is mandatory")
            raise RuntimeError("--m_yolo option is mandatory")
        
        log.info("Models are loaded")
    
    # フレーム毎の処理
    def process(self, frame):
        assert len(frame.shape) == 3,    "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], "Expected BGR or BGRA input"
        
        # 前処理
        # orig_image = frame.copy()
        frame = frame.transpose((2, 0, 1)) # HWC to CHW
        frame = np.expand_dims(frame, axis=0)
        
        # 検出器初期化
        self.yolo_detector.clear()
        
        # log.info("Face Detect")
        # 認識処理
        self.yolo_detector.start_async(frame)
        
        # 結果取得
        rois = self.yolo_detector.get_roi_proposals(frame)
        
        return rois
    
    # パフォーマンスステータスの取得
    def get_performance_stats(self):
        stats = {'yolo_detector': self.yolo_detector.get_performance_stats()}
        return stats
