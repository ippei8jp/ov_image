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
from ssd_detector import SsdDetector

class SsdFrameProcessor(FrameProcessor) :
    def __init__(self, args):
        # 推論に使用するデバイス一覧
        used_devices = set([args.d_ssd])
        
        super(SsdFrameProcessor, self).__init__(args, used_devices)
        
        # モデルのロード
        log.info("Loading models")
        if (args.m_ssd) :
            log.info("    SSD model")
            ssd_detector_net = self.load_model(args.m_ssd)
            self.ssd_detector = SsdDetector(ssd_detector_net,
                                              confidence_threshold=args.t_ssd)
            self.ssd_detector.deploy(args.d_ssd, self.iecore)
            
            # ラベルのロード
            label_path = os.path.splitext(args.m_ssd)[0] + ".labels"
            self.ssd_detector.labels_map = self.load_label(label_path)
        
        else :
            # SSDモデルが指定されていなければエラー
            log.error("--m_ssd option is mandatory")
            raise RuntimeError("--m_ssd option is mandatory")
        
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
        self.ssd_detector.clear()
        
        # log.info("Face Detect")
        # 認識処理
        self.ssd_detector.start_async(frame)
        
        # 結果取得
        rois = self.ssd_detector.get_roi_proposals(frame)
        
        return rois
    
    # パフォーマンスステータスの取得
    def get_performance_stats(self):
        stats = {'ssd_detector': self.ssd_detector.get_performance_stats()}
        return stats
