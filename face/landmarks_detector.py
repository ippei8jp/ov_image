import sys
import os
import numpy as np

# importのサーチパスを追加
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"../common"))

# ユーザ定義モジュール
from detector import Detector

# 特徴点検出器クラス
class LandmarksDetector(Detector):
    # 結果クラス
    class Result:
        # 検出される特徴点の数(左目,右目, 鼻, 唇左端, 唇右端)
        OUTPUT_SIZE = 5
        
        def __init__(self, output):
            self.raw_result = output
            self.left_eye = output[0]
            self.right_eye = output[1]
            self.nose_tip = output[2]
            self.left_lip_corner = output[3]
            self.right_lip_corner = output[4]
        
        def __getitem__(self, idx):
            return self.raw_result[idx]
        
        def get_array(self):
            return np.array(self.raw_result, dtype=np.float64)
    
    def __init__(self, model):
        # 基底クラスのコンストラクタ呼び出し
        super(LandmarksDetector, self).__init__(model)
        
        # パラメータ等チェック
        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape
        
        assert np.array_equal([1, self.Result.OUTPUT_SIZE * 2, 1, 1], model.outputs[self.output_blob].shape), \
            f"Expected model output shape {[1, self.Result.OUTPUT_SIZE * 2, 1, 1]}, but got {model.outputs[self.output_blob].shape}"
    
    # 前処理
    def preprocess(self, frame, rois):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = [roi.extract_image(frame) for roi in rois]
        inputs = [self.resize_input(input, self.input_shape) for input in inputs]
        return inputs
    
    # 認識処理開始
    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)
    
    # 認識処理要求をキューイング
    def enqueue(self, input):
        return super(LandmarksDetector, self).enqueue({self.input_blob: input})
    
    # 認識結果取得
    def get_landmarks(self):
        outputs = self.get_outputs()
        results = [LandmarksDetector.Result(out[self.output_blob].reshape((-1, 2))) \
                      for out in outputs]
        return results
