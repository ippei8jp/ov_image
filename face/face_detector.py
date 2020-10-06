import sys
import numpy as np

# importのサーチパスを追加
sys.path.insert(0,"../common")

# ユーザ定義モジュール
from detector import Detector

# 顔検出器クラス
class FaceDetector(Detector):
    # 結果クラス
    class Result:
        # モデルの出力ノードの数(image_id, label, confidence, x1, y1, x2, y2)
        OUTPUT_SIZE = 7
        
        def __init__(self, output, frame_width, frame_height):
            self.raw_result = output
            self.image_id = output[0]
            self.label = int(output[1])
            self.confidence = output[2]
            self.position = np.array((output[3], output[4]))    # (x1, y1)
            self.size = np.array((output[5], output[6]))        # (x2, y2)  ここではまだ右下座標
            
            # pixel座標に変換
            self.position[0] *= frame_width
            self.position[1] *= frame_height
            self.size[0] = self.size[0] * frame_width - self.position[0]        # ここでsizeに変換される
            self.size[1] = self.size[1] * frame_height - self.position[1]
        
        def __getitem__(self, idx):
            return self.raw_result[idx]
        
        # ROIの拡大/縮小
        def rescale_roi(self, roi_scale_factor=1.0):
            self.position -= self.size * 0.5 * (roi_scale_factor - 1.0)
            self.size *= roi_scale_factor
        
        # ROIを元画像範囲に収める
        def clip(self, width, height):
            min = [0, 0]
            max = [width, height]
            self.position[:] = np.clip(self.position, min, max)
            self.size[:] = np.clip(self.size, min, max)
        
        # ROI範囲の画像切り取り
        def extract_image(self, frame):
            p1 = self.position.astype(int)
            p1 = np.clip(p1, [0, 0], [frame.shape[-1], frame.shape[-2]])
            p2 = (self.position + self.size).astype(int)
            p2 = np.clip(p2, [0, 0], [frame.shape[-1], frame.shape[-2]])
            return np.array(frame[:, :, p1[1]:p2[1], p1[0]:p2[0]])
    
    def __init__(self, model, confidence_threshold=0.5, roi_scale_factor=1.15):
        super(FaceDetector, self).__init__(model)
    
        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape
        self.output_shape = model.outputs[self.output_blob].shape
    
        assert len(self.output_shape) == 4 and \
               self.output_shape[3] == self.Result.OUTPUT_SIZE, \
            "Expected model output shape with %s outputs" % \
            (self.Result.OUTPUT_SIZE)
    
        assert 0.0 <= confidence_threshold and confidence_threshold <= 1.0, \
            "Confidence threshold is expected to be in range [0; 1]"
        self.confidence_threshold = confidence_threshold
    
        assert 0.0 < roi_scale_factor, "Expected positive ROI scale factor"
        self.roi_scale_factor = roi_scale_factor
    
    # 前処理
    def preprocess(self, frame):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        assert frame.shape[0] == 1
        assert frame.shape[1] == 3
        input = self.resize_input(frame, self.input_shape)
        return input
    
    # 認識処理開始
    def start_async(self, frame):
        input = self.preprocess(frame)
        self.enqueue(input)
    
    # 認識処理要求をキューイング
    def enqueue(self, input):
        return super(FaceDetector, self).enqueue({self.input_blob: input})
    
    # 認識結果取得
    def get_roi_proposals(self, frame):
        outputs = self.get_outputs()[0][self.output_blob]
        # outputs shape is [N_requests, 1, 1, N_max_faces, 7]
        
        frame_width = frame.shape[-1]
        frame_height = frame.shape[-2]
        
        results = []
        for output in outputs[0][0]:
            result = FaceDetector.Result(output, frame_width, frame_height)
            if result.confidence < self.confidence_threshold:
                break # results are sorted by confidence decrease
            
            result.rescale_roi(self.roi_scale_factor)
            result.clip(frame_width, frame_height)
            
            results.append(result)
        
        return results
