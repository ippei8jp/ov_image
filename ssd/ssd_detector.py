import sys
import os
import numpy as np

# importのサーチパスを追加
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"../common"))

# ユーザ定義モジュール
from detector import Detector

# SSDクラス
class SsdDetector(Detector):
    # 結果クラス
    class Result:
        # モデルの出力ノードの数(image_id, label, confidence, x1, y1, x2, y2)
        OUTPUT_SIZE = 7
        
        def __init__(self, output, frame_width, frame_height, labels_map):
            self.raw_result = output
            class_id = int(output[1])
            self.class_id = class_id
            if labels_map :
                if len(labels_map) > class_id :
                    class_name = labels_map[class_id]
                else :
                    class_name = str(class_id)
            else :
                class_name = str(class_id)
            
            self.confidence = output[2]
            self.label =  f"{class_name}  {round(self.confidence * 100, 1)}%"
            
            self.position = np.array((output[3], output[4]))    # (x1, y1)
            self.size = np.array((output[5], output[6]))        # (x2, y2)  ここではまだ右下座標
            
            # pixel座標に変換
            self.position[0] *= frame_width
            self.position[1] *= frame_height
            self.size[0] = self.size[0] * frame_width - self.position[0]         # ここでsizeに変換される
            self.size[1] = self.size[1] * frame_height - self.position[1]
        
        def __getitem__(self, idx):
            return self.raw_result[idx]
        
        # ROIを元画像範囲に収める
        def clip(self, width, height):
            min = [0, 0]
            max = [width, height]
            self.position[:] = np.clip(self.position, min, max)
            self.size[:] = np.clip(self.size, min, max)
    
    def __init__(self, model, confidence_threshold=0.5):
        super(SsdDetector, self).__init__(model)
        
        # このプログラムは1出力のモデルのみサポートしているので、チェック
        # print(model.outputs)
        assert len(model.outputs) == 1, "Expected 1 output blob"
        # SSDのinputsは1とは限らないのでスキャンする
        img_info_input_blob = None
        if hasattr(model, 'input_info') :        # 2021以降のバージョン
            inputs = model.input_info
        else :
            inputs = model.inputs

        for blob_name in inputs:
            if hasattr(inputs[blob_name], 'shape') :        # 2020以前のバージョン
                input_shape = inputs[blob_name].shape
            else :                                          # 2021以降のバージョン
                input_shape = inputs[blob_name].input_data.shape
            # print(f'{blob_name}   {input_shape}')
            if len(input_shape) == 4:
                input_blob = blob_name
            elif len(input_shape) == 2:
               # こういう入力レイヤがあるものがある？モノクロ画像？
               img_info_input_blob = blob_name
            else:
                raise RuntimeError(f"Unsupported {len(input_shape)} input layer '{ blob_name}'. Only 2D and 4D input layers are supported")
        
        self.input_blob = input_blob
        if hasattr(inputs[input_blob], 'shape') :       # 2020以前のバージョン
            input_shape = inputs[input_blob].shape
        else :                                          # 2021以降のバージョン
            input_shape = inputs[input_blob].input_data.shape
        self.input_shape = input_shape
        self.img_info = None

        if img_info_input_blob:
            self.img_info_input_blob = img_info_input_blob
            self.img_info = [self.input_shape[2], self.input_shape[3], 1]
        else :
            self.img_info_input_blob = None
        
        self.output_blob = next(iter(model.outputs))
        self.output_shape = model.outputs[self.output_blob].shape
        assert len(self.output_shape) == 4 and \
               self.output_shape[3] == self.Result.OUTPUT_SIZE, \
            f"Expected model output shape with {self.Result.OUTPUT_SIZE} outputs"
        
        assert 0.0 <= confidence_threshold and confidence_threshold <= 1.0, \
            "Confidence threshold is expected to be in range [0; 1]"
        self.confidence_threshold = confidence_threshold
    
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
        feed_dict = {self.input_blob: input}
        if self.img_info_input_blob :
            feed_dict[self.img_info_input_blob] = self.img_info
        
        return super(SsdDetector, self).enqueue(feed_dict)
    
    # 認識結果取得
    def get_roi_proposals(self, frame):
        outputs = self.get_outputs()[0][self.output_blob]
        # outputs shape is [N_requests, 1, 1, N_max_faces, 7]
        
        frame_width = frame.shape[-1]
        frame_height = frame.shape[-2]
        
        results = []
        for output in outputs[0][0]:
            result = SsdDetector.Result(output, frame_width, frame_height, self.labels_map)
            if result.confidence < self.confidence_threshold:
                continue
            
            result.clip(frame_width, frame_height)
            
            results.append(result)
        
        return results
