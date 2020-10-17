import sys
import numpy as np

import math

import pprint

try :
    import ngraph as ng
    # ngraph モジュールについては以下を参照
    # https://github.com/openvinotoolkit/openvino/blob/master/docs/nGraph_DG/nGraph_Python_API.md
except ImportError :
    # 2020.4以前は ngraph モジュールがない
    ng = None
 
# importのサーチパスを追加
sys.path.insert(0,"../common")

# ユーザ定義モジュール
from detector import Detector

# YOLOクラス
class YoloDetector(Detector):
    # 結果クラス
    class Result:
        def __init__(self, x, y, w, h, class_id, confidence, frame_width, frame_height, labels_map):
            self.class_id = class_id
            if labels_map :
                if len(labels_map) > class_id :
                    class_name = labels_map[class_id]
                else :
                    class_name = str(class_id)
            else :
                class_name = str(class_id)
            
            self.confidence = confidence
            self.class_name = class_name
            self.label =  f"{class_name}  {round(self.confidence * 100, 1)}%"
            self.overlapped = False
            
            # pixel座標に変換(x,yはボックスの中心座標)
            self.position = np.array((int((x - w / 2) * frame_width), int((y - h / 2) * frame_height))) # (x1, y1)
            self.size     = np.array((int(w           * frame_width), int(h           * frame_height))) # (w, h)
        
        # ROIを元画像範囲に収める
        def clip(self, width, height):
            min = [0, 0]
            max = [width, height]
            self.position[:] = np.clip(self.position, min, max)
            self.size[:] = np.clip(self.size, min, max)
        
        def __str__(self):
            return f"position:{self.position}, size:{self.size}, confidence:{self.confidence}, class_id:{self.class_id}, class_name:{self.class_name}, overlapped:{self.overlapped}"
    
    # YOLOモデルパラメータクラス
    class YoloParams:
        # ------------------------------------------- Extracting layer parameters ------------------------------------------
        # Magic numbers are copied from yolo samples
        def __init__(self, param, side):
            # print("**** param ****")
            # pprint.pprint(param)
            
            self.side    = side
            
            if 'num'     not in param :
                # 定義されていなかったときのデフォルト
                self.num     =  3
            elif type(param['num']) is str :
                # model.layers[layer_name].params 経由で取得した場合
                self.num     =  int(param['num'])
            else :
                # nodes[layer_name]._get_attributes() 経由で取得した場合
                self.num     =  param['num']
            
            if 'coords'     not in param :
                self.coords     =  4
            elif type(param['coords']) is str :
                self.coords     =  int(param['coords'])
            else :
                self.coords     =  param['coords']
            
            if 'classes'     not in param :
                self.classes     =  80
            elif type(param['classes']) is str :
                self.classes     =  int(param['classes'])
            else :
                self.classes     =  param['classes']
            
            if 'anchors' not in param :
                self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0]
            elif type(param['anchors']) is str :
                self.anchors = [float(a) for a in param['anchors'].split(',')]
            else :
                self.anchors = param['anchors']
            
            isYoloV3 = False
            if 'mask' not in param:
                pass
            elif type(param['mask']) is str :
                if ',' in param['mask'] :               # 空のmaskキーがあったときの対策
                    isYoloV3 = True
                    mask = [int(idx) for idx in param['mask'].split(',')]
            else :
                if len(param['mask']) > 0 :               # 空のmaskキーがあったときの対策
                    isYoloV3 = True
                    mask = param['mask']
            
            if isYoloV3 :
                self.num = len(mask)
                maskedAnchors = []
                for idx in mask:
                    maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
                self.anchors = maskedAnchors
            
            self.isYoloV3 = isYoloV3
            
            # print("**** self ****")
            # pprint.pprint(vars(self))
    
    def __init__(self, model, confidence_threshold=0.5, iou_threshold=0.4):
        super(YoloDetector, self).__init__(model)
        # パラメータチェック
        assert 0.0 <= confidence_threshold and confidence_threshold <= 1.0, \
            "Confidence threshold is expected to be in range [0; 1]"
        assert 0.0 <= iou_threshold and iou_threshold <= 1.0, \
            "Intersection over union threshold is expected to be in range [0; 1]"
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        if hasattr(model, "input_info") :
            # 2021.1 以降用
            # このプログラムは1入力のモデルのみサポートしているので、チェック
            assert len(model.input_info) == 1, "Expected 1 input blob"
            self.input_blob =  next(iter(model.input_info))
            self.input_shape = model.input_info[self.input_blob].input_data.shape
        else :
            # 2020.4 以前用
            # このプログラムは1入力のモデルのみサポートしているので、チェック
            assert len(model.inputs) == 1, "Expected 1 input blob"
            self.input_blob =  next(iter(model.inputs))
            self.input_shape = model.inputs[self.input_blob].shape
        
        self.layer_params = {}
        '''
        次の行で2021.1で以下のwarningが出る対策
        DeprecationWarning: 'layers' property of IENetwork class is deprecated. 
        For iteration over network please use get_ops()/get_ordered_ops() methods from nGraph Python API
        '''
        if ng :
            # ng が定義されてたら → 2021.1以降 
            function = ng.function_from_cnn(model)
            nodes = { n.friendly_name : n for n in function.get_ops() }
            self.layer_params_aaa = {}
            for layer_name in model.outputs.keys() :
                # print(layer_name)
                output_shapes = list(nodes[layer_name].inputs()[0].get_source_output().get_node().shape)
                output_params = nodes[layer_name]._get_attributes()
                
                assert output_shapes[2] == output_shapes[3], \
                        f"Invalid size of output blob. It sould be in NCHW layout and height should be equal to width. " \
                         "Current height = {output_shapes[2]}, " \
                         "current width = {output_shapes[3]}"
                 
                self.layer_params[layer_name]  = self.YoloParams(output_params, output_shapes[2])
        else :
            layers = model.layers
            for layer_name in model.outputs.keys() :
                # print(layer_name)
                output_shapes = layers[layers[layer_name].parents[0]].out_data[0].shape
                output_params = layers[layer_name].params
                
                assert output_shapes[2] == output_shapes[3], \
                        f"Invalid size of output blob. It sould be in NCHW layout and height should be equal to width. " \
                         "Current height = {output_shapes[2]}, " \
                         "current width = {output_shapes[3]}"
                 
                self.layer_params[layer_name]  = self.YoloParams(output_params, output_shapes[2])
        
    # 前処理
    def preprocess(self, frame):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        assert frame.shape[0] == 1
        assert frame.shape[1] == 3
        input = self.resize_input(frame, self.input_shape)
        return input
    
    # ROIの重なり比率の計算
    def get_overlap_ratio(self, result_1, result_2) :
        xmin_1 , ymin_1  = result_1.position
        xmax_1 , ymax_1  = result_1.position + result_1.size
        xsize_1, ysize_1 = result_1.size
        xmin_2 , ymin_2 = result_2.position
        xmax_2 , ymax_2 = result_2.position + result_2.size
        xsize_2, ysize_2 = result_1.size
        
        width_of_overlap_area  = min(xmax_1, xmax_2) - max(xmin_1, xmin_2)
        height_of_overlap_area = min(ymax_1, ymax_2) - max(ymin_1, ymin_2)
        
        if width_of_overlap_area < 0 or height_of_overlap_area < 0:
            area_of_overlap = 0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        
        box_1_area = xsize_1 * ysize_1 
        box_2_area = xsize_2 * ysize_2 
        area_of_union = box_1_area + box_2_area - area_of_overlap       # result_1 と result_2 の 合計面積
        
        if area_of_union == 0:
            return 0
        return area_of_overlap / area_of_union
    
    # 認識処理開始
    def start_async(self, frame):
        input = self.preprocess(frame)
        self.enqueue(input)
    
    # 認識処理要求をキューイング
    def enqueue(self, input):
        feed_dict = {self.input_blob: input}
        return super(YoloDetector, self).enqueue(feed_dict)
    
    # 認識結果取得
    def get_roi_proposals(self, frame):
        results = []
        for layer_name, out_blob in  self.get_outputs()[0].items():
            params = self.layer_params[layer_name]
            side_square = params.side * params.side
            
            size_w = self.input_shape[2] if params.isYoloV3 else params.side
            size_h = self.input_shape[3] if params.isYoloV3 else params.side
            
            predictions = out_blob.flatten()                # 配列を一次元化
            
            # predictionsをそれぞれのパラメータの配列にばらす
            x_table = np.empty([params.num,                 params.side, params.side], dtype=np.float32)
            y_table = np.empty([params.num,                 params.side, params.side], dtype=np.float32)
            w_table = np.empty([params.num,                 params.side, params.side], dtype=np.float32)
            h_table = np.empty([params.num,                 params.side, params.side], dtype=np.float32)
            s_table = np.empty([params.num,                 params.side, params.side], dtype=np.float32)
            c_table = np.empty([params.num, params.classes, params.side, params.side], dtype=np.float32)
            
            for n in range(params.num):
                pred_1 = predictions[side_square * ((params.coords + params.classes + 1) * (n + 0) + 0                )  \
                                   : side_square * ((params.coords + params.classes + 1) * (n + 1) + 0                )  ]
                pred_2 = predictions[side_square * ((params.coords + params.classes + 1) * (n + 0) + params.coords    ) \
                                   : side_square * ((params.coords + params.classes + 1) * (n + 1) + params.coords    ) ]
                pred_3 = predictions[side_square * ((params.coords + params.classes + 1) * (n + 0) + params.coords + 1) \
                                   : side_square * ((params.coords + params.classes + 1) * (n + 1) + params.coords + 1) *  params.classes ]
                
                x_table[n][:][:]    = pred_1[side_square * 0 : side_square * 1             ].reshape([1,                 params.side, params.side])
                y_table[n][:][:]    = pred_1[side_square * 1 : side_square * 2             ].reshape([1,                 params.side, params.side])
                w_table[n][:][:]    = pred_1[side_square * 2 : side_square * 3             ].reshape([1,                 params.side, params.side])
                h_table[n][:][:]    = pred_1[side_square * 3 : side_square * 4             ].reshape([1,                 params.side, params.side])
                s_table[n][:][:]    = pred_2[side_square * 0 : side_square * 1             ].reshape([1,                 params.side, params.side])
                c_table[n][:][:][:] = pred_3[side_square * 0 : side_square * params.classes].reshape([1, params.classes, params.side, params.side])
            
            # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
            for row in range(params.side):
                for col in range(params.side):
                    for n in range(params.num):
                        scale = s_table[n][row][col]
                        if scale < self.confidence_threshold:
                            # スコアが低いものはスキップ
                            continue
                        
                        x     = (col + x_table[n][row][col]) / params.side
                        y     = (row + y_table[n][row][col]) / params.side
                        try:
                            w_exp = math.exp(w_table[n][row][col])
                            h_exp = math.exp(h_table[n][row][col])
                            w = w_exp * params.anchors[2 * n]     / size_w
                            h = h_exp * params.anchors[2 * n + 1] / size_h
                        except OverflowError:
                            # 幅/高さが計算でオーバフローしたらスキップ
                            continue
                        
                        for clss in range(params.classes):
                            confidence = scale * c_table[n][clss][row][col]
                            if confidence < self.confidence_threshold:
                                # スコアが低いものはスキップ
                                continue
                            
                            result = YoloDetector.Result(x, y, w, h, clss, confidence, frame.shape[3], frame.shape[2], self.labels_map)
                            
                            result.clip(frame.shape[3], frame.shape[2])
                            
                            results.append(result)
        '''
        print('#### result start #####################')
        for result in results :
            print(str(result))
        print('#### result end   #####################')
        '''
        
        results = sorted(results, key=lambda result : result.confidence, reverse=True)
        for i in range(len(results)):
            if results[i].overlapped :
                continue
            for j in range(i + 1, len(results)):
                if self.get_overlap_ratio(results[i], results[j]) > self.iou_threshold:
                    results[j].overlapped = True
        
        '''
        print('#### result start #####################')
        for result in results :
            print(str(result))
        print('#### result end   #####################')
        '''
        
        results = [result for result in results if not result.overlapped]
        
        '''
        print('#### result start #####################')
        for result in results :
            print(str(result))
        print('#### result end   #####################')
        '''
        
        return results
