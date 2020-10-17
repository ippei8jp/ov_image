import time
import logging as log
import pprint

import cv2
import numpy as np

# 検出器の基底クラス
class Detector(object):
    def __init__(self, model):
        self.model = model
        self.device_model = None
        
        self.max_requests = 0
        self.active_requests = 0
        
        self.clear()
    
    # モデルで使用しているレイヤがサポートされているかの確認
    def check_model_support(self, net, device, iecore):
        if device == "CPU":
            # サポートしているレイヤの一覧
            supported_layers = iecore.query_network(net, "CPU")
            """
            # For debug
            print("=================================")
            pprint.pprint(sorted(list(supported_layers.keys())))
            print("=================================")
            pprint.pprint(sorted(list(net.layers.keys())))
            print("=================================")
            """
            # ******** net.layers は将来廃止されるので代替処理が必要 ********
            # nGraph経由で取得できるはず。yolo/yolo_detector.py を参照
            
            # netで使用されているレイヤでサポートしているレイヤの一覧にないもの
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            # サポートされていないレイヤがある？
            if len(not_supported_layers) != 0:
                # エラー例外をスロー
                log.error(f"The following layers are not supported " \
                          f"by the plugin for the specified device {device} :\n" \
                          f"    {', '.join(not_supported_layers)}")
                log.error( "Please try to specify cpu extensions " \
                           "library path in the command line parameters using the '-l' parameter")
                raise NotImplementedError("Some layers are not supported on the device")
    
    # モデルのロード
    def deploy(self, device, iecore, queue_size=1):
        start_time = time.time()                                    # ロード時間測定用
        log.info(f"    Loading the network to {device}")
        self.max_requests = queue_size
        # 2021.1でエラーになるのでとりあえずコメントアウト
        # self.check_model_support(self.model, device, iecore)
        self.device_model = iecore.load_network(network=self.model, num_requests=self.max_requests, device_name=device)
        self.model = None
        log.info(f"    network is loaded    loading time : {time.time()- start_time:.4f}sec")
    
    # 入力データのリサイズ
    def resize_input(self, frame, target_shape):
        assert len(frame.shape) == len(target_shape), \
            f"Expected a frame with {len(target_shape)} dimensions, but got {len(frame.shape)}"
        
        assert frame.shape[0] == 1, "Only batch size 1 is supported"
        n, c, h, w = target_shape
        
        input = frame[0]
        if not np.array_equal(target_shape[-2:], frame.shape[-2:]):
            input = input.transpose((1, 2, 0)) # to HWC
            input = cv2.resize(input, (w, h))
            input = input.transpose((2, 0, 1)) # to CHW
        
        return input.reshape((n, c, h, w))
    
    # 認識処理要求をキューイング
    def enqueue(self, input):
        self.clear()
    
        if self.max_requests <= self.active_requests:
            log.warning("Processing request rejected - too many requests")
            return False
    
        self.device_model.start_async(self.active_requests, input)
        self.active_requests += 1
        return True
    
    # 処理終了待ち
    def wait(self):
        if self.active_requests <= 0:
            return
        
        self.perf_stats = [None, ] * self.active_requests
        self.outputs = [None, ] * self.active_requests
        for i in range(self.active_requests):
            self.device_model.requests[i].wait()
            # 2021.1でのワーニング対策('outputs' property of InferRequest is deprecated.)
            if hasattr(self.device_model.requests[i], "output_blobs") :
                self.outputs[i] = {}                                                    # 2021.1以降
                for ob_key, ob in self.device_model.requests[i].output_blobs.items() :
                    self.outputs[i][ob_key] = ob.buffer
            else :
                self.outputs[i] = self.device_model.requests[i].outputs                 # 2020.4以前
            self.perf_stats[i] = self.device_model.requests[i].get_perf_counts()
        
        self.active_requests = 0
    
    # 結果取得
    def get_outputs(self):
        self.wait()
        return self.outputs
    
    # 検出器初期化
    def clear(self):
        self.perf_stats = []
        self.outputs = []
    
    # パフォーマンスステータスの取得
    def get_performance_stats(self):
        return self.perf_stats
