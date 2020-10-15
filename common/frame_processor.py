import os
import sys
import time
import logging as log

import numpy as np

from openvino.inference_engine import IECore

class FrameProcessor:
    def __init__(self, args, used_devices):
        # 推論エンジン
        self.iecore = IECore()
        
        # puluginのロード
        start_time = time.time()
        log.info(f"Loading plugins for devices: {used_devices}")
        
        if 'CPU' in used_devices and not len(args.cpu_lib) == 0:
            log.info(f"Using CPU extensions library '{args.cpu_lib}'")
            assert os.path.isfile(cpu_ext), "Failed to open CPU extensions library"
            self.iecore.add_extension(args.cpu_lib, "CPU")
        
        if 'GPU' in used_devices and not len(args.gpu_lib) == 0:
            log.info(f"Using GPU extensions library '{args.gpu_lib}'")
            assert os.path.isfile(gpu_ext), "Failed to open GPU definitions file"
            self.iecore.set_config({"CONFIG_FILE": gpu_ext}, "GPU")
        
        log.info(f"Plugins are loaded.    loading time : {time.time()- start_time:.4f}sec")
        
        # パフォーマンス計測設定
        for d in used_devices:
            self.iecore.set_config({"PERF_COUNT": "YES" if args.perf_stats else "NO"}, d)
    
    # IR(Intermediate Representation ;中間表現)ファイル(.xml & .bin) の読み込み
    def load_model(self, model_path):
        start_time = time.time()                                # ロード時間測定用
        model_path = os.path.abspath(model_path)
        model_description_path = model_path
        model_weights_path = os.path.splitext(model_path)[0] + ".bin"
        log.info(f"    Loading the model from '{model_description_path}'")
        assert os.path.isfile(model_description_path), \
            f"Model description is not found at '{model_description_path}'"
        assert os.path.isfile(model_weights_path), \
            f"Model weights are not found at '{model_weights_path}'"
        
        model = self.iecore.read_network(model=model_description_path, weights=model_weights_path)
        log.info(f"    Model is loaded    loading time : {time.time()- start_time:.4f}sec")
        return model
    
    # ラベルファイルの読み込み
    def load_label(self, label_path):
        if not label_path:
            return None
        labels_map = None
        if os.path.isfile(label_path) :
            # ラベルファイルの読み込み
            with open(label_path, 'r') as f:
                labels_map = [x.strip() for x in f]
            log.info(f"    Labels file is loaded")
        else :
            log.info(f"    Labels file is not exist")
        return labels_map
